from fastapi import FastAPI, HTTPException

from src.common.config import INGESTION_URL, RENDERER_URL, SEARCH_URL
from src.common.db import init_db  # ensure tables exist
from src.common.http import post_json
from src.common.schemas import (IngestRequest, IngestResponse, QueryRequest,
                                QueryResponse, RenderRequest, RenderResponse,
                                SearchRequest, SearchResponse)

app = FastAPI(title="API Gateway")

# Initialize DB schema at startup
@app.on_event("startup")
def _startup_init_db():
    try:
        init_db()
    except Exception as e:
        # don't block startup; queries/ingest will still error with clear messages
        print(f"[api_gateway] DB init failed: {e}")

@app.get("/health")
def health():
    return {"ok": True}

# New: status endpoint that checks downstream services
@app.get("/v1/status")
def status():
    errors = []
    def check(url: str):
        try:
            return post_json(f"{url}/health", {})
        except Exception as e:
            errors.append(str(e))
            return {"ok": False, "error": str(e)}
    return {
        "api_gateway": {"ok": True},
        "search_service": check(SEARCH_URL),
        "renderer": check(RENDERER_URL),
        "ingestion_service": check(INGESTION_URL),
        "errors": errors,
    }

@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.text and not req.repo_url:
        raise HTTPException(status_code=400, detail="Provide text or repo_url")
    data = post_json(f"{INGESTION_URL}/v1/ingest", req.model_dump())
    return IngestResponse(**data)

@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # Call search with defensive handling
    try:
        search = post_json(
            f"{SEARCH_URL}/v1/search",
            SearchRequest(query=req.query, k=req.k).model_dump()
        )
    except Exception as e:
        # Upstream not reachable or error
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    sr = SearchResponse(**search)

    # If no hits, return a friendly answer instead of failing
    if not sr.hits:
        return QueryResponse(
            answer="No results found. Try different keywords or ingest more content via /v1/ingest.",
            hits=[]
        )

    # Call renderer with defensive handling
    try:
        render = post_json(
            f"{RENDERER_URL}/v1/render",
            RenderRequest(query=req.query, hits=sr.hits).model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Renderer service error: {e}")

    rr = RenderResponse(**render)
    return QueryResponse(answer=rr.answer, hits=sr.hits)
