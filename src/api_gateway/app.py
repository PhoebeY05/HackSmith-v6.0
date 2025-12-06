import uuid

from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel, Field

from src.common.config import (EMBEDDING_URL, INGESTION_URL, RENDERER_URL,
                               SEARCH_URL)
from src.common.db import init_db  # ensure tables exist
from src.common.db import upsert_document
from src.common.http import post_json
from src.common.schemas import (EmbedRequest, EmbedResponse, QueryRequest,
                                QueryResponse, RenderRequest, RenderResponse,
                                SearchRequest, SearchResponse)
from src.common.vector_store import upsert as vs_upsert


# Define a JSON schema for structured ingest
class StructuredIngest(BaseModel):
    title: str = Field(..., min_length=1)
    problem: str = Field(..., min_length=1)
    solution: str = Field(..., min_length=1)
    source: str = Field(default="entry")

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

@app.post("/v1/ingest")
def ingest(req: StructuredIngest):
    # Validate and normalize
    title = req.title.strip()
    problem = req.problem.strip()
    solution = req.solution.strip()
    if not title or not problem or not solution:
        raise HTTPException(status_code=400, detail="Provide title, problem, and solution")

    # Build combined searchable text
    combined = f"Title: {title}\nProblem: {problem}\nSolution: {solution}"

    # Embed combined text
    try:
        emb = post_json(f"{EMBEDDING_URL}/v1/embed", EmbedRequest(texts=[combined]).model_dump())
        er = EmbedResponse(**emb)
        embedding = er.embeddings[0]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")

    # Upsert into vector store and metadata DB
    doc_id = str(uuid.uuid4())
    metadata = {
        "source": req.source,
        "title": title,
        "problem": problem[:500],
    }
    try:
        vs_upsert(id=doc_id, embedding=embedding, document=combined, metadata=metadata)
        upsert_document(doc_id, combined, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}")

    return {"job_id": doc_id, "id": doc_id, "ok": True}

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
