from fastapi import FastAPI, HTTPException

from src.common.config import INGESTION_URL, RENDERER_URL, SEARCH_URL
from src.common.http import post_json
from src.common.schemas import (IngestRequest, IngestResponse, QueryRequest,
                                QueryResponse, RenderRequest, RenderResponse,
                                SearchRequest, SearchResponse)

app = FastAPI(title="API Gateway")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.text and not req.repo_url:
        raise HTTPException(status_code=400, detail="Provide text or repo_url")
    data = post_json(f"{INGESTION_URL}/v1/ingest", req.model_dump())
    return IngestResponse(**data)

@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    search = post_json(f"{SEARCH_URL}/v1/search", SearchRequest(query=req.query, k=req.k).model_dump())
    sr = SearchResponse(**search)
    render = post_json(f"{RENDERER_URL}/v1/render", RenderRequest(query=req.query, hits=sr.hits).model_dump())
    rr = RenderResponse(**render)
    return QueryResponse(answer=rr.answer, hits=sr.hits)
