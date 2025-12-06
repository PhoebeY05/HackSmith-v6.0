from fastapi import FastAPI

from src.common.config import EMBEDDING_URL, TOP_K
from src.common.http import post_json
from src.common.schemas import (EmbedRequest, EmbedResponse, SearchHit,
                                SearchRequest, SearchResponse)

app = FastAPI(title="Search & Ranking Service")

from src.common.vector_store import query as vs_query  # after app creation


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/search", response_model=SearchResponse)
def search(req: SearchRequest):
    k = req.k or TOP_K
    emb = post_json(f"{EMBEDDING_URL}/v1/embed", EmbedRequest(texts=[req.query]).model_dump())
    er = EmbedResponse(**emb)
    hits = vs_query(embedding=er.embeddings[0], k=k)
    return SearchResponse(hits=[SearchHit(**h) for h in hits])
