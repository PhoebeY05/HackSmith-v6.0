from fastapi import FastAPI, HTTPException
from redis import Redis
from rq import Queue

from src.common.config import REDIS_URL
from src.common.schemas import IngestRequest, IngestResponse

app = FastAPI(title="Ingestion Service")
redis = Redis.from_url(REDIS_URL)
queue = Queue("index", connection=redis)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if not req.text and not req.repo_url:
        raise HTTPException(status_code=400, detail="Provide text or repo_url")
    # MVP: only text; repo_url placeholder
    payload = {"text": req.text or "", "source": req.source}
    job = queue.enqueue("src.indexer_service.worker.index_document", kwargs=payload)
    return IngestResponse(job_id=job.get_id())
