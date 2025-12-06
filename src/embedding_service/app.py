from threading import Lock

from fastapi import FastAPI

from src.common.schemas import EmbedRequest, EmbedResponse

app = FastAPI(title="Embedding Service")
_model = None
_lock = Lock()

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    model = get_model()
    vectors = model.encode(req.texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
    return EmbedResponse(embeddings=vectors)
