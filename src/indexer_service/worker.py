import uuid

from src.common.config import EMBEDDING_URL
from src.common.db import init_db, upsert_document
from src.common.http import post_json
from src.common.vector_store import upsert as vs_upsert

# Ensure DB tables exist when worker module loads
init_db()

def index_document(text: str, source: str = "user"):
    if not text:
        return {"ok": False, "error": "empty"}
    # Generate deterministic id for identical content (MVP: uuid4)
    doc_id = str(uuid.uuid4())
    # Embed
    emb_resp = post_json(f"{EMBEDDING_URL}/v1/embed", {"texts": [text]})
    embedding = emb_resp["embeddings"][0]
    meta = {"source": source}
    # Upsert vector and metadata
    vs_upsert(id=doc_id, embedding=embedding, document=text, metadata=meta)
    upsert_document(doc_id, text, meta)
    return {"ok": True, "id": doc_id}
