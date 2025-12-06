import chromadb
from chromadb.config import Settings

from .config import CHROMA_PATH, COLLECTION_NAME

_client = None
_collection = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))
    return _client

def get_collection(name: str = COLLECTION_NAME):
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return _collection

def upsert(id: str, embedding: list[float], document: str, metadata: dict | None = None):
    col = get_collection()
    col.upsert(ids=[id], embeddings=[embedding], documents=[document], metadatas=[metadata or {}])

def query(embedding: list[float], k: int = 5):
    col = get_collection()
    res = col.query(query_embeddings=[embedding], n_results=k, include=["documents", "metadatas", "distances"])
    # Normalize outputs
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    # Convert distance to score (cosine): score = 1 - distance
    hits = []
    for i, _id in enumerate(ids):
        hits.append({
            "id": _id,
            "text": docs[i],
            "metadata": metas[i] or {},
            "score": float(1.0 - distances[i]) if distances[i] is not None else 0.0
        })
    return hits
