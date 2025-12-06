import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8002")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8003")
RENDERER_URL = os.getenv("RENDERER_URL", "http://localhost:8004")
INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8005")

METADATA_DB = os.getenv("METADATA_DB", "./data/metadata.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")

# Defaults
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
TOP_K = int(os.getenv("TOP_K", "5"))
