from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: Optional[str] = None
    repo_url: Optional[str] = None
    source: Optional[str] = Field(default="user", description="Source label")

class IngestResponse(BaseModel):
    job_id: str

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    hits: List[SearchHit]

class RenderRequest(BaseModel):
    query: str
    hits: List[SearchHit]

class RenderResponse(BaseModel):
    answer: str

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    hits: List[SearchHit]
