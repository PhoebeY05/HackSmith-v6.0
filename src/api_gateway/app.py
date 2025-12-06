import uuid
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel, Field

from src.common.config import (EMBEDDING_URL, INGESTION_URL, RENDERER_URL,
                               SEARCH_URL)
from src.common.db import init_db  # ensure tables exist
from src.common.db import (add_note, list_notes, list_notes_for_doc,
                           search_notes_by_tag, upsert_document)
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
    notes: list["DecisionNoteIn"] = Field(default_factory=list)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Login fails on mobile",
                    "problem": "Users cannot log in due to stale sessions on mobile devices.",
                    "solution": "Reset session on login and rotate tokens; shorten mobile session TTL.",
                    "source": "entry",
                    "notes": [
                        {
                            "title": "Tradeoffs",
                            "content": "Shorter TTL increases re-auth frequency; mitigated by MFA remember-device.",
                            "kind": "ADR",
                            "tags": ["auth","session","ttl"]
                        },
                        {
                            "title": "Incident RCA",
                            "content": "Batch job held locks causing session store delays; fixed by reducing transaction scope.",
                            "kind": "Incident",
                            "tags": ["ops","locks","session-store"]
                        }
                    ]
                }
            ]
        }
    }

# Add a decision note (ADR/Incident/RCA)
class DecisionNoteIn(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)  # explanations, tradeoffs, root cause
    kind: str = Field(default="ADR")
    tags: list[str] = Field(default=[])
    doc_id: str | None = None  # populated automatically when sent inside ingest
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "DB deadlock tradeoffs",
                    "content": "Retry logic can hide modeling issues; use shorter transactions and proper indexing.",
                    "kind": "ADR",
                    "tags": ["db","deadlock","transactions"]
                }
            ]
        }
    }

# Pydantic forward reference resolution
StructuredIngest.model_rebuild()

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

@app.post("/v1/notes/add")
def notes_add(req: DecisionNoteIn):
    note_id = str(uuid.uuid4())
    try:
        add_note(
            note_id=note_id,
            title=req.title.strip(),
            content=req.content.strip(),
            kind=req.kind.strip(),
            tags=[t.strip() for t in req.tags],
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            doc_id=(req.doc_id.strip() if req.doc_id else None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save note: {e}")
    return {"id": note_id, "ok": True}

@app.get("/v1/notes/list")
def notes_list(limit: int = 20):
    try:
        rows = list_notes(limit=limit)
        return {"items": [
            {"id": r.id, "title": r.title, "kind": r.kind, "tags": r.tags or [], "created_at": r.created_at}
            for r in rows
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list notes: {e}")

@app.get("/v1/notes/search")
def notes_search(tag: str, limit: int = 20):
    try:
        rows = search_notes_by_tag(tag=tag, limit=limit)
        return {"items": [
            {"id": r.id, "title": r.title, "kind": r.kind, "tags": r.tags or [], "created_at": r.created_at}
            for r in rows
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search notes: {e}")

@app.get("/v1/notes/match")
def notes_match(keyword: str, limit: int = 10):
    try:
        rows = match_notes(keyword=keyword, limit=limit)
        return {"items": [
            {
                "id": r.id,
                "title": r.title,
                "kind": r.kind,
                "tags": r.tags or [],
                "created_at": r.created_at,
                "snippet": (r.content[:200] + ("..." if len(r.content) > 200 else "")),
            }
            for r in rows
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to match notes: {e}")

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

    # New: save any provided notes and link them to this document
    note_ids: list[str] = []
    for n in (req.notes or []):
        nid = str(uuid.uuid4())
        try:
            # Save note to metadata DB
            add_note(
                note_id=nid,
                title=n.title.strip(),
                content=n.content.strip(),
                kind=(n.kind or "ADR").strip(),
                tags=[t.strip() for t in (n.tags or [])],
                created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                doc_id=doc_id,
            )
            # Embed and index the note for search
            note_text = f"Note: {n.title.strip()}\n{n.content.strip()}"
            emb_note = post_json(f"{EMBEDDING_URL}/v1/embed", EmbedRequest(texts=[note_text]).model_dump())
            er_note = EmbedResponse(**emb_note)
            vs_upsert(
                id=nid,
                embedding=er_note.embeddings[0],
                document=note_text,
                metadata={
                    "type": "note",
                    "note_kind": (n.kind or "ADR").strip(),
                    "title": n.title.strip(),
                    "doc_id": doc_id,
                    "source": "note",
                    # New: include a snippet of the note content for filtering/display
                    "content_snippet": (n.content.strip()[:500])
                }
            )
            note_ids.append(nid)
        except Exception as e:
            note_ids.append(f"error:{e}")

    return {"job_id": doc_id, "id": doc_id, "ok": True, "note_ids": note_ids}

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

    # New: filter by minimum score and keyword presence
    min_score = 0.25
    q_tokens = {t for t in req.query.lower().split() if len(t) > 2}
    def passes(h):
        text = (h.text or "").lower()
        # basic token match: any query token present in text
        token_match = any(t in text for t in q_tokens) if q_tokens else True
        return h.score >= min_score and token_match

    filtered_hits = [h for h in sr.hits if passes(h)]
    if not filtered_hits:
        # fallback: keep only top hit if everything filtered out
        filtered_hits = sr.hits[:1]

    # Call renderer with defensive handling
    try:
        render = post_json(
            f"{RENDERER_URL}/v1/render",
            RenderRequest(query=req.query, hits=filtered_hits).model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Renderer service error: {e}")

    rr = RenderResponse(**render)
    return QueryResponse(answer=rr.answer, hits=filtered_hits)

# New: list notes for a given document id (provenance-aware)
@app.get("/v1/notes/by_doc")
def notes_by_doc(doc_id: str):
    try:
        rows = list_notes_for_doc(doc_id=doc_id)
        return {"items": [
            {"id": r.id, "title": r.title, "kind": r.kind, "tags": r.tags or [], "created_at": r.created_at}
            for r in rows
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list notes for doc: {e}")

# Query decision notes only (filters hits with metadata.type == 'note')
@app.post("/v1/query/notes", response_model=QueryResponse)
def query_notes(req: QueryRequest):
    try:
        search = post_json(
            f"{SEARCH_URL}/v1/search",
            SearchRequest(query=req.query, k=req.k).model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    sr = SearchResponse(**search)
    note_hits = [h for h in sr.hits if (h.metadata or {}).get("type") == "note"]

    if not note_hits:
        return QueryResponse(
            answer="No decision notes found. Try different keywords or add notes via /v1/notes/add or in /v1/ingest.",
            hits=[]
        )

    # Filter using both vector text and metadata content_snippet
    min_score = 0.20
    q_tokens = {t for t in req.query.lower().split() if len(t) > 2}
    def passes(h):
        text = (h.text or "").lower()
        meta = h.metadata or {}
        content_meta = (meta.get("content_snippet", "") or "").lower()
        token_match_text = any(t in text for t in q_tokens) if q_tokens else True
        token_match_meta = any(t in content_meta for t in q_tokens) if q_tokens else True
        token_match = token_match_text or token_match_meta
        return h.score >= min_score and token_match

    filtered = [h for h in note_hits if passes(h)] or note_hits[:1]

    try:
        render = post_json(
            f"{RENDERER_URL}/v1/render",
            RenderRequest(query=req.query, hits=filtered).model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Renderer service error: {e}")

    rr = RenderResponse(**render)
    return QueryResponse(answer=rr.answer, hits=filtered)
