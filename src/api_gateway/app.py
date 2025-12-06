import uuid
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text as sa_text

from src.common.config import (CHROMA_PATH, COLLECTION_NAME, EMBEDDING_URL,
                               INGESTION_URL, RENDERER_URL, SEARCH_URL)
# Add import for note matching used by the summarizer
from src.common.db import init_db  # ensure tables exist
from src.common.db import (SessionLocal, add_note, list_notes,
                           list_notes_for_doc, match_notes,
                           search_notes_by_tag, upsert_document)
from src.common.http import post_json
from src.common.schemas import (EmbedRequest, EmbedResponse, QueryRequest,
                                QueryResponse, RenderRequest, RenderResponse,
                                SearchRequest, SearchResponse)
from src.common.vector_store import get_client  # use shared client
from src.common.vector_store import upsert as vs_upsert


# Define a JSON schema for structured ingest
class StructuredIngest(BaseModel):
    title: str = Field(..., min_length=1)
    problem: str = Field(..., min_length=1)
    solution: str = Field(..., min_length=1)
    source: str = Field(default="entry")
    notes: list["DecisionNoteIn"] = Field(default_factory=list)
    # New: optional code snippet to support copy-ready blocks
    code_snippet: str | None = Field(default=None, description="Optional code snippet relevant to the solution")
    # New: security-specific optional metadata
    ioc_type: str | None = Field(default=None, description="ip | domain | hash | url | registry | malware family")
    threat_level: str | None = Field(default=None, description="low | medium | high")
    incident_type: str | None = Field(default=None, description="phishing | malware | misconfiguration | vuln")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Login fails on mobile",
                    "problem": "Users cannot log in due to stale sessions on mobile devices.",
                    "solution": "Reset session on login and rotate tokens; shorten mobile session TTL.",
                    "source": "entry",
                    "code_snippet": "useEffect(() => { rotateTokens(); resetSession(); }, []);",
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
    # Example: GET /health
    # curl -s http://localhost:8000/health
    return {"ok": True}

# New: status endpoint that checks downstream services
@app.get("/v1/status")
def status():
    # Example: GET /v1/status
    # curl -s http://localhost:8000/v1/status
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
    # Example: POST /v1/notes/add
    # curl -s -X POST http://localhost:8000/v1/notes/add \
    #   -H "Content-Type: application/json" \
    #   -d '{ "title":"DB deadlock tradeoffs","content":"Retry logic...","kind":"ADR","tags":["db","deadlock"] }'
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
    # Example: GET /v1/notes/list?limit=10
    # curl -s "http://localhost:8000/v1/notes/list?limit=10"
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
    # Example: GET /v1/notes/search?tag=auth&limit=5
    # curl -s "http://localhost:8000/v1/notes/search?tag=auth&limit=5"
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
    # Example: GET /v1/notes/match?keyword=session&limit=10
    # curl -s "http://localhost:8000/v1/notes/match?keyword=session&limit=10"
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

# Helpers: normalize and extract security artifacts from free-form text
def _normalize_security_artifacts(text: str) -> dict:
    import re
    s = (text or "").strip()
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", s)
    domains = re.findall(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", s)
    urls = re.findall(r"https?://[^\s]+", s)
    hashes = re.findall(r"\b[a-f0-9]{32,64}\b", s, flags=re.IGNORECASE)
    registry = re.findall(r"\bHKEY_[A-Z_]+\\[^\s]+", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    log_snippet = "\n".join(lines[:20])
    def _dedupe(arr): return sorted(list(dict.fromkeys(arr)))
    return {
        "ips": _dedupe(ips),
        "domains": _dedupe([d for d in domains if d not in ips]),
        "urls": _dedupe(urls),
        "hashes": _dedupe(hashes),
        "registry": _dedupe(registry),
        "log_snippet": log_snippet[:2000]
    }

@app.post("/v1/ingest")
def ingest(req: StructuredIngest):
    # Example: POST /v1/ingest
    # curl -s -X POST http://localhost:8000/v1/ingest \
    #   -H "Content-Type: application/json" \
    #   -d '{ "title":"Login fails on mobile","problem":"Stale sessions","solution":"Reset session on login","source":"entry",
    #         "code_snippet":"authRouter.post(...)", "ioc_type":"url","threat_level":"medium","incident_type":"phishing",
    #         "notes":[{"title":"Tradeoffs","content":"Shorter TTL...","kind":"ADR","tags":["auth","ttl"]}] }'
    # Validate and normalize
    title = req.title.strip()
    problem = req.problem.strip()
    solution = req.solution.strip()
    if not title or not problem or not solution:
        raise HTTPException(status_code=400, detail="Provide title, problem, and solution")

    # Build combined searchable text
    combined = f"Title: {title}\nProblem: {problem}\nSolution: {solution}"
    # New: optionally include a trimmed code snippet to improve retrieval (kept short to avoid noise)
    if (req.code_snippet or "").strip():
        snippet_trim = _clean_snippet(req.code_snippet.strip(), max_len=300)
        combined += f"\nCode:\n{snippet_trim}"
    # New: security artifact normalization (augment indexed text minimally)
    sec_meta = {}
    # If user pasted artifacts into problem/solution fields, extract and attach
    artifacts = _normalize_security_artifacts(f"{title}\n{problem}\n{solution}\n{req.code_snippet or ''}")
    if any(artifacts[k] for k in ("ips","domains","urls","hashes","registry")) or artifacts["log_snippet"]:
        combined += "\nArtifacts:\n"
        if artifacts["ips"]:
            combined += "IPs: " + ", ".join(artifacts["ips"][:10]) + ("\n" if artifacts["ips"] else "")
        if artifacts["domains"]:
            combined += "Domains: " + ", ".join(artifacts["domains"][:10]) + ("\n" if artifacts["domains"] else "")
        if artifacts["urls"]:
            combined += "URLs: " + ", ".join(artifacts["urls"][:10]) + ("\n" if artifacts["urls"] else "")
        if artifacts["hashes"]:
            combined += "Hashes: " + ", ".join(artifacts["hashes"][:10]) + ("\n" if artifacts["hashes"] else "")
        if artifacts["registry"]:
            combined += "Registry: " + ", ".join(artifacts["registry"][:5]) + ("\n" if artifacts["registry"] else "")
        if artifacts["log_snippet"]:
            combined += "Log:\n" + _clean_snippet(artifacts["log_snippet"], max_len=600) + "\n"
        # stash artifacts into metadata
        sec_meta = {
            "artifacts_ips": artifacts["ips"],
            "artifacts_domains": artifacts["domains"],
            "artifacts_urls": artifacts["urls"],
            "artifacts_hashes": artifacts["hashes"],
            "artifacts_registry": artifacts["registry"],
        }

    # Embed combined text
    try:
        emb = post_json(f"{EMBEDDING_URL}/v1/embed", EmbedRequest(texts=[combined]).model_dump())
        er = EmbedResponse(**emb)
        embedding = er.embeddings[0]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")

    # Upsert into vector store and metadata DB
    doc_id = str(uuid.uuid4())
    # Helper: serialize lists to compact comma-joined strings for Chroma metadata
    def _meta_list(name: str, vals: list[str] | None, max_items: int = 20) -> dict:
        v = (vals or [])
        if not v:
            return {}
        joined = ", ".join(v[:max_items])
        return {name: joined}

    metadata = {
        "source": req.source,
        "title": title,
        "problem": problem[:500],
        # New: persist raw code snippet in metadata for answer composition
        "code_snippet": (req.code_snippet.strip() if (req.code_snippet or "").strip() else None),
        # New: include security metadata if provided (scalars only)
        "ioc_type": (req.ioc_type.strip() if req.ioc_type else None),
        "threat_level": (req.threat_level.strip() if req.threat_level else None),
        "incident_type": (req.incident_type.strip() if req.incident_type else None),
        # DO NOT spread sec_meta here (it contains lists)
    }

    # Inject serialized artifacts (avoid list values in metadata)
    metadata.update(_meta_list("artifacts_ips", sec_meta.get("artifacts_ips")))
    metadata.update(_meta_list("artifacts_domains", sec_meta.get("artifacts_domains")))
    metadata.update(_meta_list("artifacts_urls", sec_meta.get("artifacts_urls")))
    metadata.update(_meta_list("artifacts_hashes", sec_meta.get("artifacts_hashes")))
    metadata.update(_meta_list("artifacts_registry", sec_meta.get("artifacts_registry")))

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
def query(req: QueryRequest, ioc_type: str | None = None, incident_type: str | None = None, threat_level: str | None = None):
    # Example: POST /v1/query (JSON body + optional filters as query params)
    # curl -s -X POST "http://localhost:8000/v1/query?incident_type=phishing&threat_level=high" \
    #   -H "Content-Type: application/json" -d '{ "query":"session ttl", "k":5 }'
    # Call search with defensive handling
    try:
        search = post_json(
            f"{SEARCH_URL}/v1/search",
            SearchRequest(query=req.query, k=req.k).model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    sr = SearchResponse(**search)

    # If no hits, return a friendly answer instead of failing
    if not sr.hits:
        return QueryResponse(answer="No results found. Try different keywords or ingest more content via /v1/ingest.", hits=[])

    # Filter by minimum score and keyword presence
    min_score = 0.25
    q_tokens = {t for t in req.query.lower().split() if len(t) > 2}
    def base_passes(h):
        text = (h.text or "").lower()
        token_match = any(t in text for t in q_tokens) if q_tokens else True
        return h.score >= min_score and token_match

    # Security metadata filters (optional)
    def sec_passes(h):
        meta = h.metadata or {}
        if ioc_type and (meta.get("ioc_type") or "").lower() != ioc_type.lower():
            return False
        if incident_type and (meta.get("incident_type") or "").lower() != incident_type.lower():
            return False
        if threat_level and (meta.get("threat_level") or "").lower() != threat_level.lower():
            return False
        return True

    filtered_hits = [h for h in sr.hits if base_passes(h) and sec_passes(h)]

    # Strict filtering: if nothing matches, do NOT fallback to top hit
    if not filtered_hits:
        return QueryResponse(
            answer="No results matched the applied filters. Try adjusting ioc_type, incident_type, or threat_level.",
            hits=[]
        )

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
    # Example: GET /v1/notes/by_doc?doc_id=abcd-1234
    # curl -s "http://localhost:8000/v1/notes/by_doc?doc_id=abcd-1234"
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
    # Example: POST /v1/query/notes
    # curl -s -X POST http://localhost:8000/v1/query/notes \
    #   -H "Content-Type: application/json" -d '{ "query":"deadlock tradeoffs", "k":5 }'
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

# Helpers: clean snippet and compose neat answer blocks
def _clean_snippet(text: str, max_len: int = 600) -> str:
    if not text:
        return ""
    s = text.strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = " ".join(lines)  # flatten
    return (s[:max_len].rstrip() + ("..." if len(s) > max_len else ""))

def _safe_title(meta: dict, fallback: str) -> str:
    title = (meta or {}).get("title") or fallback or "Reference"
    return title.strip()

def _is_tradeoff(meta: dict, title: str) -> bool:
    t = (title or "").lower()
    kind = (meta or {}).get("note_kind", "").lower()
    return ("tradeoff" in t) or ((meta or {}).get("type") == "note" and kind == "adr")

def _is_incident(meta: dict, title: str) -> bool:
    t = (title or "").lower()
    kind = (meta or {}).get("note_kind", "").lower()
    return ("incident" in t) or ("rca" in t) or ((meta or {}).get("type") == "note" and kind in ("incident", "bug rca", "rca"))

def _is_warning_or_ops(meta: dict, text: str) -> bool:
    s = (text or "").lower()
    return ("warning" in s) or ("caveat" in s) or ("operations" in s) or ("ops" in s)

# New: define bulletization helper before usage in _compose_blocks
def _to_bullets(text: str, max_items: int = 5, max_item_len: int = 160) -> list[str]:
    if not text:
        return []
    import re
    parts = [p.strip() for p in re.split(r"[.!?;]\s+|\n+", text) if p.strip()]
    bullets, seen = [], set()
    for p in parts:
        p = p[:max_item_len].strip()
        if p and p not in seen and len(p) >= 3:
            bullets.append(p)
            seen.add(p)
        if len(bullets) >= max_items:
            break
    return bullets

def _compose_blocks(query: str, hits: list, notes_summary: str | None) -> str:
    # Pick the first non-note hit for Problem and Solution sections
    def _first_non_note(hs: list):
        for h in hs or []:
            meta = h.metadata or {}
            if (meta.get("type") or "").lower() != "note":
                return h
        return None

    primary = _first_non_note(hits)
    primary_meta = (primary.metadata if primary else {}) or {}

    # Section 1: Problem Addressed (prefer primary doc metadata.problem; fallback to query)
    problem_text = (primary_meta.get("problem") or query or "").strip()
    problem_section = "Problem Addressed\n- " + problem_text

    # Section 2: Solution (exclude notes entirely)
    solution_text = (primary_meta.get("code_snippet") or "")
    solution_section = "Code Snippet"
    if solution_text:
        solution_section += "\n" + _clean_snippet(solution_text.strip(), max_len=280)
    else:
        solution_section += "\n- No direct fix found. Refine the query or ingest more content."

    # Section 3: Tradeoffs (ADR + tradeoff-like hits + tradeoff text from notes_summary)
    tradeoffs_section = "Tradeoffs"
    tradeoff_bullets = _to_bullets(notes_summary or "", max_items=5, max_item_len=160)
    for h in hits[:6]:
        meta = h.metadata or {}
        title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
        if _is_tradeoff(meta, title):
            snippet = _clean_snippet(h.text, 200)
            for b in _to_bullets(snippet, max_items=2, max_item_len=160):
                if b not in tradeoff_bullets:
                    tradeoff_bullets.append(b)
    tradeoff_bullets = tradeoff_bullets[:5]
    if tradeoff_bullets:
        for b in tradeoff_bullets:
            tradeoffs_section += f"\n- {b}"
    else:
        tradeoffs_section += "\n- No prior tradeoffs found. Consider adding ADR notes."

    # Section 4: Other Considerations (incidents/RCA, ops/warnings, non-tradeoff references)
    other_section = "Other Considerations"
    seen_titles = set()
    other_items = []
    for h in hits[:8]:
        meta = h.metadata or {}
        title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        # Skip tradeoff items; collect incidents/rca/warnings/ops or general refs
        if _is_tradeoff(meta, title):
            continue
        snippet = _clean_snippet(h.text, 200)
        if _is_incident(meta, title) or _is_warning_or_ops(meta, h.text):
            other_items.append(f"- {title} (score: {h.score:.3f})\n  {snippet}")
        else:
            # keep concise general references, avoid very short/vague snippets
            if len(snippet) >= 20:
                other_items.append(f"- {title} (score: {h.score:.3f})\n  {snippet}")
    # Cap and render
    if other_items:
        other_section += "\n" + "\n".join(other_items[:6])
    else:
        other_section += "\n- No additional considerations."

    # Final composition
    return "\n\n".join([
        problem_section.strip(),
        solution_section.strip(),
        tradeoffs_section.strip(),
        other_section.strip()
    ])

# New: JSON request model for compose endpoint
class ComposeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    include_notes: bool = True
    # No change needed; compose reads stored code snippet from metadata

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "phishing mitigation", "k": 5, "include_notes": True},
                {"query": "db deadlock retries", "k": 8, "include_notes": True},
                {"query": "session ttl login", "k": 3, "include_notes": False}
            ]
        }
    }

def _extract_keywords(q: str) -> list[str]:
    import re
    toks = re.findall(r"[a-z0-9]{3,}", (q or "").lower())
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:5]

def _summarize_notes_inline(query: str) -> str | None:
    # Summarize decision notes related to the query
    kws = _extract_keywords(query)
    matched = []
    for kw in kws:
        try:
            rows = match_notes(keyword=kw, limit=10)
            matched.extend(rows)
        except Exception:
            continue
    if not matched:
        return None
    # Dedup by id
    uniq = {}
    for r in matched:
        uniq.setdefault(r.id, r)
    items = list(uniq.values())
    count = len(items)
    kinds = {}
    for r in items:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    # Build short bullets from content
    bullets = []
    for r in items[:3]:
        content = (r.content or "").strip()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        snippet = " ".join(lines[:2])[:240] if lines else content[:240]
        bullets.append(f"- {r.title}: {snippet}")
    kinds_str = ", ".join([f"{k}:{v}" for k, v in kinds.items()])
    return f"Previously, you had {count} notes related to '{' '.join(kws)}' ({kinds_str}).\n" + "\n".join(bullets)

@app.post("/v1/answers/compose")
def compose_answer(req: ComposeRequest):
    # Example: POST /v1/answers/compose
    # curl -s -X POST http://localhost:8000/v1/answers/compose \
    #   -H "Content-Type: application/json" \
    #   -d '{ "query":"phishing mitigation", "k":5, "include_notes":true }'
    # Search top-k
    try:
        search = post_json(
            f"{SEARCH_URL}/v1/search",
            SearchRequest(query=req.query, k=req.k).model_dump()
        )
        sr = SearchResponse(**search)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    hits = sr.hits or []
    notes_summary = _summarize_notes_inline(req.query) if req.include_notes else None
    answer = _compose_blocks(req.query, hits, notes_summary)
    return {"answer": answer, "hits": [h.model_dump() for h in hits]}

@app.post("/v1/admin/clear")
def admin_clear():
    # Example: POST /v1/admin/clear
    # curl -s -X POST http://localhost:8000/v1/admin/clear
    # Response: { "ok": true, "cleared": { "documents": N, "decision_notes": M, ... } }
    # Clear SQLite tables
    cleared = {"documents": 0, "decision_notes": 0, "vector_collection_deleted": False, "vector_collection_recreated": False}
    try:
        with SessionLocal() as s:
            # Count before delete
            docs_before = s.execute(sa_text("SELECT COUNT(*) FROM documents")).scalar() or 0
            notes_before = 0
            try:
                notes_before = s.execute(sa_text("SELECT COUNT(*) FROM decision_notes")).scalar() or 0
            except Exception:
                notes_before = 0  # table may not exist yet
            s.execute(sa_text("DELETE FROM documents"))
            try:
                s.execute(sa_text("DELETE FROM decision_notes"))
            except Exception:
                pass
            s.commit()
            cleared["documents"] = int(docs_before)
            cleared["decision_notes"] = int(notes_before)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB clear failed: {e}")

    # Clear Chroma collection using the shared client to avoid settings conflicts
    try:
        client = get_client()  # shared PersistentClient with consistent settings
        # Delete collection if present
        try:
            client.delete_collection(COLLECTION_NAME)
            cleared["vector_collection_deleted"] = True
        except Exception:
            # If not exists, ignore
            pass
        # Recreate collection with the same metadata used elsewhere
        client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        cleared["vector_collection_recreated"] = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store clear failed: {e}")

    return {"ok": True, "cleared": cleared}

# Telemetry heatmap (if present)
@app.get("/v1/telemetry/heatmap")
def telemetry_heatmap(limit: int = 1000):
    # Example: GET /v1/telemetry/heatmap?limit=500
    # curl -s "http://localhost:8000/v1/telemetry/heatmap?limit=500"
    pass  # ...existing code...
