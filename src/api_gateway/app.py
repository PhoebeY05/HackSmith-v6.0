import json
import os
import urllib.request
import uuid
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# Enable CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Endpoint I/O summary
# /v1/ingest
# Request (JSON, StructuredIngest):
# {
#   "title": string,
#   "problem": string,
#   "solution": string,
#   "source": string = "entry",
#   "code_snippet": string | null,
#   "ioc_type": string | null,            # "ip" | "domain" | "hash" | "url" | "registry" | "malware family"
#   "threat_level": string | null,        # "low" | "medium" | "high"
#   "incident_type": string | null,       # "phishing" | "malware" | "misconfiguration" | "vuln"
#   "notes": [{                           # optional decision notes to link to this document
#     "title": string,
#     "content": string,
#     "kind": string = "ADR",
#     "tags": [string]
#   }]
# }
# Response:
# {
#   "job_id": string,     # equals id
#   "id": string,         # document id (UUID)
#   "ok": true,
#   "note_ids": [string]  # UUIDs of saved notes (or "error:<msg>" entries)
# }
#
# /v1/query
# Request (JSON, QueryRequest) + optional query params:
# Body: { "query": string, "k": number }
# Query params (optional): ?ioc_type=...&incident_type=...&threat_level=...
# Response (QueryResponse):
# {
#   "answer": string,     # renderer-composed answer from hits
#   "hits": [             # top results (filtered)
#     {
#       "id": string,
#       "text": string,
#       "score": number,
#       "metadata": {     # includes enrichment if present
#         "title": string,
#         "source": string,
#         "problem": string,
#         "code_snippet": string | null,
#         "ioc_type": string | null,
#         "threat_level": string | null,
#         "incident_type": string | null,
#         "artifacts_ips": string | null,
#         "artifacts_domains": string | null,
#         "artifacts_urls": string | null,
#         "artifacts_hashes": string | null,
#         "artifacts_registry": string | null,
#         "vt_hash_reputation": string | null,
#         "abuseipdb_ip_score": string | null,
#         "urlhaus_signature": string | null,
#         # for notes: type="note", note_kind, content_snippet, doc_id
#       }
#     }
#   ]
# }
#
# /v1/answers/compose
# Request (JSON, ComposeRequest):
# { "query": string, "k": number = 5, "include_notes": boolean = true }
# Response:
# {
#   "answer": string,     # structured text with Problem, Code Snippet, Cybersecurity Context, Tradeoffs, Other Considerations
#   "hits": [Hit]         # same shape as in /v1/query (model_dump of hits)
# }

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

def _http_get_json(url: str, headers: dict | None = None, timeout: int = 6) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _http_post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 6) -> dict | None:
    try:
        data = json.dumps(body).encode("utf-8")
        base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "api-gateway/1.0"
        }
        req = urllib.request.Request(url, data=data, headers={**base_headers, **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Explicitly handle 307 Temporary Redirect for POST
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers.get("Location")
            if loc:
                try:
                    req2 = urllib.request.Request(loc, data=json.dumps(body).encode("utf-8"),
                                                  headers={**({"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "api-gateway/1.0"}), **(headers or {})})
                    with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                        return json.loads(resp2.read().decode("utf-8"))
                except Exception as e2:
                    print(f"[api_gateway] Redirect POST failed url={loc} err={e2}")
                    return None
        print(f"[api_gateway] POST failed url={url} code={e.code} err={e}")
        return None
    except Exception as e:
        print(f"[api_gateway] POST failed url={url} err={e}")
        return None

def _enrich_threat_intel(artifacts: dict) -> dict:
    """
    Best-effort, optional enrichment:
    - VirusTotal hash reputation (requires VT_API_KEY; skips if not present)
    - AbuseIPDB IP score (requires ABUSEIPDB_API_KEY; skips if not present)
    - URLhaus URL signature/family (no key required)
    Returns scalar strings suitable for Chroma metadata.
    """
    vt_key = os.getenv("VT_API_KEY", "").strip()
    abuse_key = os.getenv("ABUSEIPDB_API_KEY", "").strip()

    out = {}
    # VirusTotal hash reputation (sha256 preferred; accept any 32-64 hex)
    hashes = artifacts.get("hashes") or []
    if vt_key and hashes:
        # pick first hash
        h = hashes[0]
        vt_url = f"https://www.virustotal.com/api/v3/files/{h}"
        vt = _http_get_json(vt_url, headers={"x-apikey": vt_key})
        if vt and isinstance(vt.get("data"), dict):
            attrs = vt["data"].get("attributes") or {}
            stats = attrs.get("last_analysis_stats") or {}
            verdict = attrs.get("meaningful_name") or ""
            # Compose a compact reputation string
            detected = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
            harmless = int(stats.get("harmless", 0))
            out["vt_hash_reputation"] = f"detected:{detected}, harmless:{harmless}, name:{verdict}"[:200]

    # AbuseIPDB for IPs
    ips = artifacts.get("ips") or []
    if abuse_key and ips:
        ip = ips[0]
        abuse_url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&maxAgeInDays=90"
        abuse = _http_get_json(abuse_url, headers={"Key": abuse_key, "Accept": "application/json"})
        if abuse and isinstance(abuse.get("data"), dict):
            data = abuse["data"]
            score = data.get("abuseConfidenceScore")
            country = (data.get("countryCode") or "")
            out["abuseipdb_ip_score"] = f"score:{score}, country:{country}"[:200]

    # URLhaus for URLs (no API key; POST supported)
    urls = artifacts.get("urls") or []
    if urls:
        u = urls[0]
        try:
            import re
            m = re.match(r"https?://([^/]+)", u)
            host = m.group(1) if m else ""
            if host:
                urlhaus_api = "https://urlhaus.abuse.ch/api/host/"
                j = _http_post_json(urlhaus_api, {"host": host})
                if j and j.get("query_status") == "ok":
                    entries = j.get("urls") or []
                    sig = entries[0].get("signature") if entries else ""
                    out["urlhaus_signature"] = (sig or "unknown")[:200]
        except Exception:
            pass

    # Return only scalar strings to store in metadata
    return out

@app.post("/v1/ingest")
def ingest(req: StructuredIngest):
    # Example: POST /v1/ingest
    # curl -s -X POST http://localhost:8000/v1/ingest \
    #   -H "Content-Type: application/json" \
    #   -d '{
    #     "title":"Suspicious outbound traffic",
    #     "problem":"Firewall logs show repeated connections to known bad IP and domain. Hash observed in EDR alert.",
    #     "solution":"Block IP/domain at edge; quarantine host; investigate related hashes.",
    #     "source":"entry",
    #     "ioc_type":"url",
    #     "threat_level":"high",
    #     "incident_type":"malware",
    #     "code_snippet":"IPs: 45.83.64.12\nDomain: bad-example.co\nURL: https://bad-example.co/payload\nHash: 44d88612fea8a8f36de82e1278abb02f",
    #     "notes":[{"title":"Tradeoffs","content":"Blocking entire /24 may impact legitimate services.","kind":"ADR","tags":["network","blocklist"]}]
    #   }'
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
    # Ensure enrichment is defined before use
    try:
        enrichment = _enrich_threat_intel(artifacts)
    except Exception:
        enrichment = {}

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
    # Inject enrichment scalars (optional; only if present)
    if enrichment.get("vt_hash_reputation"):
        metadata["vt_hash_reputation"] = enrichment["vt_hash_reputation"]
    if enrichment.get("abuseipdb_ip_score"):
        metadata["abuseipdb_ip_score"] = enrichment["abuseipdb_ip_score"]
    if enrichment.get("urlhaus_signature"):
        metadata["urlhaus_signature"] = enrichment["urlhaus_signature"]

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
        msg = str(e)
        # Friendly fallback when the Chroma collection is missing in the search service
        if "InvalidCollectionException" in msg or "does not exist" in msg:
            return QueryResponse(
                answer="Search index is not initialized on the search service. Run /v1/admin/clear to recreate the collection or initialize it on the search backend.",
                hits=[]
            )
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    sr = SearchResponse(**search)

    # New: defensively ensure each hit has metadata and non-empty text; drop orphans
    safe_hits = []
    for h in sr.hits or []:
        # normalize metadata to dict
        if not isinstance(h.metadata, dict) or h.metadata is None:
            h.metadata = {}
        # patch text from metadata if missing
        if not isinstance(h.text, str) or not h.text.strip():
            patched = (h.metadata.get("content_snippet") or h.metadata.get("problem") or "") or ""
            h.text = patched if isinstance(patched, str) else ""
        # skip hits that still have empty text after patching (likely orphaned after delete)
        if not h.text.strip():
            continue
        safe_hits.append(h)
    sr.hits = safe_hits

    # New: define helper BEFORE any branch uses it
    def _attach_related_notes(hs: list) -> list:
        enriched = []
        for h in hs or []:
            meta = h.metadata or {}
            if ((meta.get("type") or "").lower() != "note") and h.id:
                try:
                    rows = list_notes_for_doc(doc_id=h.id)
                    meta["related_notes"] = [
                        {"id": r.id, "title": r.title, "kind": r.kind, "created_at": r.created_at, "text": r.content, "tags": r.tags}
                        for r in rows
                    ]
                    h.metadata = meta
                except Exception:
                    pass
            enriched.append(h)
        return enriched

    # New: list recent non-note documents directly from DB (apply optional filters)
    def _list_all_docs(limit: int = 100) -> list:
        try:
            with SessionLocal() as s:
                rows = s.execute(
                    sa_text("SELECT id, content, metadata FROM documents ORDER BY rowid DESC LIMIT :n"),
                    {"n": limit}
                ).fetchall()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read documents: {e}")

        def _sec_passes_meta(meta: dict) -> bool:
            m = meta or {}
            if ioc_type and (m.get("ioc_type") or "").lower() != ioc_type.lower():
                return False
            if incident_type and (m.get("incident_type") or "").lower() != incident_type.lower():
                return False
            if threat_level and (m.get("threat_level") or "").lower() != threat_level.lower():
                return False
            return True

        hits = []
        for doc_id, content, meta in rows:
            # Normalize metadata from DB: it can be a dict, a JSON string, or None
            if isinstance(meta, dict):
                m = meta
            elif isinstance(meta, str):
                try:
                    m = json.loads(meta) if meta.strip() else {}
                except Exception:
                    m = {}
            else:
                m = {}

            # skip explicit notes (defensive; notes are stored in decision_notes, but keep guard)
            if (m.get("type") or "").lower() == "note":
                continue
            if not _sec_passes_meta(m):
                continue
            # Build a SearchHit-compatible dict
            hits.append(
                SearchResponse.model_fields["hits"].annotation.__args__[0](
                    id=doc_id,
                    text=(content or ""),
                    score=1.0,
                    metadata=m
                )
            )
        return hits

    # Empty query: return all docs from DB with filters, then enrich with related notes
    if not (req.query or "").strip():
        all_hits = _list_all_docs(limit=100)
        if not all_hits:
            return QueryResponse(answer="No non-note entries found for the empty query.", hits=[])
        all_hits = _attach_related_notes(all_hits)
        return QueryResponse(answer="Listing recent entries from the database (non-note).", hits=all_hits)

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

    # Attach related notes
    filtered_hits = _attach_related_notes(filtered_hits)

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
    # New: append security context (incident_type + threat_level) if present
    incident = (primary_meta.get("incident_type") or "").strip()
    threat = (primary_meta.get("threat_level") or "").strip()
    sec_suffix = ""
    if incident or threat:
        ctx = []
        if incident:
            ctx.append(f"incident_type: {incident}")
        if threat:
            ctx.append(f"threat_level: {threat}")
        sec_suffix = f" [{'; '.join(ctx)}]"
    problem_section = "Problem Addressed\n- " + problem_text + sec_suffix

    # Section 2: Solution (exclude notes entirely)
    solution_text = (primary_meta.get("code_snippet") or "")
    solution_section = "Code Snippet"
    if solution_text:
        solution_section += "\n" + _clean_snippet(solution_text.strip(), max_len=280)
    else:
        solution_section += "\n- No direct fix found. Refine the query or ingest more content."

    # New: add Cybersecurity context bullets under Solution when relevant
    ioc_type = (primary_meta.get("ioc_type") or "").strip()
    artifacts_ips = (primary_meta.get("artifacts_ips") or "").strip()
    artifacts_domains = (primary_meta.get("artifacts_domains") or "").strip()
    artifacts_urls = (primary_meta.get("artifacts_urls") or "").strip()
    artifacts_hashes = (primary_meta.get("artifacts_hashes") or "").strip()
    artifacts_registry = (primary_meta.get("artifacts_registry") or "").strip()

    # New: enrichment fields; if missing on primary, fallback to first hit that has them
    vt_hash_rep = (primary_meta.get("vt_hash_reputation") or "").strip()
    abuse_ip_score = (primary_meta.get("abuseipdb_ip_score") or "").strip()
    urlhaus_sig = (primary_meta.get("urlhaus_signature") or "").strip()
    if not (vt_hash_rep and abuse_ip_score and urlhaus_sig):
        for h in hits or []:
            meta_h = h.metadata or {}
            vt_hash_rep = vt_hash_rep or (meta_h.get("vt_hash_reputation") or "").strip()
            abuse_ip_score = abuse_ip_score or (meta_h.get("abuseipdb_ip_score") or "").strip()
            urlhaus_sig = urlhaus_sig or (meta_h.get("urlhaus_signature") or "").strip()
            if vt_hash_rep and abuse_ip_score and urlhaus_sig:
                break

    sec_lines = []
    if ioc_type:
        sec_lines.append(f"- IOC Type: {ioc_type}")
    if artifacts_ips:
        sec_lines.append(f"- IPs: {artifacts_ips}")
    if artifacts_domains:
        sec_lines.append(f"- Domains: {artifacts_domains}")
    if artifacts_urls:
        sec_lines.append(f"- URLs: {artifacts_urls}")
    if artifacts_hashes:
        sec_lines.append(f"- Hashes: {artifacts_hashes}")
    if artifacts_registry:
        sec_lines.append(f"- Registry: {artifacts_registry}")
    # Include enrichment, if present
    if vt_hash_rep:
        sec_lines.append(f"- VT Hash Reputation: {vt_hash_rep}")
    if abuse_ip_score:
        sec_lines.append(f"- AbuseIPDB IP Score: {abuse_ip_score}")
    if urlhaus_sig:
        sec_lines.append(f"- URLhaus Signature: {urlhaus_sig}")

    if sec_lines:
        solution_section += "\n\nCybersecurity Context\n" + "\n".join(sec_lines)

    # Fix: define notes_only and helper before any usage
    notes_only = [h for h in (hits or []) if ((h.metadata or {}).get("type") or "").lower() == "note"]
    def _note_snippet(h) -> str:
        meta = h.metadata or {}
        raw = (meta.get("content_snippet") or h.text or "") or ""
        return _clean_snippet(raw, 220)

    # New: tracking sets to deduplicate across sections
    trade_titles_seen = set()       # normalized titles added to Tradeoffs
    other_titles_seen = set()       # normalized titles added to Other Considerations

    def _norm_title(t: str) -> str:
        return (t or "").strip().lower()

    # Section 3: Tradeoffs (prioritize ADR notes, dedupe by title)
    tradeoffs_section = "Tradeoffs"
    tradeoff_bullets = []
    for h in notes_only:
        meta = h.metadata or {}
        kind = (meta.get("note_kind") or "").lower()
        if kind == "adr":
            title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
            nt = _norm_title(title)
            if nt in trade_titles_seen:
                continue
            trade_titles_seen.add(nt)
            snippet = _note_snippet(h)
            tradeoff_bullets.append(f"- {title}: {snippet}" if snippet else f"- {title} (score: {h.score:.3f})")
            if len(tradeoff_bullets) >= 5:
                break
    if not tradeoff_bullets:
        for h in hits[:6]:
            meta = h.metadata or {}
            title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
            nt = _norm_title(title)
            if nt in trade_titles_seen:
                continue
            if _is_tradeoff(meta, title):
                trade_titles_seen.add(nt)
                tradeoff_bullets.append(f"- {title} (score: {h.score:.3f})")

    if notes_summary:
        for b in _to_bullets(notes_summary, max_items=3, max_item_len=160):
            # avoid adding an identical bullet twice
            if b not in tradeoff_bullets:
                tradeoff_bullets.append(b)

    tradeoffs_section += ("\n" + "\n".join(tradeoff_bullets[:6])) if tradeoff_bullets else "\n- No tradeoffs or ADRs found."

    # Section 4: Other Considerations (dedupe against trade_titles_seen and within section)
    other_section = "Other Considerations"
    incidents, warnings, general_refs = [], [], []

    # Prefer Incident/RCA notes first
    for h in notes_only:
        meta = h.metadata or {}
        title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
        nt = _norm_title(title)
        if nt in other_titles_seen or nt in trade_titles_seen:
            continue
        kind = (meta.get("note_kind") or "").lower()
        if kind in ("incident", "bug rca", "rca"):
            other_titles_seen.add(nt)
            snippet = _note_snippet(h)
            incidents.append(f"- {title}: {snippet}" if snippet else f"- {title} (score: {h.score:.3f})")
            if len(incidents) >= 3:
                break

    # Prefer warnings/ops notes next
    for h in notes_only:
        meta = h.metadata or {}
        title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
        nt = _norm_title(title)
        if nt in other_titles_seen or nt in trade_titles_seen:
            continue
        text_lower = (_note_snippet(h) or h.text or "").lower()
        if "warning" in text_lower or "caveat" in text_lower or "ops" in text_lower or "operations" in text_lower:
            other_titles_seen.add(nt)
            snippet = _note_snippet(h)
            warnings.append(f"- {title}: {snippet}" if snippet else f"- {title} (score: {h.score:.3f})")
            if len(warnings) >= 3:
                break

    # Fill remaining from general (non-note) references (exclude tradeoff-like items)
    for h in hits[:8]:
        meta = h.metadata or {}
        title = _safe_title(meta, (h.text.splitlines()[0][:50] if h.text else ""))
        nt = _norm_title(title)
        if nt in other_titles_seen or nt in trade_titles_seen:
            continue
        # Skip tradeoff-like items in general refs
        if _is_tradeoff(meta, title):
            continue
        other_titles_seen.add(nt)
        if _is_incident(meta, title):
            incidents.append(f"- {title} (score: {h.score:.3f})")
        elif _is_warning_or_ops(meta, h.text):
            warnings.append(f"- {title} (score: {h.score:.3f})")
        else:
            snippet = _clean_snippet(h.text, 200)
            if len(snippet) >= 20:
                general_refs.append(f"- {title} (score: {h.score:.3f})\n  {snippet}")

    if incidents:
        other_section += "\nIncidents/RCA:\n" + "\n".join(incidents[:3])
    if warnings:
        other_section += "\nWarnings/Ops Considerations:\n" + "\n".join(warnings[:3])
    if general_refs:
        other_section += "\nGeneral References:\n" + "\n".join(general_refs[:3])
    if not (incidents or warnings or general_refs):
        other_section += "\n- No additional considerations."

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
            out.append(t)
            seen.add(t)
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
    try:
        search = post_json(
            f"{SEARCH_URL}/v1/search",
            SearchRequest(query=req.query, k=req.k).model_dump()
        )
        sr = SearchResponse(**search)
    except Exception as e:
        msg = str(e)
        if "InvalidCollectionException" in msg or "does not exist" in msg:
            return {"answer": "Search index is not initialized on the search service. Run /v1/admin/clear to recreate the collection or initialize it on the search backend.", "hits": []}
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    # New: defensively normalize metadata, patch text, and drop orphans
    safe_hits = []
    for h in (sr.hits or []):
        if not isinstance(h.metadata, dict) or h.metadata is None:
            h.metadata = {}
        if not isinstance(h.text, str) or not h.text.strip():
            patched = (h.metadata.get("content_snippet") or h.metadata.get("problem") or "") or ""
            h.text = patched if isinstance(patched, str) else ""
        if not h.text.strip():
            continue
        safe_hits.append(h)

    # New: enforce minimum score for compose
    min_score = 0.25
    hits = [h for h in safe_hits if (h.score or 0.0) >= min_score]

    notes_summary = _summarize_notes_inline(req.query) if req.include_notes else None
    answer = _compose_blocks(req.query, hits, notes_summary)

    # New: return only non-note hits meeting the score threshold
    non_note_hits = [h for h in hits if ((h.metadata or {}).get("type") or "").lower() != "note"]
    return {"answer": answer, "hits": [h.model_dump() for h in non_note_hits]}

@app.post("/v1/admin/clear")
def admin_clear():
    # Example: POST /v1/admin/clear
    # curl -s -X POST http://localhost:8000/v1/admin/clear
    cleared = {"documents": 0, "decision_notes": 0, "vector_collection_deleted": False, "vector_collection_recreated": False}
    try:
        with SessionLocal() as s:
            # Count before delete
            docs_before = s.execute(sa_text("SELECT COUNT(*) FROM documents")).scalar() or 0
            try:
                notes_before = s.execute(sa_text("SELECT COUNT(*) FROM decision_notes")).scalar() or 0
            except Exception:
                notes_before = 0
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

    # Clear Chroma collection using the shared client
    try:
        client = get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
            cleared["vector_collection_deleted"] = True
        except Exception:
            pass
        client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        cleared["vector_collection_recreated"] = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store clear failed: {e}")

    return {"ok": True, "cleared": cleared}

@app.delete("/v1/docs/delete")
def docs_delete(id: str):
    # Example: DELETE /v1/docs/delete?id=<uuid>
    deleted = {"document": False, "notes": 0, "vectors": {"doc": False, "notes": 0}}
    try:
        # Collect linked note ids
        try:
            rows = list_notes_for_doc(doc_id=id)
            note_ids = [r.id for r in rows]
        except Exception:
            note_ids = []

        # Delete from DB
        with SessionLocal() as s:
            doc_exists = s.execute(sa_text("SELECT 1 FROM documents WHERE id = :id"), {"id": id}).fetchone()
            if not doc_exists:
                raise HTTPException(status_code=404, detail="Document not found")
            try:
                s.execute(sa_text("DELETE FROM decision_notes WHERE doc_id = :id"), {"id": id})
                deleted["notes"] = len(note_ids)
            except Exception:
                deleted["notes"] = 0
            s.execute(sa_text("DELETE FROM documents WHERE id = :id"), {"id": id})
            s.commit()
            deleted["document"] = True

        # Delete vectors from Chroma (best-effort)
        try:
            client = get_client()
            coll = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            coll.delete(ids=[id])
            deleted["vectors"]["doc"] = True
            if note_ids:
                coll.delete(ids=note_ids)
                deleted["vectors"]["notes"] = len(note_ids)
        except Exception:
            pass

        return {"ok": True, "deleted": deleted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@app.post("/v1/admin/seed")
def admin_seed():
    # Seed cybersecurity-related documents with linked notes (50 entries),
    # using the exact same path as /v1/ingest to ensure embeddings/metadata/indexing are consistent.
    seeds = []
    ioc_types = ["domain", "ip", "url", "hash", "registry"]
    incident_types = ["phishing", "malware", "misconfiguration", "vuln", "phishing"]
    threat_levels = ["low", "medium", "high"]
    domains = [f"login-secure-{i}.co" for i in range(1, 21)]
    ips = [f"45.83.64.{i}" for i in range(10, 30)]
    urls = [f"https://bad-example{i}.co/payload" for i in range(1, 21)]
    hashes = [f"{i:064x}"[:64] for i in range(1, 21)]
    registries = [r"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run\Updater" for _ in range(1, 21)]

    def _mk_notes(idx: int, topic: str):
        return [
            {"title": f"Tradeoffs #{idx}: Response hardening", "content": "Tight policies may break legacy dependencies; phase rollout and whitelist trusted apps.", "kind": "ADR", "tags": ["ops","security", topic, "tradeoffs"]},
            {"title": f"Incident RCA #{idx}", "content": "Root cause tied to user behavior and missing detections; improve awareness and detections.", "kind": "Incident", "tags": ["ops","security", topic, "rca"]}
        ]

    for i in range(50):
        ioc = ioc_types[i % len(ioc_types)]
        inc = incident_types[i % len(incident_types)]
        thr = threat_levels[i % len(threat_levels)]
        title = f"[Seed #{i+1}] {inc.title()} event with {ioc.upper()} indicators"
        problem = f"Detection #{i+1}: Observed suspicious activity tied to {ioc} indicators during {inc}."
        if ioc == "domain":
            dom = domains[i % len(domains)]
            code_snippet = f"Suspicious domain: {dom}\nLogin page lookalike reported by users."
        elif ioc == "ip":
            ip = ips[i % len(ips)]
            code_snippet = f"Beacon IP: {ip}\nFirewall logs show repeated outbound connections."
        elif ioc == "url":
            u = urls[i % len(urls)]
            code_snippet = f"Malicious URL: {u}\nEDR flagged download attempts."
        elif ioc == "hash":
            h = hashes[i % len(hashes)]
            code_snippet = f"Artifact hash: {h}\nObserved in endpoint telemetry."
        else:
            reg = registries[i % len(registries)]
            code_snippet = f"Persistence key: {reg}\nAutorun entry added by unknown binary."

        solution = "Contain, block indicators, and remediate; add detection rules and improve user awareness."
        notes = _mk_notes(i+1, ioc)

        # Use source="entry" to mirror normal ingest payloads
        seeds.append({
            "title": title,
            "problem": problem,
            "solution": solution,
            "source": "entry",
            "ioc_type": ioc,
            "threat_level": thr,
            "incident_type": inc,
            "code_snippet": code_snippet,
            "notes": notes
        })

    created = []
    for item in seeds:
        try:
            # Build StructuredIngest exactly like normal ingest usage
            req = StructuredIngest(
                title=item["title"],
                problem=item["problem"],
                solution=item["solution"],
                source=item["source"],  # "entry"
                code_snippet=item.get("code_snippet"),
                ioc_type=item.get("ioc_type"),
                threat_level=item.get("threat_level"),
                incident_type=item.get("incident_type"),
                notes=[DecisionNoteIn(**n) for n in (item.get("notes") or [])],
            )
            # Call ingest() so embeddings, metadata, artifacts, and notes are processed identically
            resp = ingest(req)
            created.append({"id": resp["id"], "note_ids": resp["note_ids"]})
        except Exception as e:
            created.append({"error": str(e)})

    return {"ok": True, "seeded": len(created), "items": created}

