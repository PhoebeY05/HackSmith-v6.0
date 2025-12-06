from sqlalchemy import JSON, Column, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import METADATA_DB

engine = create_engine(f"sqlite:///{METADATA_DB}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)  # same as vector id
    content = Column(Text, nullable=False)
    meta = Column("metadata", JSON, nullable=True)

# Decision notes: store ADRs, incident RCAs, tradeoffs, etc.
class DecisionNote(Base):
    __tablename__ = "decision_notes"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # reasoning/tradeoffs/root-cause/explanations
    kind = Column(String, nullable=False)   # e.g., "ADR", "Incident", "Bug RCA"
    tags = Column(JSON, nullable=True)      # e.g., ["db","deadlock","sqlite"]
    created_at = Column(String, nullable=False)  # ISO timestamp string
    # Optional link to a document id (provenance)
    doc_id = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(engine)
    # Lightweight migration: ensure decision_notes.doc_id exists
    try:
        with engine.begin() as conn:
            cols = conn.exec_driver_sql("PRAGMA table_info(decision_notes);").fetchall()
            col_names = {row[1] for row in cols}  # row[1] = name
            if "doc_id" not in col_names:
                conn.exec_driver_sql("ALTER TABLE decision_notes ADD COLUMN doc_id TEXT;")
    except Exception:
        # Safe to ignore; table may not exist yet or already migrated
        pass

def upsert_document(doc_id: str, content: str, metadata: dict | None = None):
    with SessionLocal() as s:
        existing = s.get(Document, doc_id)
        if existing:
            existing.content = content
            existing.meta = metadata or {}
        else:
            s.add(Document(id=doc_id, content=content, meta=metadata or {}))
        s.commit()

def get_documents(ids: list[str]) -> dict[str, Document]:
    with SessionLocal() as s:
        rows = s.query(Document).filter(Document.id.in_(ids)).all()
        return {r.id: r for r in rows}

# Add a decision note
def add_note(note_id: str, title: str, content: str, kind: str, tags: list[str], created_at: str, doc_id: str | None = None):
    with SessionLocal() as s:
        s.add(DecisionNote(
            id=note_id,
            title=title,
            content=content,
            kind=kind,
            tags=tags or [],
            created_at=created_at,
            doc_id=doc_id
        ))
        s.commit()

# List recent notes
def list_notes(limit: int = 20) -> list[DecisionNote]:
    with SessionLocal() as s:
        return s.query(DecisionNote).order_by(DecisionNote.created_at.desc()).limit(limit).all()

# Simple tag search
def search_notes_by_tag(tag: str, limit: int = 20) -> list[DecisionNote]:
    t = tag.lower()
    with SessionLocal() as s:
        rows = s.query(DecisionNote).all()
        matched = []
        for r in rows:
            tags = [x.lower() for x in (r.tags or [])]
            if t in tags:
                matched.append(r)
        return matched[:limit]

# Keyword match across title + content
def match_notes(keyword: str, limit: int = 10) -> list[DecisionNote]:
    kw = keyword.lower()
    with SessionLocal() as s:
        rows = s.query(DecisionNote).all()
        matched = []
        for r in rows:
            text = f"{r.title}\n{r.content}".lower()
            if kw in text:
                matched.append(r)
        return matched[:limit]

# New: list notes attached to a given document
def list_notes_for_doc(doc_id: str) -> list[DecisionNote]:
    with SessionLocal() as s:
        return s.query(DecisionNote).filter(DecisionNote.doc_id == doc_id).order_by(DecisionNote.created_at.desc()).all()
