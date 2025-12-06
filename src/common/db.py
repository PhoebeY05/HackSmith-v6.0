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
    metadata = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(engine)

def upsert_document(doc_id: str, content: str, metadata: dict | None = None):
    with SessionLocal() as s:
        existing = s.get(Document, doc_id)
        if existing:
            existing.content = content
            existing.metadata = metadata or {}
        else:
            s.add(Document(id=doc_id, content=content, metadata=metadata or {}))
        s.commit()

def get_documents(ids: list[str]) -> dict[str, Document]:
    with SessionLocal() as s:
        rows = s.query(Document).filter(Document.id.in_(ids)).all()
        return {r.id: r for r in rows}
