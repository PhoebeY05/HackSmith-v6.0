HackSmith v6.0 MVP

Run

- cp .env.example .env
- docker compose up --build

Core flows

- Ingest: POST <http://localhost:8000/v1/ingest> { "text": "..." }
- Query:  POST <http://localhost:8000/v1/query> { "query": "..." }

Notes

- Vector store persisted in ./data/chroma
- Metadata DB: ./data/metadata.db (SQLite)
- RQ queue name: index
