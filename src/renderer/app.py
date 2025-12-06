from fastapi import FastAPI

from src.common.schemas import RenderRequest, RenderResponse

app = FastAPI(title="Renderer / Formatter")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/render", response_model=RenderResponse)
def render(req: RenderRequest):
    bullets = "\n".join([f"- ({h.score:.3f}) {h.text[:200]}" for h in req.hits])
    answer = f"Answer for: {req.query}\n\nTop results:\n{bullets}"
    return RenderResponse(answer=answer)
