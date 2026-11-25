from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from . import config
from .answer import answer_question
from .embeddings import get_embedder
from .ingest import IngestError, parse_document
from .models import AskRequest, AskResponse, DocumentInfo, UploadResponse
from .store import Store

app = FastAPI(title="HR Policy Assistant", version="0.1.0")


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(config.DB_PATH, get_embedder())


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required to manage documents.")


@app.get("/health")
def health():
    store = get_store()
    return {"status": "ok", "documents": len(store.list_documents()), "chunks": len(store.chunks),
            "llm": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}", "embeddings": config.EMBED_MODEL}


@app.post("/documents", response_model=UploadResponse, dependencies=[Depends(require_admin)])
def upload_document(file: UploadFile = File(...)):
    raw = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    try:
        doc = parse_document(file.filename or "upload.txt", raw)
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    r = get_store().upsert(doc)
    return UploadResponse(doc_id=r.doc_id, title=r.title, version=r.version, chunks=r.chunks, status=r.status)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return get_store().list_documents()


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_admin)])
def delete_document(doc_id: str):
    if not get_store().delete(doc_id):
        raise HTTPException(status_code=404, detail="No such document.")
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return answer_question(get_store(), req.question)
