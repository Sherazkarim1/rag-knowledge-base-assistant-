"""
RAG Knowledge Base Assistant — FastAPI REST API
Exposes the RAG pipeline as a production-ready HTTP API with
document upload, indexing, and Q&A endpoints.
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import load_documents, build_vector_store, load_vector_store, build_rag_chain, query

# ── App Setup ─────────────────────────────────
app = FastAPI(
    title="RAG Knowledge Base Assistant",
    description="Upload documents and ask questions — powered by GPT-4 + ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR = "chroma_db"

# ── Lazy-load chain ───────────────────────────
_chain = None

def get_chain():
    global _chain
    if _chain is None:
        vs = load_vector_store(PERSIST_DIR)
        _chain = build_rag_chain(vs)
    return _chain


# ── Schemas ───────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


# ── Endpoints ─────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "RAG Knowledge Base Assistant"}


@app.post("/upload", tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, TXT, DOCX).
    They will be indexed into the ChromaDB vector store.
    """
    saved = []
    for file in files:
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    # Re-index
    docs = load_documents(str(UPLOAD_DIR))
    if not docs:
        raise HTTPException(status_code=400, detail="No readable documents found.")
    build_vector_store(docs, PERSIST_DIR)

    # Reset chain so it picks up new docs
    global _chain
    _chain = None

    return {"indexed": saved, "total_docs": len(docs)}


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(body: QuestionRequest):
    """
    Ask a question against your indexed knowledge base.
    Returns the answer and source document names.
    """
    if not Path(PERSIST_DIR).exists():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload documents first."
        )
    chain = get_chain()
    result = query(chain, body.question)
    return AnswerResponse(answer=result["answer"], sources=result["sources"])


@app.delete("/reset", tags=["Documents"])
def reset_knowledge_base():
    """Delete all indexed documents and reset the vector store."""
    global _chain
    _chain = None
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True)
    return {"status": "Knowledge base reset successfully."}
