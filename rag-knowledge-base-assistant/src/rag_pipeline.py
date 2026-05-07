"""
RAG Knowledge Base Assistant — Core Pipeline
Author: Sheraz Karim
Description: Production-ready Retrieval-Augmented Generation pipeline
             using OpenAI GPT-4, LangChain, and ChromaDB vector store.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

load_dotenv()


# ─────────────────────────────────────────────
# 1. Document Ingestion
# ─────────────────────────────────────────────

def load_documents(data_dir: str = "data/sample_docs") -> list:
    """
    Load documents from a directory.
    Supports: PDF, TXT, DOCX
    """
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARNING] Loader skipped: {e}")
    print(f"[INFO] Loaded {len(docs)} document(s).")
    return docs


# ─────────────────────────────────────────────
# 2. Chunking & Embedding
# ─────────────────────────────────────────────

def build_vector_store(docs: list, persist_dir: str = "chroma_db") -> Chroma:
    """
    Split documents into chunks, embed them, and store in ChromaDB.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunk(s).")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vector_store.persist()
    print(f"[INFO] Vector store saved to '{persist_dir}'.")
    return vector_store


def load_vector_store(persist_dir: str = "chroma_db") -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


# ─────────────────────────────────────────────
# 3. RAG Chain
# ─────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable assistant. Use ONLY the context below to answer.
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""",
)


def build_rag_chain(vector_store: Chroma, streaming: bool = False) -> RetrievalQA:
    """
    Build a RetrievalQA chain with GPT-4 and the vector store retriever.
    """
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        streaming=streaming,
        callbacks=callbacks,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",           # Maximal Marginal Relevance for diversity
        search_kwargs={"k": 5, "fetch_k": 20},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ─────────────────────────────────────────────
# 4. Query Interface
# ─────────────────────────────────────────────

def query(chain: RetrievalQA, question: str) -> dict:
    """Run a question through the RAG chain and return answer + sources."""
    result = chain({"query": question})
    sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
    return {
        "answer": result["result"],
        "sources": list(set(sources)),
    }


# ─────────────────────────────────────────────
# 5. Main — Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Knowledge Base Assistant ===\n")

    # Step 1: Load & index documents
    docs = load_documents("data/sample_docs")
    if not docs:
        print("[ERROR] No documents found in data/sample_docs/")
        exit(1)

    vs = build_vector_store(docs)

    # Step 2: Build chain
    chain = build_rag_chain(vs, streaming=True)

    # Step 3: Interactive Q&A loop
    print("\nAsk questions about your documents (type 'exit' to quit):\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        response = query(chain, q)
        print(f"\nAnswer: {response['answer']}")
        print(f"Sources: {response['sources']}\n")
