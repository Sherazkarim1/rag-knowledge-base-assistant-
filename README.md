#  RAG Knowledge Base Assistant

### Production-Ready Retrieval-Augmented Generation (RAG) Pipeline with GPT-4, LangChain & ChromaDB

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?logo=chainlink)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange?logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **Built by [Sheraz Karim](https://www.upwork.com/freelancers/sherazkarim)** — Top Rated AI & Full Stack Developer on Upwork | Expert in Generative AI, LLM Apps, RAG Pipelines & AI Agents

---

##  What Is This?

Most developers "connect an API to ChatGPT" and call it an AI solution. This is different.

This project is a **fully production-ready RAG (Retrieval-Augmented Generation) system** that lets you upload your own documents (PDFs, Word files, text files) and ask natural language questions against them — with accurate, source-cited answers powered by GPT-4.

Built for real-world use cases:
-  **Enterprise knowledge management** — Let employees search internal policies, SOPs, and reports naturally
-  **Customer support automation** — Answer customer questions from your product docs automatically
-  **Research assistance** — Query large sets of research papers or legal documents instantly
-  **Education** — Build intelligent tutoring systems over course materials


##  Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG KNOWLEDGE BASE ASSISTANT                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📄 Documents (PDF/DOCX/TXT)                                       │
│        │                                                            │
│        ▼                                                            │
│   [LangChain Document Loaders]                                      │
│        │                                                            │
│        ▼                                                            │
│   [Recursive Text Splitter]  ──→  Chunks (1000 tokens, 150 overlap) │
│        │                                                            │
│        ▼                                                            │
│   [OpenAI Embeddings]  ──→  text-embedding-3-small                  │
│        │                                                            │
│        ▼                                                            │
│   [ChromaDB Vector Store]  ──→  Persisted on disk                   │
│        │                                                            │
│   ─────┼──────────────────── Query Time ─────────────────────────  │
│        │                                                            │
│   User Question ──→ [MMR Retriever (k=5)] ──→ Top Chunks           │
│                              │                                      │
│                              ▼                                      │
│                        [GPT-4 LLM]                                  │
│                              │                                      │
│                              ▼                                      │
│                     Answer + Source Citations                       │
└─────────────────────────────────────────────────────────────────────┘
```


##  Key Features

| Feature | Details |
|---|---|
|  **Multi-format Ingestion** | PDF, DOCX, TXT — upload any document |
|  **MMR Retrieval** | Maximal Marginal Relevance for diverse, accurate results |
|  **GPT-4 Powered** | Accurate, context-aware answers with zero hallucination |
|  **Source Citations** | Every answer includes which document it came from |
|  **FastAPI REST API** | Production-ready endpoints for upload, Q&A, and reset |
|  **Streamlit UI** | Clean chat interface — no code needed for end users |
|  **Docker Ready** | One-command deployment with Docker Compose |
|  **Persistent Storage** | ChromaDB persists to disk — survives restarts |
|  **Secure** | API key managed via environment variables, never hardcoded |


##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Sherazkarim1/rag-knowledge-base-assistant.git
cd rag-knowledge-base-assistant
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add Your Documents
```bash
# Drop any PDF, DOCX, or TXT files into:
data/sample_docs/
```

### 4. Run the API
```bash
uvicorn src.api:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 5. Run the UI (in a new terminal)
```bash
streamlit run src/app.py
# UI running at http://localhost:8501
```

### 6. Or Run Everything with Docker
```bash
docker-compose up --build
```

---

##  Usage — Python API

```python
from src.rag_pipeline import load_documents, build_vector_store, build_rag_chain, query

# Index your documents
docs = load_documents("data/sample_docs")
vs = build_vector_store(docs)

# Build RAG chain
chain = build_rag_chain(vs)

# Ask questions
result = query(chain, "What is our policy on customer data handling?")
print(result["answer"])
print("Sources:", result["sources"])
```

---

##  REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload & index documents |
| `POST` | `/ask` | Ask a question |
| `DELETE` | `/reset` | Clear knowledge base |

### Example: Ask a Question via cURL
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the prohibited AI use cases?"}'
```

### Example Response
```json
{
  "answer": "The prohibited AI use cases include autonomous financial decision-making without human approval, facial recognition in public spaces, and employee performance scoring without HR oversight.",
  "sources": ["company_ai_policy.txt"]
}
```

---

##  Project Structure

```
rag-knowledge-base-assistant/
├── src/
│   ├── rag_pipeline.py      # Core RAG logic (ingestion, embedding, retrieval)
│   ├── api.py               # FastAPI REST API
│   └── app.py               # Streamlit chat UI
├── data/
│   └── sample_docs/         # Drop your documents here
│       └── company_ai_policy.txt  # Example document
├── tests/
│   └── test_pipeline.py     # Unit tests
├── scripts/
│   └── ingest.py            # Standalone ingestion script
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── README.md
```

---

##  Configuration

All configuration is via environment variables in `.env`:

```env
OPENAI_API_KEY=sk-...          # Required: Your OpenAI key
OPENAI_MODEL=gpt-4             # LLM model (gpt-4, gpt-3.5-turbo)
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model
CHROMA_PERSIST_DIR=chroma_db   # Where to store vector DB
API_HOST=0.0.0.0               # API host
API_PORT=8000                  # API port
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4 |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Orchestration** | LangChain 0.1+ |
| **Vector Store** | ChromaDB |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Containers** | Docker + Docker Compose |
| **Language** | Python 3.11 |

---

##  Running Tests

```bash
pytest tests/ -v
```


##  Real-World Use Cases I've Built

This repo is a foundation. In client projects, I've extended this pattern to build:

-  **Customer Support Bots** — RAG over product documentation with Zendesk integration
-  **Legal Document Assistants** — Query contracts and compliance docs in natural language
-  **Internal HR Knowledge Bases** — Employees ask policy questions 24/7 without HR involvement
-  **Research Paper Search Engines** — Semantic search over 10,000+ academic papers
-  **Multi-tenant SaaS RAG Platforms** — Each customer gets isolated vector stores


##  About the Author

**Sheraz Karim** — AI Engineer & Full Stack Developer

-  **Top Rated** on Upwork with **100% Job Success Score**
-  Specialist in **Generative AI, LLM Apps, RAG Pipelines & AI Agents**
-  **AWS | Azure | DevOps** certified practitioner
-  Based in Pakistan — Available globally

 **Hire me on Upwork:** [upwork.com/freelancers/sherazkarim](https://www.upwork.com/freelancers/sherazkarim)
 **GitHub:** [github.com/Sherazkarim1](https://github.com/Sherazkarim1)


##  License

MIT License — feel free to use, modify, and distribute.


##  If this helped you, please star the repo!

```
Keywords: RAG, Retrieval-Augmented Generation, LangChain, GPT-4, ChromaDB, OpenAI,
LLM, AI Chatbot, Knowledge Base, Document Q&A, FastAPI, Streamlit, Python,
AI Agent, Vector Database, Embeddings, NLP, Generative AI, AI SaaS
```
