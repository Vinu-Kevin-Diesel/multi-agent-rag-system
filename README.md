# Autonomous Document Intelligence Agent

A multi-agent RAG system that ingests heterogeneous documents and routes queries to specialized sub-agents using LangGraph, FastAPI, and pgvector. Features a React + Tailwind frontend for document management and interactive querying.

## Architecture

```
                    +------------------+
                    |   POST /ingest   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Layout Detection |  Stage 1: unstructured
                    |   (+ OCR inside)  |  tesseract for images/scanned PDFs
                    +--------+---------+
                             |
                    +--------v---------+
                    | Semantic Chunking |  Stage 2: sentence-transformers
                    +--------+---------+
                             |
                    +--------v---------+
                    |  pgvector (HNSW) |  384-dim vectors, cosine similarity
                    +------------------+

                    +------------------+
                    |   POST /query    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Router Agent   |  Classifies: factual / comparative / multihop
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     |  Factual   |  | Comparative |  |  Multi-hop  |
     |   Agent    |  |    Agent    |  |    Agent     |
     +--------+---+  +------+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   Critic Agent   |  Cosine similarity scoring
                    +--------+---------+
                             |
                     confidence < 0.78?
                        YES -> re-retrieve with refined query (up to 3 attempts)
                        NO  -> return answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React + Vite + TypeScript + Tailwind CSS |
| Orchestration | LangGraph (StateGraph) |
| LLM | Any OpenAI-compatible endpoint — hosted (NVIDIA NIM, free tier) or local (Ollama / vLLM) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, 384-dim) |
| Vector DB | PostgreSQL 16 + pgvector (HNSW index) |
| API | FastAPI (async) |
| Document Parsing | unstructured (tesseract OCR for images / scanned PDFs) |
| Semantic Chunking | sentence-transformers (cosine similarity-based splits) |
| ORM | SQLAlchemy 2.0 (async + asyncpg) |
| Migrations | Alembic |
| Containerization | Docker Compose (3 services) |

## Features

- **Drag-and-drop document upload** supporting PDF, DOCX, HTML, TXT, and images
- **Multi-agent query routing** -- automatically classifies queries as factual, comparative, or multi-hop
- **Self-correcting retrieval** -- critic agent scores answers and triggers re-retrieval with refined queries if confidence is low
- **Cross-document search** -- query across all ingested documents simultaneously
- **Source attribution** -- every answer includes source chunks with page numbers and similarity scores
- **Zero API cost** -- local embeddings (sentence-transformers) + free LLM tier (NVIDIA NIM)

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI-compatible LLM endpoint. Either:
  - a free NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com), or
  - a local server (Ollama / vLLM) — set `LLM_BASE_URL` and no key is needed

### Setup

```bash
# Clone the repo
git clone https://github.com/Vinu-Kevin-Diesel/multi-agent-rag-system.git
cd multi-agent-rag-system

# Configure environment
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY

# Start all services (db + backend + frontend)
docker compose up --build
```

Three services will start:
- **Frontend**: http://localhost:3000 (React UI)
- **Backend API**: http://localhost:8000 (FastAPI + Swagger at /docs)
- **Database**: PostgreSQL 16 + pgvector on port 5432

### Usage

1. Open http://localhost:3000
2. **Upload** a PDF (or DOCX, HTML, TXT, image) via the sidebar
3. **Select** a document or choose "All Documents" for cross-document search
4. **Ask** a question and get a grounded answer with confidence score and source citations

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest` | Upload and process a document |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/query` | Query documents with natural language |
| `GET` | `/health` | Health check |

#### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@policy.pdf"
```

Response:
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "policy.pdf",
  "num_chunks": 22,
  "page_count": 5
}
```

#### Query Documents

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the coverage criteria?", "top_k": 5}'
```

Response:
```json
{
  "answer": "Based on the source chunks...",
  "query_type": "factual",
  "confidence": 0.87,
  "sources": [
    {
      "chunk_id": "...",
      "content": "...",
      "page_number": 1,
      "similarity": 0.75
    }
  ],
  "retrieval_attempts": 1
}
```

## How It Works

### Document Ingestion Pipeline

1. **Layout Detection** -- `unstructured` parses the uploaded file into typed regions (Title, NarrativeText, Table, Image, etc.). OCR happens here rather than as a stage of its own: the library runs tesseract internally for standalone images and for PDF pages with no extractable text. `INGESTION_STRATEGY` controls it -- `fast` skips OCR, `hi_res` and `ocr_only` force it. Blank regions are dropped.
2. **Semantic Chunking** -- Sentences are grouped by semantic similarity using `all-MiniLM-L6-v2` embeddings. Chunk boundaries are created when cosine similarity drops below 0.5 or token count exceeds 512
3. **Embedding & Storage** -- All chunks are embedded to 384-dim vectors and stored in pgvector with an HNSW index (m=16, ef_construction=64) for fast cosine similarity search

### Query Pipeline (LangGraph)

1. **Router Agent** -- Classifies the query as `factual`, `comparative`, or `multihop`
2. **Retrieval** -- pgvector HNSW index finds the top-k most similar chunks. Multi-hop queries are first decomposed into sub-questions, each retrieved for independently, then merged and deduped
3. **Specialized Agent** -- Routes to the appropriate agent for answer generation
4. **Critic Agent** -- Scores the answer via embedding cosine similarity against source chunks
5. **Retry Loop** -- If confidence < 0.78 and attempts < 3, the critic generates a refined query and re-retrieves

The refined query drives *retrieval only*. Agents always answer the user's original question,
so a rewritten search string can never become the question being answered.

## Project Structure

```
multi-agent-rag-system/
├── app/
│   ├── main.py                # FastAPI app + CORS middleware
│   ├── config.py              # Settings (pydantic-settings)
│   ├── models.py              # SQLAlchemy models + pgvector Vector(384)
│   ├── database.py            # Async DB session
│   ├── schemas.py             # Request/response Pydantic models
│   ├── dependencies.py        # NVIDIA NIM client setup
│   ├── ingestion/
│   │   ├── pipeline.py        # ingestion orchestrator
│   │   ├── layout_detection.py # unstructured parsing (+ OCR)
│   │   └── chunking.py
│   ├── agents/
│   │   ├── graph.py           # LangGraph StateGraph
│   │   ├── router_agent.py    # Query classifier
│   │   ├── factual_agent.py
│   │   ├── comparative_agent.py
│   │   ├── multihop_agent.py
│   │   ├── critic_agent.py    # Answer validation + query refinement
│   │   └── utils.py           # Response content extraction helper
│   ├── retrieval/
│   │   └── vector_store.py    # pgvector HNSW similarity search
│   └── utils/
│       └── embeddings.py      # Local sentence-transformers embeddings
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts         # Vite dev server + API proxy
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx            # Main layout (sidebar + query area)
│       ├── api/client.ts      # API client (fetch wrappers)
│       ├── components/
│       │   ├── DocumentUpload.tsx   # Drag-and-drop upload zone
│       │   ├── DocumentList.tsx     # Document sidebar with delete
│       │   ├── QueryInput.tsx       # Chat-style input
│       │   ├── AnswerDisplay.tsx    # Markdown answer + badges
│       │   ├── ConfidenceBadge.tsx  # Color-coded confidence bar
│       │   ├── QueryTypeBadge.tsx   # factual/comparative/multihop pill
│       │   ├── SourceCard.tsx       # Expandable source chunk
│       │   └── SourceList.tsx       # Source cards container
│       ├── hooks/              # useDocuments, useUpload, useQuery
│       └── types/index.ts      # TypeScript interfaces
├── alembic/                    # Database migrations
├── tests/                      # pytest test suite
├── docker-compose.yml          # 3 services: db + app + frontend
├── Dockerfile                  # Python backend image
└── .env.example                # Environment template
```

## Development

```bash
# Run only the backend + db (without frontend)
docker compose up db app --build

# Run tests
docker compose exec app pytest

# Access Swagger API docs
open http://localhost:8000/docs
```

## Environment Variables

The LLM is reached over the OpenAI-compatible protocol, so any provider works — a hosted
endpoint like NVIDIA NIM, or a local Ollama / vLLM server — by changing configuration only.

The **judge** is configured separately from the **model under test** and is used solely by the
evaluation harness. It never serves user traffic. Keeping them apart matters: scoring a
model's answers with that same model measures self-consistency, not correctness.

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_MODEL` | No | `deepseek-ai/deepseek-v4-flash` | Model that answers user queries |
| `LLM_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | Any OpenAI-compatible endpoint (e.g. `http://host.docker.internal:11434/v1` for Ollama) |
| `LLM_API_KEY` | No | falls back to `NVIDIA_API_KEY` | Ignored by local servers |
| `JUDGE_MODEL` | No | `deepseek-ai/deepseek-v4-flash` | Model used by the evaluation harness |
| `JUDGE_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | Endpoint for the judge |
| `JUDGE_API_KEY` | No | falls back to `NVIDIA_API_KEY` | |
| `NVIDIA_API_KEY` | No | -- | Legacy single-key setting; still honoured as the fallback for both keys above. Free key from build.nvidia.com |
| `DATABASE_URL` | No | `postgresql+asyncpg://docagent:docagent@db:5432/docagent` | PostgreSQL connection |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Local embedding model |
| `INGESTION_STRATEGY` | No | `auto` | `unstructured` parse strategy: `auto`, `fast` (no OCR), `hi_res`, `ocr_only` |
| `CHUNK_SIZE` | No | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | No | `64` | Overlap tokens between chunks |
| `CRITIC_SIMILARITY_THRESHOLD` | No | `0.78` | Min confidence to accept an answer |
| `MAX_RETRIEVAL_ATTEMPTS` | No | `3` | Max retry loops for low-confidence answers |
