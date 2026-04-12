# Autonomous Document Intelligence Agent

A multi-agent RAG system that ingests heterogeneous documents through a 3-stage pipeline and routes queries to specialized sub-agents using LangGraph, Claude API, FastAPI, and pgvector.

## Architecture

```
                    +------------------+
                    |   POST /ingest   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Layout Detection |  Stage 1: unstructured
                    +--------+---------+
                             |
                    +--------v---------+
                    |       OCR        |  Stage 2: pytesseract
                    +--------+---------+
                             |
                    +--------v---------+
                    | Semantic Chunking |  Stage 3: sentence-transformers
                    +--------+---------+
                             |
                    +--------v---------+
                    |  pgvector (HNSW) |
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
                        YES -> re-retrieve with refined query
                        NO  -> return answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Claude API (Anthropic) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | PostgreSQL 16 + pgvector (HNSW) |
| API | FastAPI |
| Document Parsing | unstructured, pytesseract |
| Chunking | sentence-transformers |
| ORM | SQLAlchemy (async) |
| Migrations | Alembic |
| Containerization | Docker Compose |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Anthropic API key
- OpenAI API key (for embeddings)

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/autonomous-document-intelligence-agent.git
cd autonomous-document-intelligence-agent

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### Ingest a Document

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@report.pdf"
```

Response:
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "report.pdf",
  "num_chunks": 42,
  "page_count": 12
}
```

#### Query Documents

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Q3 revenue?"}'
```

Response:
```json
{
  "answer": "The Q3 2024 revenue was $5.2 billion...",
  "query_type": "factual",
  "confidence": 0.91,
  "sources": [
    {
      "chunk_id": "...",
      "content": "...",
      "page_number": 3,
      "similarity": 0.94
    }
  ],
  "retrieval_attempts": 1
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run migrations
alembic upgrade head

# Run tests
pytest

# Run server locally
uvicorn app.main:app --reload
```

## Project Structure

```
app/
├── main.py              # FastAPI application
├── config.py            # Settings (pydantic-settings)
├── models.py            # SQLAlchemy models + pgvector
├── database.py          # Async DB session
├── schemas.py           # Request/response models
├── ingestion/
│   ├── pipeline.py      # 3-stage orchestrator
│   ├── layout_detection.py
│   ├── ocr.py
│   └── chunking.py
├── agents/
│   ├── graph.py         # LangGraph StateGraph
│   ├── router_agent.py  # Query classifier
│   ├── factual_agent.py
│   ├── comparative_agent.py
│   ├── multihop_agent.py
│   └── critic_agent.py  # Answer validation
├── retrieval/
│   └── vector_store.py  # pgvector HNSW search
└── utils/
    └── embeddings.py    # Embedding helper
```
