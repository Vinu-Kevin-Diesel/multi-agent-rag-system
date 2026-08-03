# Autonomous Document Intelligence Agent

A multi-agent RAG system for document Q&A: upload PDFs, DOCX, HTML or images, ask
natural-language questions, get answers grounded in the source with citations and a confidence
score. A LangGraph state machine routes each query to a specialised agent, and a critic scores the
answer and re-retrieves when it looks poorly grounded.

**The model that answers queries runs locally**, on a consumer laptop GPU, behind a
provider-agnostic OpenAI-compatible client — the same code runs against a hosted API or a local
model with only configuration changes. A separate hosted model is reserved as the evaluation
judge, never as the model under test.

Every component was then measured to see whether it earns its keep. Two of them do not.

## Results

Measured on an **RTX 4060 Laptop (8 GB)** running **qwen3:8b at Q4_K_M**, fully on-GPU, against a
frozen 5-document corpus and a 50-question hand-verified golden set. Marginal cost per query: **$0**.

Each configuration changes exactly one component from the row above it. The `--config` label is
verified against the app's live flags at collection time, so a run cannot be silently mislabelled.

| config | router | decompose | critic | routing acc. | multi-hop acc. | latency (mean) |
|---|---|---|---|---|---|---|
| `baseline` | off | off | off | 34% | 0/16 | 20.5s |
| `+router` | LLM | off | off | **80%** | 12/16 | 19.1s |
| `+decompose` | LLM | on | off | 80% | 12/16 | 18.2s |
| `full` | LLM | on | cosine | 80% | 12/16 | **55.0s** |
| `full-clf` | classifier | on | cosine | 66% | **5/16** | 55.4s |
| `full-nli` | LLM | on | NLI | 78% | 12/16 | 81.1s |

RAGAS scores for `full` (judge: `deepseek-v4-flash` via NVIDIA NIM, never the local model):

| metric | overall | factual | comparative | multi-hop |
|---|---|---|---|---|
| `faithfulness` | 0.755 | 0.744 | 0.797 | 0.725 |
| `answer_relevancy` | 0.896 | 0.943 | 0.873 | 0.868 |
| `context_precision` | 0.732 | 0.802 | 0.811 | **0.539** |
| `context_recall` | 0.934 | 1.000 | 1.000 | **0.797** |

### What the ablation found

**The router earns its keep; the trained classifier does not.** Routing carries the whole jump
from 34% to 80%, and multi-hop questions are unreachable without it — `baseline` sends everything
to the factual agent and answers 0 of 16 correctly. But the MiniLM+LogReg classifier built to
*replace* the LLM router is 14 points worse (66%), and its failure is directional rather than
diffuse: it sends **11 of 16 multi-hop questions to the factual agent**, collapsing exactly the
class the decompose path exists to serve. It reported F1=0.96 on a held-out split of synthetic
template questions; on hand-written ones it does not hold up. **The classifier is the component
this evaluation would delete.**

**Multi-hop loses in retrieval, not in answering.** `context_precision` falls to 0.539 and
`context_recall` to 0.797 on multi-hop, against ~0.80 and a perfect 1.000 elsewhere — while
`faithfulness` stays flat across all three types (0.73–0.80). The agent handles worse context
about as well as it handles good context; the deficit is upstream.

**The critic costs 3× latency and its confidence predicts nothing.** Turning it on (`+decompose` →
`full`, one flag) takes mean latency from 18.2s to 55.0s, with 62% of queries exhausting all three
retries — mean confidence 0.737 sits below the 0.78 threshold, so the loop usually runs to
exhaustion instead of converging. And the score itself does not track groundedness: Spearman
between `critic.confidence` and RAGAS `faithfulness` is **+0.207 (p=0.163, n=47)** — not
distinguishable from zero. Cosine similarity measures topical overlap, so a fluent answer reusing
source vocabulary scores high whether or not the source supports it.

**Replacing it with entailment did not fix the correlation.** An NLI cross-encoder critic
(`critic_mode=nli`) is a far better discriminator in isolation — on a true/false pair it separates
1658× where cosine manages 2.1× — and spans a 0.96 score range against cosine's compressed 0.24.
Paired on the 28 items scored in both runs, the correlation went **+0.205 → +0.157**, both null.
Mean faithfulness rose 0.788 → 0.874, but a sign test gives p=0.332, so that is not claimed either.

The likely reason is a flaw in the measurement rather than the metric: **the recorded confidence
is the score of the *accepted* answer**, and the retry loop stops as soon as a score clears the
threshold — conditioning on the very variable being measured. On rows that exhausted all retries
(never selected on score), NLI shows +0.365 against cosine's +0.116. That is a post-hoc subgroup
at n=21, p=0.102, so it is a hypothesis, not a result. Settling it needs a measure-but-do-not-act
critic mode; see [Limitations](#limitations).

Reproduce any of this with [`scripts/reproduce-eval.ps1`](scripts/reproduce-eval.ps1); the method
is documented in [`eval/README.md`](eval/README.md).

## Limitations

Stated plainly, because several of the numbers above would be easy to over-read.

- **50 questions, one corpus.** The golden set is 50 hand-verified items over five synthetic
  policy documents. It measures whether the pipeline reads *these* documents correctly and says
  nothing about domain generalisation. At n≈30–50 the evaluation can see a large effect and cannot
  resolve a small one — which is exactly why the critic comparison is reported as unresolved
  rather than as a win for either scorer.
- **22 chunks is a small store.** With `top_k=5` a query retrieves roughly a fifth of the corpus,
  so `context_precision` is an easier problem here than against a production-scale index.
- **Judge-model bias.** All RAGAS metrics come from one hosted judge (`deepseek-v4-flash`). A
  single judge has its own preferences, and metrics like `faithfulness` inherit them. The judge is
  deliberately never the model under test — scoring a local 8B with that same 8B measures
  self-consistency, not correctness — but one judge is not a panel.
- **Uneven scoring coverage.** The judge runs on a free tier that rate-limits and exhausts. `full`
  scored 47/50 but the NLI run only 30/50, so those two correlations differ in statistical power
  as well as in value; every comparison that matters is computed on the *common* subset. Judge
  failures were also observed to cluster by query type, so per-type breakdowns are suppressed
  below 80% coverage rather than computed over whichever rows happened to finish.
- **The critic question is open.** The confidence-vs-faithfulness measurement is confounded by the
  retry loop selecting on the score. A `critic_mode` that scores without retrying would settle it;
  it is not built.
- **Latency is laptop-class.** ~19s per query without the critic, ~55s with it, on an 8 GB mobile
  GPU with a Q4 8B model. Numbers are not comparable to a hosted frontier model.

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
                    |   Critic Agent   |  Cosine similarity, or NLI entailment
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
- **Self-correcting retrieval** -- critic agent scores answers and triggers re-retrieval with
  refined queries if confidence is low. Costs 3× latency; whether the confidence score is
  meaningful is [unresolved](#what-the-ablation-found)
- **Cross-document search** -- query across all ingested documents simultaneously
- **Source attribution** -- every answer includes source chunks with page numbers and similarity scores
- **Runs on your own GPU** -- the answering model, embeddings and both critics are local, so a
  query costs nothing per call. The hosted judge is used only by the evaluation harness
- **Measured, not asserted** -- a frozen corpus, a 50-question hand-verified golden set, a
  six-configuration ablation and a RAGAS harness, all reproducible from one script

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
# Edit .env — add an NVIDIA_API_KEY for the hosted path, or configure a local
# model (see "Running with a local model" below)

# Start all services (db + backend + frontend)
docker compose up --build
```

Three services will start:
- **Frontend**: http://localhost:3000 (React UI)
- **Backend API**: http://localhost:8000 (FastAPI + Swagger at /docs)
- **Database**: PostgreSQL 16 + pgvector on port 5432

### Running with a local model (Ollama)

The model that answers queries can run entirely on your own GPU — no API cost, no data
leaving the machine. Ollama runs **natively on the host**, not in Docker: GPU passthrough
into Compose on Windows is a WSL2 detour, and Ollama ships its own CUDA runtime.

```powershell
# One-time host setup: sets the env vars that matter, restarts Ollama, pulls the model
./scripts/setup-ollama.ps1
```

Then point the app at it in `.env`:

```bash
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama                      # ignored by Ollama; the OpenAI SDK just needs a value
```

The container reaches the host server through `host.docker.internal` (wired into
`docker-compose.yml`). The **judge** stays on a hosted model — see [Environment Variables](#environment-variables).

Two settings decide whether this works well:

- **`OLLAMA_HOST=0.0.0.0`** — Ollama binds to `127.0.0.1` by default, which the container
  *cannot* reach. Without this, every request fails with a connection refused. The setup
  script handles it.
- **`OLLAMA_CONTEXT_LENGTH=16384`** — big enough for the ~8–10k-token prompts the multi-hop
  path builds, small enough that weights + KV cache fit fully in 12 GB VRAM. Too large and
  the model spills to CPU (`ollama ps` shows a CPU/GPU split instead of `100% GPU`); too
  small and long prompts are silently truncated, dropping your source chunks.

Two measured reference points, both with `qwen3:8b` fully on GPU:

| GPU | context | `MAX_CONTEXT_TOKENS` | query latency |
|---|---|---|---|
| RTX 5070, 12 GB | 16384 | 8000 | ~30–40s |
| RTX 4060 Laptop, 8 GB | 8192 | 4000 | ~19s without the critic, ~55s with it |

The 8 GB row is the hardware every number in [Results](#results) was measured on. Run
`scripts/setup-ollama.ps1` — it sizes the context window from the detected VRAM rather than
assuming, and prints the matching `.env` block.

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

### Reproducing the evaluation

```bash
# Everything that needs no API key: corpus + golden-set gates, all six configurations,
# the ablation table and the router confusion matrices.
./scripts/reproduce-eval.ps1 -SkipScore

# Add RAGAS scoring (needs a judge key; free tiers rate-limit and exhaust).
./scripts/reproduce-eval.ps1

# Re-score existing runs, e.g. once judge quota resets.
./scripts/reproduce-eval.ps1 -SkipCollect
```

Collection is GPU-bound and costs nothing but time (~50 min per configuration); scoring depends on
a hosted judge and is the part that fails on a free tier. They are separate steps on purpose — a
quota failure should never cost hours of GPU work. See [`eval/README.md`](eval/README.md) for the
method, the metric definitions, and the failure modes worth knowing about before trusting a number.

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
