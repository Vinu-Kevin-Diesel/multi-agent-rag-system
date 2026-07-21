# CLAUDE.md

Guidance for working in this repository. Read this before making changes.

## What this project is

A multi-agent RAG (retrieval-augmented generation) system for document Q&A. Upload PDFs, DOCX,
HTML, or images; ask natural-language questions; get answers grounded in the source with
citations and a confidence score. A LangGraph state machine routes each query to a specialised
agent, and a critic scores the answer and re-retrieves if it's poorly grounded.

The distinguishing goal is that **the model that answers queries runs locally** on a consumer GPU
(via Ollama), behind a provider-agnostic OpenAI-compatible client — so the same code runs against
a hosted API (NVIDIA NIM) or a local model with only configuration changes. A separate hosted
model is reserved as the evaluation judge.

## Architecture

### Ingestion (`POST /api/ingest`) — `app/ingestion/`

1. **Layout detection** (`layout_detection.py`) — `unstructured` partitions the file into typed
   regions (Title, NarrativeText, Table, Image…). **OCR happens inside `unstructured`** (tesseract),
   governed by `INGESTION_STRATEGY` (`auto` | `fast` (no OCR) | `hi_res` | `ocr_only`). Blank
   regions are dropped here.
2. **Semantic chunking** (`chunking.py`) — sentences are grouped by embedding similarity;
   a new chunk starts when consecutive-sentence cosine similarity drops below 0.5 or the buffer
   exceeds `CHUNK_SIZE` (512) tokens, with `CHUNK_OVERLAP` (64) carried across.
3. **Embed + store** — chunks are embedded to 384-dim vectors (`all-MiniLM-L6-v2`, local, on CPU)
   and stored in pgvector with an HNSW cosine index (`m=16, ef_construction=64`).

There is **no separate OCR stage** — an earlier `ocr.py` existed but its pytesseract branch was
unreachable (the metadata key it gated on was never populated). It was removed; `unstructured`
already does OCR internally.

### Query (`POST /api/query`) — `app/agents/graph.py`

A LangGraph `StateGraph`, compiled once at startup (`get_agent_graph`, memoised; warmed in the
FastAPI lifespan). `AgentState` carries two distinct strings:

- `original_question` — what the user asked. **Written once, never mutated. Agents answer this.**
- `question` — the current search string. The critic loop may refine it. **Retrieval uses this.**

Keeping these separate is load-bearing: the critic rewrites the *search* query on a retry, and an
earlier bug let that rewrite become the question the agent *answered*.

Flow:

```
route → (factual/comparative: retrieve → agent)
      → (multihop: decompose → multi_retrieve → multihop agent)
   → critic → (confidence < threshold and attempts < max: refine → loop back) | done
```

- **route** (`router_agent.py`) — classifies `factual` / `comparative` / `multihop`. Output is
  constrained by a JSON schema; parsing tolerates both a JSON object and a bare word.
- **decompose** (`decompose.py`) — multi-hop only: splits the question into ≤4 sub-questions,
  also schema-constrained. Few-shot examples in the prompt carry the quality.
- **multi_retrieve** — one similarity search per sub-question, results **interleaved round-robin**
  (not concatenated per sub-question) so the context budget can't starve a later hop, then deduped.
- **agents** (`factual_agent.py`, `comparative_agent.py`, `multihop_agent.py`) — a system prompt
  each over the retrieved chunks; they answer `original_question`.
- **critic** (`critic_agent.py`) — `score_answer` is the **max cosine similarity between the answer
  embedding and the chunk embeddings** (local, ~0.05s). It measures topical grounding, not
  factual correctness — a fluent hallucination reusing source vocabulary scores high. Replacing it
  with an NLI/entailment check is planned.
- **refine** — asks the LLM for a better *search* query, derived from `original_question` (not the
  previous refinement, so drift doesn't compound). Loops back to retrieve/decompose.

### Retrieval — `app/retrieval/vector_store.py`

pgvector `<=>` cosine distance over the HNSW index; returns chunks best-first with a `similarity`
score. Optionally filtered to one `document_id`.

## Local inference (the important operational part)

`app/dependencies.py` is the **only** module that knows which provider we talk to. Everything else
sees an `AsyncOpenAI` client.

- `get_llm_client()` — the model under test (answers queries).
- `get_judge_client()` — the evaluation judge. **Never serves user traffic.** Kept on a strong
  hosted model on purpose: scoring a local model's answers with that same model measures
  self-consistency, not correctness.

Both are configured independently (`LLM_*` / `JUDGE_*`); `NVIDIA_API_KEY` is a legacy fallback for
both. Local servers ignore the API key, but the SDK requires one, so a placeholder is supplied.

### Ollama setup — `scripts/`

Ollama runs **natively on the host, not in Docker** (GPU passthrough into Compose on Windows is a
WSL2 detour; Ollama ships its own CUDA runtime). The app container reaches it via
`host.docker.internal:11434` (wired in `docker-compose.yml` via `extra_hosts`).

- `scripts/setup-ollama.ps1` — sets the env that matters and restarts Ollama.
- `scripts/build-router-model.ps1` — builds the `qwen3-router` variant (see below).

Environment that matters (the script sets these):

| var | value | why |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0` | Ollama binds `127.0.0.1` by default, which a container **cannot** reach |
| `OLLAMA_CONTEXT_LENGTH` | `16384` | fits the ~8–10k-token multi-hop prompts; small enough that weights + KV cache stay 100% on a 12 GB GPU |
| `OLLAMA_KEEP_ALIVE` | `-1` | keep the model resident across the 3–6 LLM calls per query |

### Setting up on a new machine

Copying the project folder is **not** sufficient. Four things live outside it:

1. **The Docker image** — built from the `Dockerfile`, never copied. `docker compose up --build -d db app`
   takes 5–10 minutes the first time (torch + `unstructured[all-docs]`) and needs internet.
2. **The database** — Postgres lives in the `pgdata` named volume, so it starts empty. Re-ingest:
   `docker compose run --rm app python eval/ingest_corpus.py`
3. **Ollama models** — `ollama pull qwen3:8b` for the base, then **build the router variant**:
   `qwen3-router` does not exist on ollama.com; `scripts/build-router-model.ps1` derives it from
   the installed base. Run the pull first.
4. **Host environment** — `scripts/setup-ollama.ps1` sets `OLLAMA_HOST` etc. and restarts Ollama.

`.env` is gitignored, so a `git clone` will not have one — copy `.env.example`. Note
`NVIDIA_API_KEY` is still required even when the LLM is local, because the judge runs on a
hosted model.

Also note the embedding model (`all-MiniLM-L6-v2`) is **not** baked into the image; it downloads
on first use, so the first run needs internet.

**The failure that wastes an hour:** without `OLLAMA_HOST=0.0.0.0`, Ollama binds `127.0.0.1`,
which a container cannot reach. Every query fails with a connection error while Ollama looks
perfectly healthy from a browser on the host.

### Hardware sizing (VRAM)

The defaults are tuned for a **12 GB** card. Measured footprints for `qwen3:8b` (Q4_K_M) via
`ollama ps`: weights ≈5.2 GB, KV cache ≈2.3 GB at 16384 ctx and ≈0.4 GB at 4096 ctx.

Always sanity-check with `ollama ps` after setup: **`PROCESSOR` must read `100% GPU`**. A
CPU/GPU split means the model spilled and everything will be several times slower.

| VRAM | `OLLAMA_CONTEXT_LENGTH` | `MAX_CONTEXT_TOKENS` | notes |
|---|---|---|---|
| 12 GB | `16384` | `8000` (default) | qwen3:8b ≈7.5 GB, fits fully |
| **8 GB** | **`8192`** | **`4000`** | qwen3:8b ≈6.3 GB; 16384 (7.5 GB) will spill once the desktop takes its share |
| 6 GB or less | use `qwen3:4b` | `4000` | an 8B won't fit with usable context; drop model size, not just context |

**Lower `MAX_CONTEXT_TOKENS` alongside the context window — they are coupled.** The app's budget
guard caps the chunks put into a prompt; if it stays at 8000 while Ollama's window drops to 8192,
a multi-hop prompt can exceed the window and Ollama will **silently truncate the front**, dropping
the source chunks and leaving the model to invent an answer. Keep the budget comfortably under the
window (roughly half) to leave room for the system prompt, question and generated answer.

**On 8 GB, avoid two resident models.** Two 8B models cannot co-reside even on 12 GB (see below),
and on 8 GB the swapping is punishing. Prefer:

```bash
ROUTER_MODE=classifier   # trained classifier: no LLM router at all, no second model for routing
ROUTER_MODEL=qwen3:8b    # decompose falls back to the main model — slower (~11s vs ~0.4s) but
                         # keeps a single model resident instead of swapping on every multi-hop query
```

That trades multi-hop decompose latency for the elimination of model swaps. On 12 GB, keep
`ROUTER_MODEL=qwen3-router` — the swap is cheaper than the reasoning penalty there.

### Non-obvious findings (don't re-derive these the hard way)

- **`think: false` does not work through Ollama's OpenAI `/v1` endpoint.** It's only honoured by
  the native `/api/chat`. qwen3 keeps reasoning, fills the token budget, and returns empty content
  — which is why the router (classification via `/v1`) needs a thinking-disabled *model*, not an
  API flag. `scripts/build-router-model.ps1` derives `qwen3-router` from the installed base and
  forces the no-think branch in the chat template (`IsThinkSet→true`, `Think→false`). Point the
  router at it with `ROUTER_MODEL=qwen3-router`. It's also used for decompose.
- **Ollama returns thinking in a field called `reasoning`, not `reasoning_content`.** Do **not**
  add `reasoning` to `extract_content`'s fallback chain — it holds the thinking, not the answer.
- **Router and decompose don't benefit from reasoning.** Classification is 1-of-3; decomposition is
  structured extraction guided by few-shot examples. The no-think model is ~30–50× faster with
  equal or better quality. Agents (final answers) keep the thinking model.
- **Two 8B models don't co-reside in 12 GB.** Main (7.5 GB at 16384 ctx) + `qwen3-router`
  (5.6 GB at 4096 ctx) ≈ 13 GB, so alternating them costs a model swap. `qwen3-router` is capped
  at `num_ctx 4096` (it only sees a one-line question) to keep it small. The trained classifier
  (planned) removes the second model entirely.

Measured latencies are tracked in `docs/performance.md` — update it when you change the pipeline.

## Configuration

`app/config.py` (`pydantic-settings`, read from `.env`; see `.env.example`). Notable settings:

- `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`, `JUDGE_*`, `ROUTER_MODEL`
- `INGESTION_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `CRITIC_SIMILARITY_THRESHOLD` (0.78), `MAX_RETRIEVAL_ATTEMPTS` (3), `MAX_CONTEXT_TOKENS` (8000)
- `LLM_TIMEOUT_SECONDS` (120), `LLM_MAX_RETRIES` (2) — the SDK default timeout is 600s
- **Ablation flags** (all default to the full system):
  - `ROUTER_MODE` — `llm` | `off` (off → everything to the factual agent)
  - `DECOMPOSE_ENABLED` — `true` | `false` (false → multi-hop single-retrieval)
  - `CRITIC_MODE` — `cosine` | `off` (off → one pass, no retry; confidence reported as 1.0)

The flags exist so the eval harness can measure each component's contribution. The graph topology
is unchanged by them — they alter node behaviour at runtime.

## Running and testing

Everything runs in Docker Compose (`db` = pgvector/postgres, `app` = FastAPI, `frontend` = Vite/React).

```bash
docker compose up --build            # full stack
docker compose up -d db app          # backend + db only

# tests (./tests is bind-mounted, so no rebuild needed to pick up edits)
docker compose run --rm --no-deps app pytest -q
docker compose run --rm --no-deps app pytest tests/test_graph_refine.py -q
```

For local inference, run `scripts/setup-ollama.ps1` on the host first and set the `LLM_*` /
`ROUTER_MODEL` block in `.env` (see `.env.example`).

### Evaluation corpus

`eval/corpus/` holds five synthetic policy documents (22 chunks) used for every eval run.
`eval/ingest_corpus.py` drops the database and re-ingests them deterministically; `--verify`
compares against the committed `corpus_manifest.json` and exits 1 on drift, so it can gate a run.

Determinism is checked by **SHA-256 of the chunk text**, not chunk count — a mutated corpus was
observed producing the same 22 chunks with a different hash. See `eval/README.md`.

### Testing conventions

- **Graph-level tests drive the real compiled graph** (`build_agent_graph().ainvoke(...)`) with
  collaborators stubbed via `monkeypatch` on `app.agents.graph.<name>` (the nodes resolve the
  module-global binding). See `test_graph_refine.py`, `test_graph_budget.py`, `test_ablation_flags.py`.
- **LLM clients are stubbed with `SimpleNamespace`, not bare `AsyncMock`** (`conftest.py`'s
  `make_llm_client`). A bare `AsyncMock` auto-vivifies any attribute, so a wrong-shaped mock fails
  silently; `SimpleNamespace` raises on a shape mismatch. The agents call
  `client.chat.completions.create` (OpenAI shape), never `client.messages.create`.
- **Tests must not depend on your `.env`.** The app service passes `env_file: .env` into the test
  container, so a local `ROUTER_MODE=classifier` once silently changed what the graph tests
  exercised. `conftest` pins the ablation flags per-test; keep it that way.
- **A regression test must be seen to fail on the pre-fix code.** Several tests here were verified
  by temporarily reverting the fix and confirming the assertion fails (e.g. the budget starvation
  test, the original-question test). A test that can't fail proves nothing.

## Conventions

- **Commits and authorship.** Commits are authored solely by the repository owner. Do **not** add
  `Co-Authored-By` trailers or "Generated with…" lines.
- **Branch per unit of work**, named for the problem (`feat/constrained-router-output`), not a day
  or ticket number. Open a PR; merge with a **merge commit, never squash** (squashing collapses the
  history). Don't merge without explicit approval.
- **`PLAN.md` is a local-only working document** (gitignored). Do not commit it.
- Match the surrounding code's style, comment density, and idiom. Comments explain *why*, and
  several encode a finding that cost real effort to discover — keep them.
- Update `docs/performance.md` when a change affects latency, and `.env.example` when a setting is
  added.

## Current state

The query pipeline runs locally and is instrumented. Router and decompose are schema-constrained
and run on the no-think model; the context budget, per-node latency logging, and bounded LLM
timeouts are in place; ablation flags are wired.

Planned (not yet built): a trained classifier to replace the LLM router (removing the second
model), an NLI-based critic, a frozen eval corpus and golden question set, and a RAGAS evaluation
harness with an ablation study across the flag configurations. The judge client already exists for
that harness.
