# Evaluation corpus

A frozen document set used for every evaluation run, so results are comparable across
configurations and across time.

## Contents

`corpus/` — five synthetic insurance/pharmacy policy documents (~22 chunks total):

| document | ref | what it carries |
|---|---|---|
| `zeltavir-policy.md` | PA-2291 | Tier 4 drug: prior auth, **test dose**, 12-month authorization, ICD-10 codes |
| `morvanex-policy.md` | PA-2304 | Tier 3 drug: **no** test dose, step therapy, 6-month authorization |
| `plan-benefits.md` | BEN-2025 | deductibles, tier→copay table, service cost sharing, network rules |
| `appeals-procedures.md` | ADM-118 | appeal windows, expedited criteria, administrative vs clinical denials |
| `specialty-pharmacy.md` | RX-540 | specialty network, first-fill rules, claim rejection codes |

**Why synthetic.** No licensing questions, and — more importantly — the ground truth is known
exactly, which is what makes a trustworthy golden question set possible.

**Designed for all three query types.** The documents deliberately cross-reference each other so
that genuine multi-hop questions exist, not just questions that happen to be long:

- *factual* — a single lookup ("how long is Zeltavir authorization valid?")
- *comparative* — the two drug policies and the two plans contrast on purpose (test dose vs none,
  12 vs 6 months, deductible vs premium)
- *multi-hop* — a chain across documents: Zeltavir is **Tier 4** (PA-2291) → Tier 4 costs **20%
  coinsurance** under GoldCare HMO (BEN-2025). Neither document answers that alone.

## Deterministic ingest

```bash
docker compose up -d db
docker compose run --rm app python eval/ingest_corpus.py            # ingest + write manifest
docker compose run --rm app python eval/ingest_corpus.py --verify   # exit 1 on any drift
```

The script drops every document, re-ingests `corpus/` in sorted order, and records a per-document
chunk count **and SHA-256 of the chunk text** in `corpus_manifest.json`.

The hash matters more than the count. A mutated corpus was verified to produce the *same* 22
chunks with a *different* hash — a count-only check would have passed it. `--verify` exits 1 on
drift and 0 when clean, so it can gate an eval run in CI.

Document IDs are `uuid4` and differ between runs, so they are excluded from the hash. The harness
queries across all documents, so IDs are irrelevant; chunk text and ordering are not.

## Golden question set

`golden_set.jsonl` — one JSON object per line, the questions every eval run is scored against.

| field | meaning |
|---|---|
| `id` | stable identifier (`gs-001`…), never reused or renumbered |
| `question` | asked verbatim over HTTP by the harness |
| `query_type` | gold routing label — `factual` \| `comparative` \| `multihop` |
| `ground_truth` | the correct answer, for RAGAS `answer_correctness` / `context_recall` |
| `expected_docs` | corpus filenames that contain the answer |

**50 items, `gs-001`…`gs-050`, balanced 17 factual / 17 comparative / 16 multi-hop.** Built in
three batches (factual-heavy first, then comparative + multi-hop, then a balancing top-up); the
batch boundaries no longer matter now that it is complete.

Every multi-hop item's answer genuinely spans **two or more documents** — neither source answers it
alone. The canonical shape: Zeltavir is Tier 4 (PA-2291) → Tier 4 costs 20% coinsurance under
GoldCare HMO (BEN-2025). A question that is merely long is not multi-hop, and padding the set with
those would make the decompose ablation look better than it is — so it is a hard rule the validator
enforces, not a guideline.

### Validation

`validate_golden_set.py` is the structural gate. No services required — it reads files only:

```bash
python eval/validate_golden_set.py            # exit 0 if valid, 1 on the first problem
```

It checks: every row is valid JSON with exactly the five required fields; ids are the contiguous
run `gs-001..gs-050`, unique; `query_type` is one of the three; `question` and `ground_truth` are
non-empty; questions are unique; every `expected_docs` entry resolves to a file in `corpus/`; and
every `multihop` row names ≥ 2 documents.

It is structural only. It **cannot** check that a `ground_truth` is *correct* — that is done by eye
against the source document, and is the one step no script replaces.

**Every `ground_truth` was read out of the source document, not generated.** A hallucinated ground
truth produces confident, precise, entirely fictional metrics, and nothing downstream will flag it.

`query_type` is gold for two purposes at once: RAGAS scores broken down per type, and the router
confusion matrix (LLM router vs. trained classifier vs. this label).

### Why `expected_docs` and not `expected_pages`

The corpus is markdown, which carries no pagination, so `page_number` is not a usable locator here
— `SourceChunk.page_number` is nullable for exactly this reason. Document attribution is what can
actually be verified by eye and what retrieval failures are diagnosed against, so the field records
filenames instead. Note the query API returns `chunk_id`/`content` but not a filename, so this
field documents and diagnoses; it is not matched automatically by the harness.

## Collection harness

`run_eval.py` runs the golden set against a **running** instance and records raw results for
scoring. It hits `POST /api/query` over HTTP on purpose — it evaluates the system as shipped (API,
compiled graph, real vector store, local model under the running configuration), not an imported
graph that would bypass all of that.

```bash
docker compose up -d db app                              # app must be reachable
docker compose run --rm app python eval/ingest_corpus.py --verify   # populate + gate the corpus
docker compose exec -T app python eval/run_eval.py --config full     # collect a run
```

- `--config` is a **label only**. It does not change the system — the ablation flags live in the
  running app's environment (`ROUTER_MODE`, `DECOMPOSE_ENABLED`, `CRITIC_MODE`). The day-17 sweep
  restarts the app per configuration and runs this with a matching `--config`. No endpoint reports
  those flags, so the label is trusted, not verified — set it to match what you booted.
- Output: `runs/{config}_{timestamp}.jsonl`, **append-only, never overwritten** — a fresh
  timestamped file per run. Rows are flushed as they complete, so an interrupted run keeps its
  partial results. `runs/` is git-ignored (generated data); the scored CSVs are added explicitly.
- Each row carries the gold fields (`question`, `gold_query_type`, `ground_truth`, `expected_docs`)
  next to the prediction (`answer`, `predicted_query_type`, `confidence`, `retrieval_attempts`,
  `contexts` = `sources[].content`, `latency_s`), so the scoring step needs no join back. The four
  RAGAS inputs — question, answer, contexts, ground_truth — are present on every row.
- A failed query is recorded with an `error` string and the run continues; one bad item never
  abandons the other 49. A summary (routing accuracy, latency, mean confidence) prints at the end.
- `--limit N` runs only the first N items — a fast smoke check before committing to a ~45-minute
  50-item sweep on an 8 GB laptop (measured ~50 s/item).

## Scoring with RAGAS

`run_ragas.py` scores a collected run. Collection and scoring are separate steps on purpose:
collection is slow (~50 s/item on a local 8B) and scoring costs hosted-judge calls, so a bug in
one must never force you to redo the other.

```bash
docker compose run --rm --no-deps app python eval/run_ragas.py --latest
docker compose run --rm --no-deps app python eval/run_ragas.py --run eval/runs/full_....jsonl --limit 3
```

Writes `{run_stem}_scored.csv` next to the input, one row per item, and prints the aggregate.

| metric | what it catches |
|---|---|
| `faithfulness` | answer not supported by the retrieved contexts (hallucination) |
| `answer_relevancy` | answer does not address the question |
| `context_precision` | retrieved chunks are noise |
| `context_recall` | retrieval missed what the ground truth needs |
| `answer_correctness` | answer disagrees with the ground truth |

**The judge is never the model under test.** It resolves from `JUDGE_*` (falling back to
`NVIDIA_API_KEY`) exactly as `app/dependencies.py` does, and runs on a strong hosted model.
Scoring a local 8B with that same 8B measures self-consistency, not correctness — the numbers
would look fine and mean nothing.

**Embeddings run locally**, on the same MiniLM the app uses, through a small
`langchain_core.embeddings.Embeddings` adapter over `sentence-transformers` (already a dependency,
so no extra package). RAGAS makes many embedding calls for `answer_relevancy` and
`answer_correctness`; routing those through the hosted API would spend the free tier's rate-limit
budget on embeddings instead of on judging. Judge concurrency defaults to 3 with 10 retries for
the same reason — a rate-limited run that dies halfway wastes the whole collection.

The scored CSV carries `id`, `gold_query_type`, `predicted_query_type`, `confidence`,
`retrieval_attempts` and `latency_s` alongside the metrics, so the day-18 analysis (per-type
breakdown, router confusion matrix, critic-vs-faithfulness correlation) needs no join back.

### Judge reliability — read before trusting a metric

The judge is a hosted model on a shared free tier, and it fails in a way that **biases results
rather than announcing itself**. A metric whose judge call ultimately fails comes back `NaN`, and
`.mean()` skips it silently — so the script always prints the denominator (`32/50 scored`). Treat a
metric with partial coverage as provisional, never as a headline number.

Two failure modes, both observed on the first full run:

- **`503 ResourceExhausted: Worker local total request limit reached (N/48)`** — NIM's *shared*
  concurrency ceiling. It fires regardless of our own `--workers`, so retries are mandatory. Keep
  them bounded at one layer: the SDK owns retries (it honours `Retry-After`), `RunConfig` keeps a
  small outer retry. Configuring both at 10 made a 3-item job sit at 0% CPU for 15 minutes.
- **Timeouts on long answers.** `faithfulness` extracts statements from an answer and judges each
  one, so cost scales with answer length. Comparative answers here have a median of ~1350
  characters against ~334 for factual — 4×. At `--timeout 300` this produced coverage of 15/17
  factual but only **5/17 comparative**.

That second one is the dangerous one: the losses are **concentrated by query type**, so a per-type
breakdown computed from a partially-scored run is a selection-biased artefact — the comparative
mean would be taken over whichever comparative answers happened to be short. Raise `--timeout` for
long-answer configurations and check the printed coverage before reporting anything per type.

### Cost, and what the ablation sweep drops

Measured on this hardware (RTX 4060 8 GB laptop, judge on NIM's free tier), per configuration:

| stage | cost |
|---|---|
| collection (50 items, local 8B) | ~47 min |
| scoring (5 metrics × 50 items = 250 judge jobs) | ~3.5 h |

Scoring dominates, and not because of our concurrency — NIM's shared 503 ceiling serialises the
workers down to an effective ~1. Five configurations at that rate is ~20 hours, which does not fit
a day.

**The sweep therefore scores four metrics, dropping `answer_correctness`** (`--metrics
faithfulness,answer_relevancy,context_precision,context_recall`). It is the most expensive metric
— it decomposes both answer and ground truth into statements before comparing — and was the least
reliable, at 28% coverage on the first full run. The ablation question ("does each component earn
its keep?") is still answered: `faithfulness` and `answer_relevancy` cover generation quality,
`context_precision`/`context_recall` cover retrieval, which is what the router and decompose flags
actually move. The headline `full` run keeps all five.

### Dependency note

`ragas 0.4.3` hard-imports `langchain_community.chat_models.vertexai`, which `langchain-community`
0.4.x removed — so `import ragas` fails outright against a current one. `pyproject.toml` pins
`langchain-community<0.4`; 0.3.31 still allows `langchain-core<2.0,>=0.3.78`, so `langchain-core`
stays at 1.5.2 and `langgraph` is untouched. That matters beyond convenience: **the eval's own
dependencies must not alter the graph being evaluated.**

## Known limitations

22 chunks is small. With `top_k=5` a query retrieves roughly a fifth of the corpus, so
`context_precision` is an easier problem here than against a production-scale store. The corpus is
sized for a hand-verifiable golden set, not for retrieval stress-testing — worth stating alongside
any metric taken from it.

The golden set is synthetic and single-corpus. It measures whether the pipeline reads *these*
documents correctly; it says nothing about domain generalisation. Both figures — 50 items, one
corpus — belong next to any headline metric.
