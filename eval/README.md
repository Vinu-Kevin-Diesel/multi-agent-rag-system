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

40 items so far: **16 factual, 13 comparative, 11 multi-hop**. Batch 1 was deliberately
factual-heavy; batch 2 weights comparative and multi-hop to pull the mix toward balance. Batch 3
tops it up to ~50, roughly even across the three types.

Every multi-hop item's answer genuinely spans **two or more documents** — neither source answers it
alone. The canonical shape: Zeltavir is Tier 4 (PA-2291) → Tier 4 costs 20% coinsurance under
GoldCare HMO (BEN-2025). A question that is merely long is not multi-hop, and padding the set with
those would make the decompose ablation look better than it is.

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

## Known limitations

22 chunks is small. With `top_k=5` a query retrieves roughly a fifth of the corpus, so
`context_precision` is an easier problem here than against a production-scale store. The corpus is
sized for a hand-verifiable golden set, not for retrieval stress-testing — worth stating alongside
any metric taken from it.

The golden set is synthetic and single-corpus. It measures whether the pipeline reads *these*
documents correctly; it says nothing about domain generalisation. Both figures — 50 items, one
corpus — belong next to any headline metric.
