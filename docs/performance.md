# Latency baseline

Measured end-to-end against the local model, model already warm. This is the **before** for
the optimisation work that follows; every later change is measured against these numbers.

**Setup:** qwen3:8b (Q4_K_M) on an RTX 5070 12 GB via Ollama, 16384 context, resident 100% on
GPU. Embeddings run on CPU (`all-MiniLM-L6-v2`). Backend in Docker, model server on the host.

## Factual query

`"What must happen before Drug A is authorized?"` — `top_k=4`, 1 attempt, confidence 0.915.

| node | time | |
|---|---:|---|
| `route_node` | 17.21s | **45% of the query, to choose one word from three** |
| `retrieve_node` | 1.87s | pgvector HNSW + query embedding |
| `factual_node` | 19.23s | answer generation |
| `critic_node` | 0.05s | local embeddings — effectively free |
| **total** | **38.36s** | |

## Multi-hop query

`"What must a patient complete before Drug A is authorized, and how long does that
authorization then last?"` — `top_k=6`, 1 attempt, confidence 0.822.

Measured with the router **forced** to `multihop`, because it currently misroutes this
question to `factual` (see below). Router time therefore excluded.

| node | time | |
|---|---:|---|
| `decompose_node` | 10.51s | → 2 sub-questions |
| `multi_retrieve_node` | 1.64s | 2 sub-questions → 6 unique chunks, no trim |
| `multihop_node` | 14.63s | chain-of-thought answer |
| `critic_node` | 0.06s | |
| **total** | **26.87s** | excludes the router |

## What the breakdown says

**The router is the single worst component, and it is also broken.** It costs 17–27s to make a
three-way classification, and on hard questions it returns nothing at all:

```
Q='What must happen before Drug A is authorized?'                    raw_response='factual'
Q='What must a patient complete ... and how long does it last?'      raw_response=''
                                                                     -> factual (no match found)
```

`max_tokens=512` is too small for a thinking model. On an easy question qwen3 finishes thinking
and emits the category. On a harder one it spends the entire budget *still thinking*, generation
is cut off before any answer token, and `extract_content` returns `""` — so the router falls
through to its `factual` default.

The failure is invisible from outside and inverted: the router works on the questions that don't
need routing and fails on the ones that do. A question taken almost verbatim from the router's
own few-shot examples still classified as `factual`.

**The critic is free.** ~0.05s, because it scores with local embeddings rather than an LLM call.
Whatever else is worth optimising, this is not.

**The context budget is not binding.** The multi-hop path produced 6 chunks against an 8000-token
budget, so no trim occurred — the guard is a safety rail against unbounded `top_k` and
sub-question fan-out, not a routine cost.

**Retrieval is negligible.** ~2s including the query embedding; the rest is generation.

---

# After: constrained router (day 6)

The router now uses a schema-constrained response and a thinking-disabled qwen3 variant
(`qwen3-router`, see `scripts/build-router-model.ps1`). Thinking cannot be disabled through
Ollama's OpenAI `/v1` endpoint, so the fix lives in the model, not the code.

| | day-5 router | day-6 router | |
|---|---:|---:|---|
| latency (warm, resident) | 17.21s | **0.35s** | ~49× faster |
| hard-question correctness | silently `factual` | correct `multihop` | the failure is gone |

**Correctness was the real win.** The exact multi-hop question that day 5 misrouted to `factual`
with an empty response now routes `multihop`, and the decompose → multi-retrieve → multihop path
— **previously unreachable through the API** — runs end to end.

End-to-end totals, model warm:

| | day-5 | day-6 | note |
|---|---:|---:|---|
| factual | 38.36s | **17.36s** | router 17s → ~2s |
| multi-hop | 26.87s* | **19.39s** | *day-5 excluded the router; it couldn't route this at all |

**The catch: two 8B models do not co-reside in 12 GB.** The main model at 16384 context is 7.5 GB;
`qwen3-router` (context capped at 4096) is 5.6 GB — together 13 GB, so loading one evicts the other.
Each query therefore pays a ~5 s model-swap when it first hits the router (visible as
`route_node 5.43s` above, versus 0.35 s resident). Still a clear net win, but not free. Options if
it bites: fold routing into a single no-think model for everything (pending an answer-quality check
under RAGAS), or move routing off the LLM entirely — which is exactly what the trained classifier
on day 10 does.

**Decompose is still slow (~11s)** because it remains on the thinking model — day 7 gives it the
same treatment as the router.
