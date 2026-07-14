# 3-Week Plan — Local Inference + RAGAS Evaluation

**Day 1: Mon 13 Jul 2026 → Day 21: Sun 2 Aug 2026**

**Goal:** move the system off a hosted API onto a local 8B model (RTX 5070, 12GB), remove every place the
pipeline blindly trusts the LLM to behave, and prove with RAGAS that each component earns its keep.

## Ground rules

- **Every day ends green.** The app boots, tests pass, and the commit is a self-contained reviewable unit.
- **No streak-padding.** If a day produces nothing real, commit nothing and eat a buffer day.
- **Buffer days are 7, 14, 21.** Do not pre-spend them. Something will slip — probably Day 4 or Day 16.
- Commit style follows the repo's existing convention: `feat:` / `fix:` / `refactor:` / `docs:` / `chore:` / `data:` / `eval:`.

## Branch workflow

**One branch per day.** Branch off `main` at the start of the day, push at the end of it, open a PR.

Name the branch after **the problem being solved**, never the day number — the history should make sense to someone who has never seen this plan:

```
feat/provider-agnostic-llm-client      not  day-03-...
feat/local-inference-ollama            not  day-04-...
feat/constrained-router-output         not  day-06-...
```

- Each change within the day is its own commit.
- **Merge with a merge commit — never squash.** Squashing collapses the day's commits into one and destroys the dated history the GitHub contribution graph reads.
- Commits only count toward the contribution graph once they land on `main`. A pushed branch alone shows nothing.

*(Days 1-2 predate this rule and shared `fix/pipeline-correctness` — PR #5.)*

---

# Week 1 — Correctness, then local inference (Jul 13–19)

### Day 1 (Mon 13 Jul) — Split `question` from `original_question`

- [ ] Add `original_question: str` to `AgentState` in `app/agents/graph.py`
- [ ] Set it once in `app/routers/query.py`; no node ever writes it again
- [ ] `factual_node` / `comparative_node` / `multihop_node` answer `original_question`
- [ ] `retrieve_node` / `decompose_node` / `multi_retrieve_node` still consume the refinable `question`
- [ ] `route_node` classifies `original_question`
- [ ] `generate_refined_query` takes `original_question` (not the drifted one)
- [ ] Fallback branch in `refine_query_node` falls back to `original_question`, not the drifted `question`
- [ ] **Write the first graph-level test** — the one that would have caught this bug

**Commit:** `fix: preserve original question across critic refinement loop`
**Done when:** a query forced into ≥2 retries returns an answer addressing what you actually asked, with a test that fails on the old code.

### Day 2 (Tue 14 Jul) — Delete the dead OCR stage

- [x] ~~Fix the vestigial `mock_anthropic` fixture (stubs `client.messages.create`, Anthropic-style, while every agent calls `client.chat.completions.create`)~~ — **pulled forward to Day 1**, `453f0f1`. Also bind-mounted `./tests` (`5156dcf`); without it, editing a test silently re-ran the copy baked into the image.
- [x] Remove `app/ingestion/ocr.py` and its call in `pipeline.py` (`3a3d636`) — its one live behaviour, dropping blank regions, moved into `detect_layout` with its test
- [x] Set an explicit `strategy` in `detect_layout` (`e20ac3f`) — now `settings.ingestion_strategy`, default `auto`
- [x] Compile the LangGraph once at startup, not per-request (`9904b0e`) — memoised, warmed in the lifespan hook
- [x] Fix the README's Stage 2 claim (`ddc29b3`)
- [x] Drop `pytesseract` / `Pillow` as direct deps (`e707236`) — unstructured OCRs via its own `unstructured-pytesseract` fork

**Done:** 12 tests green. **OCR now genuinely runs for the first time** — verified `detect_layout()` reading a PNG with no text layer via tesseract 5.5.0. It was advertised in the README, wired into the pipeline, and unreachable.

### Day 3 (Wed 15 Jul) — Provider-agnostic client

- [ ] `nvidia_api_key` becomes optional (`str | None = None`)
- [ ] Add `llm_api_key` / `llm_base_url` / `llm_model` settings
- [ ] Add `get_judge_client()` in `dependencies.py`, pointed at NIM (it becomes the RAGAS judge)
- [ ] Add `judge_model` / `judge_base_url` / `judge_api_key` settings

**Commit:** `refactor: decouple LLM client from NVIDIA NIM, add judge client`
**Done when:** still works on NIM, but nothing outside `dependencies.py` knows the provider's name.

### Day 4 (Thu 16 Jul) — Ollama live ⚠️ RISK

- [ ] Install Ollama on the Windows host (NOT in Docker)
- [ ] `ollama pull qwen3:8b`
- [ ] `OLLAMA_HOST=0.0.0.0`
- [ ] `OLLAMA_KEEP_ALIVE=-1`
- [ ] **`OLLAMA_CONTEXT_LENGTH=16384`** ← the one that silently ruins everything if wrong
- [ ] `OLLAMA_NUM_PARALLEL=2`
- [ ] Compose: `LLM_BASE_URL=http://host.docker.internal:11434/v1` + `extra_hosts: ["host.docker.internal:host-gateway"]`

**Commit:** `feat: run inference locally via Ollama (qwen3:8b)`
**Done when:** full ingest→query round-trip with **no NIM key in `.env`**; `nvidia-smi` shows the model on the GPU; an 8k-token prompt is verifiably *not* truncated.
**If the GPU isn't engaged:** your Ollama build is too old for Blackwell (sm_120). Upgrade before anything else.

### Day 5 (Fri 17 Jul) — Harden for a small model

- [ ] Token-budget guard capping chunks passed to agents (multi-hop can pull 12+)
- [ ] Per-node latency logging
- [ ] Timeouts + retries on the LLM client

**Commit:** `feat: add context budget guard and per-node latency logging`
**Done when:** baseline latency recorded for one factual + one multi-hop query. **Write these two numbers down** — they are your "before".

### Day 6 (Sat 18 Jul) — Constrain the router

- [ ] Replace the regex ladder (`router_agent.py:44-58`) with a `json_schema` response format, `enum` of the 3 categories
- [ ] `max_tokens` 512 → ~16
- [ ] Pass `think: false` (Qwen3 is a thinking model)
- [ ] Only fallback left: `try/except → "factual"`

**Commit:** `feat: constrain router output with JSON schema, remove regex parsing`
**Done when:** the regex ladder is deleted and parse failure is structurally impossible.

### Day 7 (Sun 19 Jul) — Constrain the decomposer + BUFFER

- [ ] Schema `{"sub_questions": [string]}` with `maxItems: 4`
- [ ] Delete code-fence stripping and both fallback branches in `decompose.py`

**Commit:** `feat: constrain decomposition output with JSON schema`

---

# Week 2 — Ablation scaffolding + router classifier (Jul 20–26)

### Day 8 (Mon 20 Jul) — Feature flags

- [ ] `router_mode: llm | classifier | off` (`off` → everything to factual agent, generic prompt)
- [ ] `decompose_enabled: bool` (`false` → multihop takes the single-retrieval path)
- [ ] `critic_mode: cosine | nli | llm | off` (`off` → one attempt, no retry loop)
- [ ] All three wired through `graph.py`

**Commit:** `feat: add ablation feature flags for router, decompose, critic`
**Done when:** all three flags flip cleanly and demonstrably change behavior.
**Why now:** one hour today; a lost day if retrofitted after Week 3 starts.

### Day 9 (Tue 21 Jul) — Router training data

- [ ] Kimi drafts ~250 questions over the corpus
- [ ] **Hand-correct every single label**
- [ ] Commit `data/router_questions.jsonl`

**Commit:** `data: add 250 labeled questions for router classifier`

### Day 10 (Wed 22 Jul) — Train the classifier

- [ ] Logistic regression over the MiniLM embeddings already computed in `app/utils/embeddings.py`
- [ ] 80/20 split; pickle to `app/agents/router_model.pkl`; load at startup
- [ ] Wire behind `router_mode=classifier`
- [ ] Report accuracy + per-class F1

**Commit:** `feat: replace LLM router with MiniLM+LogReg classifier (F1=0.XX)`
**Done when:** accuracy and F1 are in the commit message and the README. Expect 90%+, at ~1ms instead of ~1s.

### Day 11 (Thu 23 Jul) — Freeze the eval corpus

- [ ] 3–5 documents committed to `eval/corpus/`
- [ ] `eval/ingest_corpus.py` — deterministic drop-and-reingest

**Commit:** `feat: add frozen eval corpus and deterministic ingest script`
**Done when:** two consecutive ingests produce identical chunk counts. Without this, no two eval runs are comparable.

### Day 12 (Fri 24 Jul) — Golden set, batch 1

- [ ] 20 items in `eval/golden_set.jsonl`
- [ ] Fields: `id`, `question`, `query_type`, `ground_truth`, `expected_pages`
- [ ] Factual-heavy today

**Commit:** `data: golden set batch 1 (20 factual/comparative questions)`

### Day 13 (Sat 25 Jul) — Golden set, batch 2

- [ ] 20 more, weighted toward comparative + multihop

**Commit:** `data: golden set batch 2 (20 comparative/multihop questions)`

### Day 14 (Sun 26 Jul) — Golden set, batch 3 + validator + BUFFER

- [ ] Top up to ~50, roughly balanced across the three types
- [ ] Schema-validation script

**Commit:** `data: complete golden set (50 items) + schema validator`
**Done when:** every `ground_truth` verified against the source document **with your own eyes**.
A golden set with hallucinated ground truths produces confident, beautiful, entirely fictional metrics — and you won't notice for two weeks.

**Do not gold-plate.** 50 solid items beats 120 rushed ones, and beats 80 you're still writing on Day 17.

---

# Week 3 — RAGAS, ablation, analysis (Jul 27 – Aug 2)

### Day 15 (Mon 27 Jul) — Eval harness

- [ ] `eval/run_eval.py --config <name>`
- [ ] Loop golden set, hit `POST /api/query` **over HTTP** (evaluate what you ship, not an imported graph)
- [ ] Capture: answer, contexts (`sources[].content` — free), predicted `query_type`, confidence, attempts, latency
- [ ] Append to `eval/runs/{config}_{timestamp}.jsonl` — never overwrite

**Commit:** `feat: add eval harness for golden-set collection`

### Day 16 (Tue 28 Jul) — RAGAS wired ⚠️ RISK

- [ ] Judge = **Kimi via NIM**. Never the 8B under test — a weak judge produces noise that looks like data.
- [ ] Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `answer_correctness`
- [ ] `RunConfig(max_workers=2-4)` + retries — NIM's free tier *will* rate-limit a 50-question sweep

**Commit:** `feat: score eval runs with RAGAS (judge=kimi via NIM)`
**Done when:** one complete scored run on the `full` config.

### Day 17 (Wed 29 Jul) — Ablation sweep

| config | router | decompose | critic |
|---|---|---|---|
| `baseline` | off | off | off |
| `+router` | llm | off | off |
| `+decompose` | llm | on | off |
| `full` | llm | on | cosine |
| `full-clf` | classifier | on | cosine |

- [ ] Same corpus, same golden set, all five configs
- [ ] Commit the result CSVs

**Commit:** `eval: ablation sweep across 5 configurations`

### Day 18 (Thu 30 Jul) — Analysis

- [ ] Per-`query_type` metric breakdowns (this is where the story lives)
- [ ] Router confusion matrix: LLM vs. classifier, predicted vs. gold
- [ ] **Spearman correlation: `critic.confidence` vs. RAGAS `faithfulness`**

**Commit:** `eval: per-type analysis, router confusion matrix, critic correlation`
**Prediction:** that correlation is near zero. Cosine similarity between an answer and a chunk measures *topical overlap* — a fluent hallucination reusing source vocabulary scores **high**. Finding this is not a setback; it sets up Day 19.

### Day 19 (Fri 31 Jul) — Replace the critic

- [ ] NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) behind `critic_mode=nli`
- [ ] Rerun the sweep
- [ ] Show the correlation improve

**Commit:** `feat: NLI-based critic; faithfulness correlation 0.XX → 0.YY`
**This before/after is the single most interesting result in the project.** Lead the README with it.

### Day 20 (Sat 1 Aug) — Rewrite the README around results

- [ ] Ablation table **first**, architecture diagram second
- [ ] Hardware stated: RTX 5070, 12GB, Q4 8B, local, $0 marginal cost
- [ ] Honest limitations: 50-item golden set, single corpus, judge-model bias

**Commit:** `docs: rewrite README around ablation results`

### Day 21 (Sun 2 Aug) — Polish and tag + BUFFER

- [ ] Reproduction scripts
- [ ] Screenshots
- [ ] Repo hygiene
- [ ] Tag `v1.0`

**Commit:** `chore: repo polish, eval reproduction scripts, v1.0`

---

## The two things that decide whether this works

**Don't gold-plate the golden set.** Days 12–14 are the grind, and they are where three-week plans die. Cap it and move.

**Let the ablation delete things.** If `+router` doesn't beat `baseline` on any metric, the correct outcome is to say so in the README and cut it. A project that reports "this component didn't earn its keep, so I removed it" reads as far more credible than one where every box on the diagram mysteriously helps. That is the difference between an evaluation and a marketing exercise.
