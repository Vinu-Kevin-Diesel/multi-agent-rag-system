"""Score a collected eval run with RAGAS.

The second half of evaluation: `run_eval.py` collects raw answers over HTTP, this scores them.
Split deliberately — collection is slow (a local 8B, ~50s/item) and scoring is a separate cost
against a hosted judge, so a scoring bug should never force you to re-run the model.

**The judge is never the model under test.** It is configured independently (`JUDGE_*` in .env,
resolved the same way `app/dependencies.py` resolves it) and runs on a strong hosted model.
Scoring a local 8B's answers with that same 8B measures self-consistency, not correctness — the
resulting numbers would look fine and mean nothing.

Metrics (all five need the judge; `answer_relevancy` and `answer_correctness` also need embeddings):

    faithfulness        is the answer supported by the retrieved contexts (hallucination check)
    answer_relevancy    does the answer actually address the question
    context_precision   are the retrieved chunks relevant (retrieval quality, signal vs noise)
    context_recall      did retrieval find what the ground truth needs (retrieval coverage)
    answer_correctness  does the answer agree with the ground truth

Embeddings run **locally** on the same MiniLM the app uses, not through the judge API. Two
reasons: the embedding calls are numerous and would dominate the rate limit budget, and using the
app's own embedder keeps the scoring free and offline-reproducible.

Usage (inside the app container, which has ragas installed):
    docker compose run --rm --no-deps app python eval/run_ragas.py --run eval/runs/full_2026....jsonl
    docker compose run --rm --no-deps app python eval/run_ragas.py --latest --limit 3

Writes `{run_stem}_scored.csv` (per-item) next to the input and prints the aggregate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

# NIM's free tier returns 503 "ResourceExhausted: Worker local total request limit reached (N/48)"
# — a *shared* concurrency ceiling, so it can fire even when our own concurrency is low. Retries
# are therefore mandatory, but they must be bounded at ONE layer only.
#
# Learned the hard way: retries configured on both the SDK client and RunConfig multiply, and a
# 3-item scoring job sat at 0% CPU for 15 minutes backing off. The SDK owns retries (it honours
# Retry-After and handles 429/503 properly); RunConfig keeps a small outer retry for the rest.
DEFAULT_WORKERS = 3          # our own concurrency; the 503 ceiling is shared, so raising this rarely helps
DEFAULT_TIMEOUT = 300        # a RAGAS prompt against a slow judge measured ~14s/call; leave headroom
SDK_MAX_RETRIES = 5          # exponential backoff inside the OpenAI SDK
OUTER_MAX_RETRIES = 3        # RunConfig's retry around the whole metric call
MAX_WAIT = 60                # cap a single backoff so a stall stays diagnosable


def _utc_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_run(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (scorable rows, skipped rows). A failed or context-less query cannot be scored."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scorable, skipped = [], []
    for r in rows:
        if r.get("error") or not r.get("answer") or not r.get("contexts"):
            skipped.append(r)
        else:
            scorable.append(r)
    return scorable, skipped


def _judge_settings() -> tuple[str, str, str]:
    """Resolve the judge exactly as app/dependencies.py does: JUDGE_* first, NVIDIA_API_KEY as the
    legacy fallback. Read from the environment rather than importing app.config, so scoring does
    not drag in the application stack."""
    model = os.getenv("JUDGE_MODEL", "deepseek-ai/deepseek-v4-flash")
    base_url = os.getenv("JUDGE_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("no judge API key: set JUDGE_API_KEY or NVIDIA_API_KEY in .env")
    return model, base_url, api_key


def _local_embeddings(model_name: str):
    """Wrap the app's sentence-transformers model in the minimal langchain Embeddings interface
    RAGAS accepts. Avoids adding langchain-huggingface, and reuses the model already in the image."""
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class _MiniLM(Embeddings):
        def __init__(self, name: str):
            self._model = SentenceTransformer(name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return _MiniLM(model_name)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="path to a run jsonl from eval/runs/")
    src.add_argument("--latest", action="store_true", help="score the most recent run in eval/runs/")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N items")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"judge concurrency (default {DEFAULT_WORKERS}; raise only off the free tier)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"per-job timeout in seconds (default {DEFAULT_TIMEOUT}). Raise it for "
                             "long answers: faithfulness judges one call per extracted statement, "
                             "so a comparative answer costs several times a factual one.")
    parser.add_argument("--types", default=None,
                        help="comma-separated gold_query_type filter, e.g. 'comparative' — for "
                             "re-scoring the slice a previous run timed out on")
    parser.add_argument("--metrics", default=None,
                        help="comma-separated subset of: faithfulness, answer_relevancy, "
                             "context_precision, context_recall, answer_correctness. Defaults to "
                             "all five. The day-17 sweep drops answer_correctness — see README.")
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default: <run>_scored_<timestamp>.csv, never overwritten)")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting an existing output file")
    args = parser.parse_args(argv)

    if args.latest:
        candidates = sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"no runs found in {RUNS_DIR} — collect one with run_eval.py first", file=sys.stderr)
            return 1
        run_path = candidates[0]
    else:
        run_path = args.run
    if not run_path.exists():
        print(f"run not found: {run_path}", file=sys.stderr)
        return 1

    scorable, skipped = _load_run(run_path)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        scorable = [r for r in scorable if r.get("gold_query_type") in wanted]
    if args.limit is not None:
        scorable = scorable[: args.limit]
    if not scorable:
        print(f"nothing scorable in {run_path.name}", file=sys.stderr)
        return 1

    # Imported here so --help and the file checks above stay fast and dependency-free.
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    available = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    if args.metrics:
        names = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = [m for m in names if m not in available]
        if unknown:
            print(f"unknown metric(s): {unknown}. Choose from {list(available)}", file=sys.stderr)
            return 1
    else:
        names = list(available)
    chosen = [available[m] for m in names]

    judge_model, judge_base_url, judge_api_key = _judge_settings()
    print(f"run:    {run_path.name}  ({len(scorable)} scorable, {len(skipped)} skipped)")
    print(f"metrics: {', '.join(names)}")
    print(f"judge:  {judge_model} @ {judge_base_url}")
    print(f"config: workers={args.workers} timeout={args.timeout:.0f}s "
          f"retries={SDK_MAX_RETRIES}(sdk)+{OUTER_MAX_RETRIES}(outer) max_wait={MAX_WAIT}s\n")

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model, base_url=judge_base_url, api_key=judge_api_key,
        temperature=0, timeout=args.timeout, max_retries=SDK_MAX_RETRIES,
    ))
    embeddings = LangchainEmbeddingsWrapper(
        _local_embeddings(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    )

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=list(r["contexts"]),
            reference=r["ground_truth"],
        )
        for r in scorable
    ])

    result = evaluate(
        dataset=dataset,
        metrics=chosen,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=args.workers, timeout=args.timeout,
                             max_retries=OUTER_MAX_RETRIES, max_wait=MAX_WAIT),
        show_progress=True,
    )

    df = result.to_pandas()
    # Carry the fields the day-18 analysis needs (per-type breakdown, router confusion matrix,
    # critic-vs-faithfulness correlation) into the same file, so it needs no join back.
    for col, key in (("id", "id"), ("config", "config"), ("gold_query_type", "gold_query_type"),
                     ("predicted_query_type", "predicted_query_type"), ("confidence", "confidence"),
                     ("retrieval_attempts", "retrieval_attempts"), ("latency_s", "latency_s")):
        df[col] = [r.get(key) for r in scorable]

    # Never overwrite a previous scoring, for the same reason run_eval.py never overwrites a run:
    # a scored file is expensive and irreplaceable. Learned by destroying one — a re-score that
    # ran while the judge quota was exhausted wrote 4/50 coverage over an existing 32/50 result,
    # and the only recovery was to score the raw run again from scratch.
    if args.out:
        out_path = args.out
    else:
        stamp = _utc_stamp()
        out_path = run_path.with_name(f"{run_path.stem}_scored_{stamp}.csv")
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path} — pass --force or --out", file=sys.stderr)
        return 1
    df.to_csv(out_path, index=False)

    metric_cols = [c for c in ("faithfulness", "answer_relevancy", "context_precision",
                               "context_recall", "answer_correctness") if c in df.columns]
    # A metric whose judge call ultimately failed comes back NaN. .mean() skips those silently,
    # so report the denominator: a 0.9 over 12 of 50 items is not the same claim as 0.9 over 50.
    print("\naggregate:")
    for c in metric_cols:
        scored = df[c].notna().sum()
        note = "" if scored == len(df) else f"   ({scored}/{len(df)} scored, rest NaN)"
        print(f"  {c:20} {df[c].mean():.3f}{note}")
    # Low coverage across every metric at once is not a property of the answers — it means the
    # judge stopped answering (free-tier quota, or a sustained 429/503). Say so, because the
    # aggregate above is otherwise a plausible-looking number computed from almost nothing.
    if metric_cols:
        worst = min(df[c].notna().sum() for c in metric_cols)
        if worst < 0.5 * len(df):
            print(f"\nWARNING: coverage as low as {worst}/{len(df)} — this usually means the judge "
                  f"was rate-limited or out of quota, not that the answers scored badly.\n"
                  f"         Do not report these numbers; re-score once quota resets.")
    if skipped:
        print(f"\nskipped {len(skipped)} unscorable row(s): {[r['id'] for r in skipped]}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
