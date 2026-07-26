"""Run the golden set against a running instance and collect raw results for scoring.

This is the *collection* half of evaluation. It hits `POST /api/query` over HTTP — deliberately,
not by importing the graph — so it measures the system as actually shipped: the FastAPI layer, the
compiled graph, retrieval against the real vector store, and the local model under whatever
configuration the running app has. Scoring (RAGAS) happens in a separate step (day 16) over the
file this writes.

The `--config` name is a label only. It does not change the system — the ablation flags live in the
running app's environment (`ROUTER_MODE`, `DECOMPOSE_ENABLED`, `CRITIC_MODE`). The day-17 sweep
restarts the app per configuration and runs this with a matching `--config`, so the output file
records which configuration produced it. There is no endpoint that reports those flags, so nothing
here can verify the label matches the app — set it to match what you booted.

Output: `eval/runs/{config}_{timestamp}.jsonl`, one row per golden item, **append-only and never
overwritten** — a new timestamped file each run, so no run can clobber another's data. Rows are
flushed as they complete, so a crash or Ctrl-C partway through keeps everything up to that point.

Each row carries the gold fields (question, gold query_type, ground_truth, expected_docs) alongside
the prediction (answer, predicted query_type, confidence, attempts, retrieved contexts, latency),
so the scoring step needs no join back to the golden set.

Usage (app must be reachable — `docker compose up -d db app`, model server running):
    python eval/run_eval.py --config full
    python eval/run_eval.py --config baseline --base-url http://localhost:8000 --limit 5
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import httpx

GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"
RUNS_DIR = Path(__file__).parent / "runs"


def _load_golden(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"golden set is empty: {path}")
    return rows


def _query(client: httpx.Client, item: dict, top_k: int) -> dict:
    """POST one question; return the row to record. Errors are captured, not raised, so one
    failed query does not abandon the rest of the run."""
    row = {
        "id": item["id"],
        "question": item["question"],
        "gold_query_type": item["query_type"],
        "ground_truth": item["ground_truth"],
        "expected_docs": item["expected_docs"],
    }
    started = time.perf_counter()
    try:
        resp = client.post("/api/query", json={"question": item["question"], "top_k": top_k})
        resp.raise_for_status()
        data = resp.json()
        row.update(
            answer=data["answer"],
            predicted_query_type=data["query_type"],
            confidence=data["confidence"],
            retrieval_attempts=data["retrieval_attempts"],
            contexts=[s["content"] for s in data["sources"]],
            num_sources=len(data["sources"]),
            error=None,
        )
    except Exception as e:  # noqa: BLE001 — record any failure and move on
        row.update(
            answer=None, predicted_query_type=None, confidence=None,
            retrieval_attempts=None, contexts=[], num_sources=0,
            error=f"{type(e).__name__}: {e}",
        )
    row["latency_s"] = round(time.perf_counter() - started, 3)
    return row


def _summary(rows: list[dict]) -> str:
    ok = [r for r in rows if r["error"] is None]
    errs = len(rows) - len(ok)
    lines = [f"{len(rows)} items, {len(ok)} ok, {errs} error(s)"]
    if ok:
        routed = [r for r in ok if r["predicted_query_type"] == r["gold_query_type"]]
        lats = [r["latency_s"] for r in ok]
        lines.append(f"routing accuracy: {len(routed)}/{len(ok)} = {len(routed) / len(ok):.1%}")
        lines.append(f"latency: mean {mean(lats):.1f}s  median {median(lats):.1f}s  max {max(lats):.1f}s")
        lines.append(f"mean confidence: {mean(r['confidence'] for r in ok):.3f}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="label for this run, e.g. baseline | +router | full | full-clf")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    golden = _load_golden(args.golden)
    if args.limit is not None:
        golden = golden[: args.limit]

    # Fail fast and clearly if the app is not reachable, rather than 50 identical connection errors.
    try:
        httpx.get(f"{args.base_url}/health", timeout=10.0).raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"app not reachable at {args.base_url} ({e}). Is `docker compose up -d db app` running?",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"{args.config}_{stamp}.jsonl"

    print(f"config={args.config}  base_url={args.base_url}  items={len(golden)}  top_k={args.top_k}")
    print(f"writing {out_path}\n")

    rows: list[dict] = []
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client, \
            out_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(golden, 1):
            row = _query(client, item, args.top_k)
            row = {"config": args.config, "run_started_at": stamp, **row}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()  # keep partial results if the run is interrupted
            rows.append(row)
            status = "ok " if row["error"] is None else "ERR"
            pred = row["predicted_query_type"] or "-"
            print(f"  [{i:2d}/{len(golden)}] {row['id']}  {status}  "
                  f"{row['latency_s']:6.1f}s  routed={pred:<11} (gold {row['gold_query_type']})"
                  + (f"  {row['error']}" if row["error"] else ""))

    print("\n" + _summary(rows))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
