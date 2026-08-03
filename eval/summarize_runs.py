"""Compare collected runs across ablation configurations, without the judge.

Everything here is computed from the gold labels already in the golden set and the predictions
already in a run: routing accuracy, the router confusion matrix, latency, critic attempts and
confidence. None of it needs RAGAS or a hosted judge, so it stays available when the judge's free
tier is exhausted — which is exactly when you still want to know whether a sweep worked.

It answers "does the router earn its keep?" and "what does the critic cost?" on its own. It does
NOT answer "are the answers any good" — that needs `run_ragas.py`.

Usage:
    python eval/summarize_runs.py                       # every config, latest run of each
    python eval/summarize_runs.py --out summary.csv     # also write a CSV

Runs are matched by the `config` field recorded inside each row, not by filename, and the flags
each run was collected under are printed so a mislabelled run is visible rather than silent.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
TYPES = ["factual", "comparative", "multihop"]

# Ablation order, so the table reads as a progression rather than alphabetically.
CONFIG_ORDER = ["baseline", "+router", "+decompose", "full", "full-clf", "full-nli"]


def _latest_per_config(runs_dir: Path) -> dict[str, Path]:
    """Newest run file per config, keyed by the config recorded in the rows."""
    latest: dict[str, tuple[float, Path]] = {}
    for path in runs_dir.glob("*.jsonl"):
        try:
            first = next(l for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
            config = json.loads(first).get("config")
        except (StopIteration, json.JSONDecodeError):
            continue
        if not config:
            continue
        mtime = path.stat().st_mtime
        if config not in latest or mtime > latest[config][0]:
            latest[config] = (mtime, path)
    return {c: p for c, (_, p) in latest.items()}


def _summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    cm = Counter((r["gold_query_type"], r["predicted_query_type"]) for r in ok)
    correct = sum(cm[(t, t)] for t in TYPES)
    lat = sorted(r["latency_s"] for r in ok)
    attempts = Counter(r["retrieval_attempts"] for r in ok)

    def _pct(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    out = {
        "items": len(rows),
        "errors": len(rows) - len(ok),
        "routing_accuracy": _pct(correct, len(ok)),
        "latency_mean_s": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "latency_median_s": round(lat[len(lat) // 2], 1) if lat else 0.0,
        "latency_max_s": round(lat[-1], 1) if lat else 0.0,
        "mean_confidence": round(sum(r["confidence"] for r in ok) / len(ok), 3) if ok else 0.0,
        # Mean attempts, not "share at the maximum": with the critic off every query takes
        # exactly one attempt, so a share-at-max would read 100% and look alarming when it is
        # simply the absence of a retry loop. The mean is comparable across configurations.
        "mean_attempts": round(sum(r["retrieval_attempts"] for r in ok) / len(ok), 2) if ok else 0.0,
        "pct_exhausted": _pct(sum(1 for r in ok if r["retrieval_attempts"] >= 3), len(ok)),
    }
    for t in TYPES:
        total = sum(cm[(t, p)] for p in TYPES)
        out[f"recall_{t}"] = _pct(cm[(t, t)], total)
    return out, cm


def _print_confusion(name: str, cm: Counter) -> None:
    print(f"\n  {name} — rows = gold, columns = predicted")
    header = " " * 14 + "".join(f"{t:>13}" for t in TYPES) + "      recall"
    print(header)
    for g in TYPES:
        total = sum(cm[(g, p)] for p in TYPES)
        cells = "".join(f"{cm[(g, p)]:>13}" for p in TYPES)
        rec = f"{cm[(g, g)]}/{total}" if total else "-"
        print(f"  {g:<12}{cells}   {rec:>10}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="write the comparison as CSV")
    parser.add_argument("--confusion", action="store_true", default=True,
                        help="print a per-config router confusion matrix")
    args = parser.parse_args(argv)

    found = _latest_per_config(args.runs_dir)
    if not found:
        print(f"no runs found in {args.runs_dir}", file=sys.stderr)
        return 1

    ordered = [c for c in CONFIG_ORDER if c in found] + sorted(set(found) - set(CONFIG_ORDER))

    summaries: dict[str, dict] = {}
    print(f"{'config':<12} {'items':>5} {'err':>4} {'routing':>8} "
          f"{'factual':>8} {'comp':>8} {'multihop':>9} "
          f"{'lat_mean':>9} {'lat_med':>8} {'conf':>6} {'att':>5} {'exh':>5}")
    for config in ordered:
        rows = [json.loads(l) for l in found[config].read_text(encoding="utf-8").splitlines() if l.strip()]
        s, cm = _summarize(rows)
        summaries[config] = (s, cm, rows[0].get("flags"))
        print(f"{config:<12} {s['items']:>5} {s['errors']:>4} {s['routing_accuracy']:>7.0%} "
              f"{s['recall_factual']:>7.0%} {s['recall_comparative']:>7.0%} {s['recall_multihop']:>8.0%} "
              f"{s['latency_mean_s']:>8.1f}s {s['latency_median_s']:>7.1f}s "
              f"{s['mean_confidence']:>6.3f} {s['mean_attempts']:>5.2f} {s['pct_exhausted']:>4.0%}")

    print("\nflags each run was actually collected under:")
    for config in ordered:
        flags = summaries[config][2]
        print(f"  {config:<12} {flags if flags else '(not recorded — pre-dates the /health flags check)'}")

    if args.confusion:
        for config in ordered:
            _print_confusion(config, summaries[config][1])

    if args.out:
        fields = ["config", "items", "errors", "routing_accuracy", "recall_factual",
                  "recall_comparative", "recall_multihop", "latency_mean_s", "latency_median_s",
                  "latency_max_s", "mean_confidence", "mean_attempts", "pct_exhausted",
                  "router_mode", "decompose_enabled", "critic_mode"]
        with args.out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for config in ordered:
                s, _, flags = summaries[config]
                flags = flags or {}
                w.writerow({"config": config, **s,
                            "router_mode": flags.get("router_mode"),
                            "decompose_enabled": flags.get("decompose_enabled"),
                            "critic_mode": flags.get("critic_mode")})
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
