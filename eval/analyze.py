"""Day-18 analysis: per-type metric breakdowns, router confusion, critic-vs-faithfulness.

Reads the scored CSVs produced by `run_ragas.py` (and, for routing, the raw runs, which need no
judge). Three questions:

1. **Per query type** — where does the pipeline actually win or lose? Aggregates hide this: a
   system can look fine overall while failing every multi-hop question.

2. **Router confusion, LLM vs classifier** — predicted against gold, per configuration.

3. **Does the critic's confidence mean anything?** Spearman between `critic.confidence` and RAGAS
   `faithfulness`. `score_answer` is the max cosine similarity between the answer embedding and
   the chunk embeddings, so it measures *topical overlap*, not entailment — a fluent hallucination
   that reuses source vocabulary scores high. The prediction on record is that this correlation is
   near zero. Confirming it is the argument for the NLI critic.

   Only configurations with `critic_mode=cosine` can answer this: with the critic off, confidence
   is a constant 1.0 and the correlation is undefined. The script says so rather than printing a
   meaningless number.

Coverage is enforced, not assumed. A metric scored on a minority of rows is reported as such and
excluded from per-type breakdowns, because judge failures were observed to concentrate in one
query type — which makes a per-type mean over survivors a selection artefact, not a measurement.

Usage:
    python eval/analyze.py                      # every scored CSV in eval/runs/
    python eval/analyze.py --out-dir eval/runs  # also write analysis CSVs
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
TYPES = ["factual", "comparative", "multihop"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
           "answer_correctness"]
CONFIG_ORDER = ["baseline", "+router", "+decompose", "full", "full-clf", "full-nli"]

# Below this share of rows a metric is reported but not broken down per type: the judge failures
# behind the gap were measured to cluster by query type, so the surviving rows are not a random
# sample of the run.
MIN_COVERAGE_FOR_BREAKDOWN = 0.8


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float | None:
    """Parse a CSV cell to float, treating blanks and NaN as missing."""
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rho via Pearson on ranks, with average ranks for ties.

    Written out rather than pulled from scipy: the only dependency this analysis would otherwise
    need, for one formula over at most a few hundred points.
    """
    n = len(xs)
    if n < 3:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None  # a constant series has no rank order — e.g. confidence with the critic off
    return num / (dx * dy)


def _permutation_p(xs: list[float], ys: list[float], observed: float, iters: int = 20000) -> float:
    """Two-sided p for a Spearman rho, by shuffling one series.

    A permutation test rather than a t-approximation: it needs no special functions, makes no
    distributional assumption, and is easy to check by eye. Seeded, so the reported p is stable
    across runs.

    This matters more than the rho itself here. At n=47 a correlation of ~0.2 sits near the noise
    floor, and quoting it bare invites reading "weak positive relationship" into what may be no
    relationship at all.
    """
    import random

    rng = random.Random(0)
    shuffled = list(ys)
    hits = 0
    for _ in range(iters):
        rng.shuffle(shuffled)
        r = _spearman(xs, shuffled)
        if r is not None and abs(r) >= abs(observed):
            hits += 1
    return (hits + 1) / (iters + 1)  # add-one keeps p strictly positive


def _latest_scored(runs_dir: Path) -> dict[str, Path]:
    """Newest *_scored_*.csv per config, keyed by the config recorded inside the file."""
    latest: dict[str, tuple[float, Path]] = {}
    for path in runs_dir.glob("*_scored*.csv"):
        rows = _read_csv(path)
        if not rows:
            continue
        config = rows[0].get("config")
        if not config:
            continue
        mtime = path.stat().st_mtime
        if config not in latest or mtime > latest[config][0]:
            latest[config] = (mtime, path)
    return {c: p for c, (_, p) in latest.items()}


def _coverage(rows: list[dict], metric: str) -> tuple[int, int]:
    present = sum(1 for r in rows if _f(r, metric) is not None)
    return present, len(rows)


def analyze_per_type(scored: dict[str, list[dict]]) -> list[dict]:
    """Metric means split by gold query type, for metrics with adequate coverage."""
    out = []
    for config in [c for c in CONFIG_ORDER if c in scored] + sorted(set(scored) - set(CONFIG_ORDER)):
        rows = scored[config]
        print(f"\n=== {config} ===")
        for metric in METRICS:
            present, total = _coverage(rows, metric)
            if present == 0:
                continue
            share = present / total
            overall = [_f(r, metric) for r in rows if _f(r, metric) is not None]
            line = f"  {metric:<20} overall {sum(overall) / len(overall):.3f}  ({present}/{total})"
            if share < MIN_COVERAGE_FOR_BREAKDOWN:
                print(line + "  -- coverage too low for a per-type split")
                out.append({"config": config, "metric": metric, "query_type": "ALL",
                            "mean": round(sum(overall) / len(overall), 4),
                            "n": present, "coverage": round(share, 3), "trustworthy": False})
                continue
            parts = []
            for t in TYPES:
                vals = [_f(r, metric) for r in rows
                        if r.get("gold_query_type") == t and _f(r, metric) is not None]
                if vals:
                    parts.append(f"{t[:4]}={sum(vals) / len(vals):.3f}")
                    out.append({"config": config, "metric": metric, "query_type": t,
                                "mean": round(sum(vals) / len(vals), 4), "n": len(vals),
                                "coverage": round(share, 3), "trustworthy": True})
            print(line + "   " + "  ".join(parts))
            out.append({"config": config, "metric": metric, "query_type": "ALL",
                        "mean": round(sum(overall) / len(overall), 4),
                        "n": present, "coverage": round(share, 3), "trustworthy": True})
    return out


def analyze_confusion(runs_dir: Path) -> list[dict]:
    """Router confusion per config, from the raw runs — no judge needed."""
    out = []
    per_config: dict[str, Counter] = {}
    for path in runs_dir.glob("*.jsonl"):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        config = rows[0].get("config")
        if not config:
            continue
        cm = Counter((r["gold_query_type"], r["predicted_query_type"])
                     for r in rows if not r.get("error"))
        per_config[config] = cm

    print("\n\n########## ROUTER CONFUSION ##########")
    for config in [c for c in CONFIG_ORDER if c in per_config] + sorted(set(per_config) - set(CONFIG_ORDER)):
        cm = per_config[config]
        total = sum(cm.values())
        correct = sum(cm[(t, t)] for t in TYPES)
        print(f"\n{config}  ({correct}/{total} = {correct / total:.0%})   rows=gold, cols=predicted")
        print(" " * 14 + "".join(f"{t:>13}" for t in TYPES))
        for g in TYPES:
            row_total = sum(cm[(g, p)] for p in TYPES)
            cells = "".join(f"{cm[(g, p)]:>13}" for p in TYPES)
            print(f"  {g:<12}{cells}   {cm[(g, g)]}/{row_total}")
            for p in TYPES:
                out.append({"config": config, "gold": g, "predicted": p, "count": cm[(g, p)]})
    return out


def analyze_critic(scored: dict[str, list[dict]]) -> list[dict]:
    """Spearman between the critic's confidence and RAGAS faithfulness."""
    print("\n\n########## CRITIC CONFIDENCE vs FAITHFULNESS ##########")
    print("\nDoes the cosine critic's confidence predict whether an answer is grounded?")
    out = []
    for config in [c for c in CONFIG_ORDER if c in scored] + sorted(set(scored) - set(CONFIG_ORDER)):
        rows = scored[config]
        pairs = [(_f(r, "confidence"), _f(r, "faithfulness")) for r in rows]
        pairs = [(c, f) for c, f in pairs if c is not None and f is not None]
        if len(pairs) < 3:
            print(f"\n  {config:<12} too few scored rows ({len(pairs)})")
            continue
        xs = [c for c, _ in pairs]
        ys = [f for _, f in pairs]
        rho = _spearman(xs, ys)
        if rho is None:
            # critic_mode=off reports a constant confidence of 1.0, so there is no rank order.
            print(f"\n  {config:<12} n={len(pairs):<3} confidence is constant "
                  f"({xs[0]:.3f}) — critic disabled, correlation undefined")
            out.append({"config": config, "n": len(pairs), "spearman": None,
                        "p_value": None, "significant": None,
                        "note": "confidence constant (critic off)"})
            continue
        p = _permutation_p(xs, ys, rho)
        verdict = "significant at p<0.05" if p < 0.05 else "NOT distinguishable from zero"
        print(f"\n  {config:<12} n={len(pairs):<3} spearman={rho:+.3f}  p={p:.3f}  -> {verdict}"
              f"\n  {'':<12} mean_conf={sum(xs) / len(xs):.3f}  mean_faith={sum(ys) / len(ys):.3f}")
        out.append({"config": config, "n": len(pairs), "spearman": round(rho, 4),
                    "p_value": round(p, 4), "significant": p < 0.05, "note": ""})
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="write analysis CSVs here")
    args = parser.parse_args(argv)

    scored_paths = _latest_scored(args.runs_dir)
    scored = {c: _read_csv(p) for c, p in scored_paths.items()}

    if scored:
        print("########## PER-QUERY-TYPE BREAKDOWN ##########")
        print("(fact=factual, comp=comparative, mult=multihop)")
        per_type = analyze_per_type(scored)
    else:
        print("no scored CSVs found — run eval/run_ragas.py first "
              "(routing analysis below needs no judge)", file=sys.stderr)
        per_type = []

    confusion = analyze_confusion(args.runs_dir)
    critic = analyze_critic(scored) if scored else []

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows, fields in (
            ("analysis_per_type.csv", per_type,
             ["config", "metric", "query_type", "mean", "n", "coverage", "trustworthy"]),
            ("analysis_confusion.csv", confusion, ["config", "gold", "predicted", "count"]),
            ("analysis_critic_correlation.csv", critic,
             ["config", "n", "spearman", "p_value", "significant", "note"]),
        ):
            if not rows:
                continue
            with (args.out_dir / name).open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {args.out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
