"""Validate the golden question set against its schema and the frozen corpus.

A golden set with a malformed row, a duplicate id, or an `expected_docs` entry that names a
document not in the corpus produces confident, precise, entirely wrong metrics — and nothing
downstream flags it. This script is the cheap gate that catches those before an eval run does.

It is structural validation only. It cannot check that a `ground_truth` is *correct* — that is
done by eye against the source document, and is the one step no script replaces.

Usage (no services required — reads files only):
    python eval/validate_golden_set.py
    python eval/validate_golden_set.py --path eval/golden_set.jsonl

Exit code is 0 when every row is valid and non-zero on the first structural violation, so it can
gate CI or a run script.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"
CORPUS_DIR = Path(__file__).parent / "corpus"

REQUIRED_FIELDS = {"id", "question", "query_type", "ground_truth", "expected_docs"}
QUERY_TYPES = {"factual", "comparative", "multihop"}


def _validate(rows: list[dict], corpus_files: set[str]) -> list[str]:
    """Return a list of human-readable errors; empty means the set is valid."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    for i, row in enumerate(rows):
        # A stable label for the row even when `id` is the thing that is broken.
        where = row.get("id") or f"line {i + 1}"

        extra = set(row) - REQUIRED_FIELDS
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{where}: missing field(s) {sorted(missing)}")
        if extra:
            errors.append(f"{where}: unexpected field(s) {sorted(extra)}")

        rid = row.get("id")
        if isinstance(rid, str):
            if rid in seen_ids:
                errors.append(f"{where}: duplicate id")
            seen_ids.add(rid)

        qtype = row.get("query_type")
        if qtype not in QUERY_TYPES:
            errors.append(f"{where}: query_type {qtype!r} not in {sorted(QUERY_TYPES)}")

        for field in ("question", "ground_truth"):
            val = row.get(field)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{where}: {field} is empty or not a string")

        q = row.get("question")
        if isinstance(q, str) and q.strip():
            key = q.strip().lower()
            if key in seen_questions:
                errors.append(f"{where}: duplicate question")
            seen_questions.add(key)

        docs = row.get("expected_docs")
        if not isinstance(docs, list) or not docs:
            errors.append(f"{where}: expected_docs must be a non-empty list")
        else:
            for d in docs:
                if d not in corpus_files:
                    errors.append(f"{where}: expected_docs entry {d!r} is not in the corpus")
            # A multi-hop question whose answer lives in a single document is mislabelled: it
            # would make the decompose ablation look better than it is. This is load-bearing,
            # so it is an error, not a warning.
            if qtype == "multihop" and len(set(docs)) < 2:
                errors.append(f"{where}: multihop question must span >= 2 documents")

    # ids should be a contiguous gs-001..gs-NNN run, so a dropped or misnumbered row is caught.
    string_ids = [r["id"] for r in rows if isinstance(r.get("id"), str)]
    expected = [f"gs-{n:03d}" for n in range(1, len(rows) + 1)]
    if string_ids != expected:
        first = next((s for s, e in zip(string_ids + [None], expected) if s != e), None)
        errors.append(f"ids are not the contiguous run gs-001..gs-{len(rows):03d} "
                      f"(first mismatch near {first!r})")

    return errors


def main(path: Path) -> int:
    if not path.exists():
        print(f"golden set not found: {path}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"line {i + 1}: invalid JSON — {e}", file=sys.stderr)
            return 1

    corpus_files = {p.name for p in CORPUS_DIR.iterdir() if p.is_file()}
    errors = _validate(rows, corpus_files)

    if errors:
        print(f"INVALID: {len(errors)} problem(s) in {path.name}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    counts = Counter(r["query_type"] for r in rows)
    mix = ", ".join(f"{counts[t]} {t}" for t in ("factual", "comparative", "multihop"))
    print(f"OK: {len(rows)} items valid ({mix})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=GOLDEN_PATH,
                        help="path to the golden set jsonl (default: eval/golden_set.jsonl)")
    args = parser.parse_args()
    raise SystemExit(main(args.path))
