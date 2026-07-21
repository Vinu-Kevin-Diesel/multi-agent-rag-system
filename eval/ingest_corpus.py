"""Drop the database and re-ingest the frozen eval corpus, deterministically.

Every eval run must start from an identical vector store, or `context_recall` and
`context_precision` are not comparable across configurations — you would be measuring corpus
drift alongside the thing you meant to measure.

Determinism is verified by **content hash**, not just chunk count: identical counts with
different chunk text would still silently break comparability.

Document IDs are uuid4 and therefore differ between runs, so they are deliberately excluded
from the hash. The eval harness queries across all documents, so IDs do not matter; chunk text
and ordering do.

Usage (from the repo root, with the db service up):
    docker compose run --rm app python eval/ingest_corpus.py            # ingest + write manifest
    docker compose run --rm app python eval/ingest_corpus.py --verify   # ingest + fail on drift
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import delete, select

from app.database import async_session
from app.ingestion import IngestionPipeline
from app.models import Document, DocumentChunk

CORPUS_DIR = Path(__file__).parent / "corpus"
MANIFEST_PATH = Path(__file__).parent / "corpus_manifest.json"


async def _reset(session) -> None:
    """Delete every document; chunks cascade."""
    await session.execute(delete(Document))
    await session.commit()


async def _ingest_all(session) -> list[dict]:
    """Ingest each corpus file in sorted order and return a per-document manifest entry."""
    entries = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx", ".html"}:
            continue

        doc = await IngestionPipeline(session).run(
            file_path=path, filename=path.name, content_type="text/markdown"
        )

        rows = (
            await session.execute(
                select(DocumentChunk.content)
                .where(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index)
            )
        ).scalars().all()

        digest = hashlib.sha256("\n\n".join(rows).encode("utf-8")).hexdigest()
        entries.append({"filename": path.name, "chunks": len(rows), "sha256": digest})
        print(f"  {path.name:28} {len(rows):3d} chunks  {digest[:12]}")
    return entries


def _manifest(entries: list[dict]) -> dict:
    total = sum(e["chunks"] for e in entries)
    combined = hashlib.sha256(
        "".join(e["sha256"] for e in entries).encode("utf-8")
    ).hexdigest()
    return {"documents": entries, "total_chunks": total, "corpus_sha256": combined}


async def main(verify: bool) -> int:
    async with async_session() as session:
        print("resetting database...")
        await _reset(session)
        print(f"ingesting {CORPUS_DIR}...")
        entries = await _ingest_all(session)

    manifest = _manifest(entries)
    print(f"\ntotal: {manifest['total_chunks']} chunks across {len(entries)} documents")
    print(f"corpus_sha256: {manifest['corpus_sha256']}")

    if verify:
        if not MANIFEST_PATH.exists():
            print("\nno manifest to verify against — run without --verify first", file=sys.stderr)
            return 1
        expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if expected == manifest:
            print("\nOK: corpus matches the committed manifest")
            return 0
        print("\nDRIFT: corpus does not match the committed manifest", file=sys.stderr)
        print(f"  expected total {expected['total_chunks']}, got {manifest['total_chunks']}",
              file=sys.stderr)
        print(f"  expected sha {expected['corpus_sha256'][:16]}, "
              f"got {manifest['corpus_sha256'][:16]}", file=sys.stderr)
        return 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="compare against the committed manifest and exit non-zero on drift")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.verify)))
