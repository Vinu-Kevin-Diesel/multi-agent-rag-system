"""Stage 3: Semantic chunking — split regions into embedding-ready chunks."""

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.ingestion.layout_detection import DocumentRegion

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


@dataclass
class Chunk:
    content: str
    chunk_index: int
    page_number: int | None = None
    element_type: str = "Text"
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split on period/newline boundaries."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [s.strip() for s in sentences if s.strip()]


async def semantic_chunk(regions: list[DocumentRegion]) -> list[Chunk]:
    """Group sentences by semantic similarity, respecting chunk_size limits."""
    encoder = _get_encoder()
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    all_sentences: list[dict] = []
    for region in regions:
        for sentence in _split_sentences(region.content):
            all_sentences.append({
                "text": sentence,
                "page_number": region.page_number,
                "element_type": region.element_type,
            })

    if not all_sentences:
        return []

    texts = [s["text"] for s in all_sentences]
    embeddings = encoder.encode(texts, normalize_embeddings=True)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    current_page = all_sentences[0]["page_number"]
    current_type = all_sentences[0]["element_type"]

    for i, sent_info in enumerate(all_sentences):
        token_est = len(sent_info["text"].split())

        should_break = (buffer_len + token_est > chunk_size) or (
            i > 0
            and float(np.dot(embeddings[i], embeddings[i - 1])) < 0.5
            and buffer_len > chunk_size // 3
        )

        if should_break and buffer:
            chunks.append(
                Chunk(
                    content=" ".join(buffer),
                    chunk_index=len(chunks),
                    page_number=current_page,
                    element_type=current_type,
                )
            )
            # keep overlap
            overlap_tokens = 0
            overlap_start = len(buffer)
            for j in range(len(buffer) - 1, -1, -1):
                overlap_tokens += len(buffer[j].split())
                if overlap_tokens >= overlap:
                    overlap_start = j
                    break
            buffer = buffer[overlap_start:]
            buffer_len = sum(len(s.split()) for s in buffer)

        buffer.append(sent_info["text"])
        buffer_len += token_est
        current_page = sent_info["page_number"]
        current_type = sent_info["element_type"]

    if buffer:
        chunks.append(
            Chunk(
                content=" ".join(buffer),
                chunk_index=len(chunks),
                page_number=current_page,
                element_type=current_type,
            )
        )

    return chunks
