"""Embedding helper — uses local sentence-transformers (all-MiniLM-L6-v2)."""

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning 384-dim vectors."""
    if not texts:
        return []
    model = _get_model()
    # Run in thread pool to avoid blocking the async event loop
    embeddings = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
    return embeddings.tolist()


async def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    results = await embed_texts([text])
    return results[0]
