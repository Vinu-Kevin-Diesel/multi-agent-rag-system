"""Critic agent — scores answer quality against source chunks via cosine similarity."""

import numpy as np

from app.config import settings
from app.agents.utils import extract_content
from app.utils.embeddings import embed_texts


async def score_answer(answer: str, source_chunks: list[dict]) -> float:
    """Compute cosine similarity between the answer and source chunk embeddings.

    Returns the maximum similarity score across all source chunks.
    This measures how well-grounded the answer is in the retrieved sources.
    """
    texts = [answer] + [c["content"] for c in source_chunks]
    embeddings = await embed_texts(texts)

    answer_emb = np.array(embeddings[0])
    chunk_embs = np.array(embeddings[1:])

    norms_answer = np.linalg.norm(answer_emb)
    norms_chunks = np.linalg.norm(chunk_embs, axis=1)

    if norms_answer == 0:
        return 0.0

    similarities = chunk_embs @ answer_emb / (norms_chunks * norms_answer + 1e-10)
    return float(np.max(similarities))


async def generate_refined_query(
    original_query: str,
    answer: str,
    client,
) -> str:
    """Ask LLM to produce a more targeted query when the critic rejects an answer."""
    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "Generate a refined search query to find better source material. Return ONLY the refined query."},
            {"role": "user", "content": (
                f"Original question: {original_query}\n"
                f"Previous answer (low confidence): {answer}\n\n"
                "Generate a more specific search query to retrieve better source chunks."
            )},
        ],
    )
    return extract_content(response).strip()
