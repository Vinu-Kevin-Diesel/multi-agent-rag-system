"""Critic agent — scores how well an answer is grounded in its source chunks.

Two scorers, selected by `critic_mode`:

- **cosine** — max cosine similarity between the answer embedding and the chunk embeddings.
  Cheap (~0.05s, local embeddings) but it measures *topical overlap*, not support. Measured
  against RAGAS faithfulness over 47 scored answers it gives Spearman +0.207 at p=0.163 — no
  detectable relationship. A fluent answer that reuses source vocabulary scores high whether or
  not the source entails it.

- **nli** — entailment probability from a cross-encoder, per answer sentence against each chunk.
  This asks the question the cosine score only appears to ask: *does the source support this
  claim?* On the probe case that motivated it — "requires documented clinical benefit and remains
  valid for 24 months" against a source saying 12 months — cosine scores high on vocabulary
  overlap while NLI returns 0.001 entailment (0.987 contradiction).

The two scores are on different scales and are NOT interchangeable: `critic_similarity_threshold`
applies to cosine, `critic_nli_threshold` to NLI. Reusing one threshold for both would silently
change the retry rate.
"""

import asyncio
from functools import lru_cache

import numpy as np

from app.config import settings
from app.agents.utils import extract_content
from app.ingestion.chunking import _split_sentences
from app.utils.embeddings import embed_texts


@lru_cache(maxsize=1)
def _get_nli_model():
    """Load the NLI cross-encoder once. Downloads on first use, like the embedding model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.critic_nli_model)


@lru_cache(maxsize=1)
def _entailment_index() -> int:
    """Which output column is P(entailment), read from the model's own label map.

    Not hardcoded: label order varies between NLI checkpoints, and getting it wrong inverts the
    metric silently — every grounded answer would score near zero and the retry loop would fire
    on exactly the answers it should accept.
    """
    labels = _get_nli_model().config.id2label
    for i, label in labels.items():
        if str(label).lower().startswith("entail"):
            return int(i)
    raise RuntimeError(f"no entailment label in {labels!r} for {settings.critic_nli_model}")


def _score_nli_sync(answer: str, chunk_texts: list[str]) -> float:
    """Mean over answer sentences of the best entailment any single chunk provides.

    Per sentence rather than per answer: NLI cross-encoders are trained on short hypothesis pairs,
    and a multi-sentence answer as one hypothesis scores poorly regardless of support. Taking the
    max over chunks then averaging over sentences mirrors how RAGAS faithfulness treats
    statements, which is what this is meant to correlate with.
    """
    sentences = [s for s in _split_sentences(answer) if len(s) > 15]
    if not sentences or not chunk_texts:
        return 0.0

    model = _get_nli_model()
    ent = _entailment_index()

    # One batched call: sentence-major, so the reshape below lines up with `sentences`.
    pairs = [(chunk, sentence) for sentence in sentences for chunk in chunk_texts]
    probs = model.predict(pairs, apply_softmax=True)

    per_sentence = [
        max(probs[i * len(chunk_texts) + j][ent] for j in range(len(chunk_texts)))
        for i in range(len(sentences))
    ]
    return float(sum(per_sentence) / len(per_sentence))


async def score_answer_nli(answer: str, source_chunks: list[dict]) -> float:
    """Entailment-based grounding score in [0, 1]. Higher means better supported."""
    if not answer.strip() or not source_chunks:
        return 0.0
    texts = [c["content"] for c in source_chunks]
    # Off the event loop: the cross-encoder is synchronous and CPU-bound, like the embedder.
    return await asyncio.to_thread(_score_nli_sync, answer, texts)


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
