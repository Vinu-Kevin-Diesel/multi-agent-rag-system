"""Shared utilities for agent LLM calls."""

import logging

logger = logging.getLogger(__name__)

# Cost of the "[Chunk 3 | Page 7]" header and surrounding blank lines each chunk carries
# once the agents render it into the prompt.
_CHUNK_FORMAT_OVERHEAD_TOKENS = 12


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting — roughly 1.3 tokens per whitespace word.

    Deliberately not a real tokenizer. The LLM is provider-agnostic (whatever sits behind
    LLM_BASE_URL can change without a code change), so there is no tokenizer we could
    import that is guaranteed to match. Over-estimating is the safe direction: it trims a
    chunk early rather than overflowing the model's context.
    """
    return int(len(text.split()) * 1.3) + 1


def fit_chunks_to_budget(chunks: list[dict], max_tokens: int) -> list[dict]:
    """Keep the leading chunks that fit inside a token budget; drop the rest.

    Takes a *prefix*, so ordering is the caller's responsibility — whatever it puts first
    survives a trim. Stops at the first chunk that does not fit rather than skipping it,
    so a lower-ranked chunk can never displace a higher-ranked one.

    Always returns at least one chunk: handing an agent zero context guarantees either a
    hallucination or a "not found", which is worse than one oversized chunk.
    """
    if not chunks:
        return []

    kept: list[dict] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.get("content", "")) + _CHUNK_FORMAT_OVERHEAD_TOKENS
        if used + cost > max_tokens:
            break
        kept.append(chunk)
        used += cost

    if not kept:
        logger.warning(
            "[context] top chunk alone exceeds the %d-token budget; passing it through anyway",
            max_tokens,
        )
        return chunks[:1]

    if len(kept) < len(chunks):
        logger.info(
            "[context] budget %d tokens: kept %d/%d chunks (~%d tokens)",
            max_tokens, len(kept), len(chunks), used,
        )
    return kept


def extract_content(response) -> str:
    """Extract text content from an OpenAI-compatible chat completion response.

    Thinking models routinely leave `content` empty and put the output in a non-standard
    field instead — `reasoning_content` is the common one. Hosted and local models alike do
    this, so the fallback chain below is not provider-specific and must not become so.
    """
    choice = response.choices[0]
    msg = choice.message

    # Standard content field
    if msg.content:
        return msg.content

    # Some models use reasoning_content for thinking models
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content

    # Try to get from model_extra (catches non-standard fields)
    if hasattr(msg, "model_extra") and msg.model_extra:
        for key in ("reasoning_content", "thinking", "thought"):
            if key in msg.model_extra and msg.model_extra[key]:
                return msg.model_extra[key]

    return ""
