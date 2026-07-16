"""Token budgeting for the chunks handed to an agent.

The guard exists because prompt size is otherwise unbounded: `top_k` is caller-supplied and
the multi-hop path multiplies it across sub-questions. A model that silently truncates drops
the source chunks and then invents an answer, which is the worst possible failure — it looks
like a confident answer.
"""

from app.agents.utils import estimate_tokens, fit_chunks_to_budget


def _chunk(words: int, chunk_id: str = "c") -> dict:
    return {"chunk_id": chunk_id, "content": " ".join(["word"] * words), "page_number": 1}


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") < estimate_tokens("a b c") < estimate_tokens("a b c d e f")


def test_estimate_tokens_overestimates_rather_than_under():
    """Over-estimating trims early; under-estimating overflows the context. Prefer the former."""
    assert estimate_tokens(" ".join(["word"] * 100)) >= 100


def test_everything_fits_is_a_passthrough():
    chunks = [_chunk(10, "a"), _chunk(10, "b")]
    assert fit_chunks_to_budget(chunks, max_tokens=8000) == chunks


def test_trims_to_the_budget():
    # ~13 tokens + 12 overhead = ~25 each; a 60-token budget fits two, not four.
    chunks = [_chunk(10, str(i)) for i in range(4)]
    kept = fit_chunks_to_budget(chunks, max_tokens=60)

    assert 0 < len(kept) < 4
    assert kept == chunks[: len(kept)], "must keep a prefix — ranking order is meaningful"


def test_stops_rather_than_skipping_an_oversized_chunk():
    """A lower-ranked chunk must never displace a higher-ranked one.

    Retrieval hands chunks over best-first. Skipping past a chunk that doesn't fit to grab a
    smaller, less relevant one behind it would quietly invert that ranking.
    """
    chunks = [_chunk(5, "small-first"), _chunk(500, "huge-second"), _chunk(5, "small-third")]
    kept = fit_chunks_to_budget(chunks, max_tokens=40)

    assert [c["chunk_id"] for c in kept] == ["small-first"]


def test_never_returns_zero_chunks():
    """Zero context guarantees a hallucination or a 'not found'. One oversized chunk is better."""
    chunks = [_chunk(5000, "enormous")]
    kept = fit_chunks_to_budget(chunks, max_tokens=10)

    assert len(kept) == 1


def test_empty_input():
    assert fit_chunks_to_budget([], max_tokens=8000) == []
