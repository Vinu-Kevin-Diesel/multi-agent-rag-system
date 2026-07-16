"""The context budget guard, exercised through the compiled graph.

The interesting case is multi-hop. The guard keeps a *prefix* of the chunk list, so how
multi_retrieve orders its merged results decides what survives a trim. Concatenating each
sub-question's chunks in blocks would let the first sub-question eat the whole budget and
leave the second with nothing — silently defeating the decomposition. Interleaving prevents it.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.graph import build_agent_graph

QUESTION = "What must happen before Drug A is authorized, and how long does it last?"


def _chunk(chunk_id: str, words: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": " ".join([chunk_id] * words),
        "page_number": 1,
        "element_type": "NarrativeText",
        "document_id": "00000000-0000-0000-0000-000000000010",
        "similarity": 0.7,
    }


def _initial_state() -> dict:
    return {
        "original_question": QUESTION,
        "question": QUESTION,
        "query_type": "",
        "sub_questions": [],
        "source_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "retrieval_attempts": 0,
        "document_id": None,
        "top_k": 6,
        "session": AsyncMock(),
        "client": AsyncMock(),
    }


@pytest.fixture
def multihop_graph(monkeypatch):
    """Force the multihop path and record the chunks the agent actually receives."""
    seen_by_agent: list[list[dict]] = []

    # Two sub-questions, each returning three fat chunks that cannot all fit a small budget.
    results = {
        "sub-a": [_chunk(f"A{i}", 200) for i in range(3)],
        "sub-b": [_chunk(f"B{i}", 200) for i in range(3)],
    }

    async def fake_search(session, query, top_k, document_id=None):
        return results[query][:top_k]

    async def fake_classify(client, question):
        return "multihop"

    async def fake_decompose(client, question):
        return ["sub-a", "sub-b"]

    async def fake_multihop(client, question, chunks, sub_questions=None):
        seen_by_agent.append(chunks)
        return "answer"

    async def fake_score(answer, chunks):
        return 0.99  # accept first attempt; the retry loop isn't what's under test

    monkeypatch.setattr("app.agents.graph.similarity_search", fake_search)
    monkeypatch.setattr("app.agents.graph.classify_query", fake_classify)
    monkeypatch.setattr("app.agents.graph.decompose_question", fake_decompose)
    monkeypatch.setattr("app.agents.graph.run_multihop_agent", fake_multihop)
    monkeypatch.setattr("app.agents.graph.score_answer", fake_score)

    return seen_by_agent


async def test_budget_trim_keeps_evidence_for_every_sub_question(multihop_graph, monkeypatch):
    """A trim must not starve a hop.

    Each chunk costs ~273 estimated tokens, so this budget fits exactly 3 of the 6. That
    number is chosen to discriminate: under block concatenation the merged order is
    [A0,A1,A2,B0,B1,B2] and the trim keeps A0,A1,A2 — sub-question B is wiped out entirely
    and the agent answers half the question at full confidence. Interleaving gives
    [A0,B0,A1,...], so the same trim keeps both.

    (A more generous budget passes either way and would prove nothing — verified.)
    """
    monkeypatch.setattr("app.agents.graph.settings.max_context_tokens", 850)

    await build_agent_graph().ainvoke(_initial_state())

    chunks = multihop_graph[0]
    ids = [c["chunk_id"] for c in chunks]

    assert len(ids) < 6, "budget should have trimmed something, or the test proves nothing"
    assert any(i.startswith("A") for i in ids), "sub-question A starved"
    assert any(i.startswith("B") for i in ids), "sub-question B starved"


async def test_generous_budget_passes_everything_through(multihop_graph, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.max_context_tokens", 100_000)

    await build_agent_graph().ainvoke(_initial_state())

    assert len(multihop_graph[0]) == 6


async def test_budget_caps_an_unbounded_top_k(monkeypatch):
    """top_k is caller-supplied and has no ceiling; the budget is what actually bounds context."""
    huge = [_chunk(f"C{i}", 200) for i in range(500)]
    seen: list[list[dict]] = []

    async def fake_search(session, query, top_k, document_id=None):
        return huge[:top_k]

    async def fake_factual(client, question, chunks):
        seen.append(chunks)
        return "answer"

    monkeypatch.setattr("app.agents.graph.similarity_search", fake_search)
    monkeypatch.setattr("app.agents.graph.classify_query", AsyncMock(return_value="factual"))
    monkeypatch.setattr("app.agents.graph.run_factual_agent", fake_factual)
    monkeypatch.setattr("app.agents.graph.score_answer", AsyncMock(return_value=0.99))

    state = _initial_state()
    state["top_k"] = 500
    await build_agent_graph().ainvoke(state)

    assert len(seen[0]) < 500
