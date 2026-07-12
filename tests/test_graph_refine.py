"""Graph-level tests for the critic refinement loop.

The invariant under test, in two halves:

    - the SEARCH side moves    -> retrieval re-runs with a refined query
    - the ANSWER side is anchored -> agents always answer the user's original question

Both halves have to hold at once. Losing the second one is the bug these tests pin down:
before the original_question/question split, a refined search string was handed to the
agent as the question to answer, so any retried query answered something the user never asked.

Everything below the graph (LLM, DB, embeddings) is stubbed, so these run offline.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.graph import build_agent_graph

ORIGINAL = "What are the coverage criteria for Drug A?"


def _chunk(content: str = "Coverage requires prior authorization.") -> dict:
    return {
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "content": content,
        "page_number": 1,
        "element_type": "NarrativeText",
        "document_id": "00000000-0000-0000-0000-000000000010",
        "similarity": 0.5,
    }


def _initial_state() -> dict:
    return {
        "original_question": ORIGINAL,
        "question": ORIGINAL,
        "query_type": "",
        "sub_questions": [],
        "source_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "retrieval_attempts": 0,
        "document_id": None,
        "top_k": 5,
        "session": AsyncMock(),
        "client": AsyncMock(),
    }


@pytest.fixture
def spy(monkeypatch):
    """Stub the graph's collaborators and record what they were called with.

    Patch targets are on app.agents.graph, not the defining modules: graph.py imports
    these names directly, so the module-global binding is what the nodes resolve.
    """
    retrieval_queries: list[str] = []
    agent_questions: list[str] = []

    async def fake_similarity_search(session, query, top_k, document_id=None):
        retrieval_queries.append(query)
        return [_chunk()]

    async def fake_classify(client, question):
        return "factual"

    async def fake_factual(client, question, chunks):
        agent_questions.append(question)
        return f"low-confidence answer {len(agent_questions)}"

    monkeypatch.setattr("app.agents.graph.similarity_search", fake_similarity_search)
    monkeypatch.setattr("app.agents.graph.classify_query", fake_classify)
    monkeypatch.setattr("app.agents.graph.run_factual_agent", fake_factual)

    return {"retrieval_queries": retrieval_queries, "agent_questions": agent_questions}


def _stub_scores(monkeypatch, scores: list[float]) -> None:
    """Force the critic to hand back a fixed sequence of confidences."""
    it = iter(scores)

    async def fake_score(answer, chunks):
        return next(it)

    monkeypatch.setattr("app.agents.graph.score_answer", fake_score)


def _stub_refinements(monkeypatch, refinements: list[str]) -> None:
    """Force the refiner to hand back a fixed sequence of search strings."""
    it = iter(refinements)

    async def fake_refine(original_question, answer, client):
        return next(it)

    monkeypatch.setattr("app.agents.graph.generate_refined_query", fake_refine)


async def test_agents_answer_the_original_question_after_refinement(spy, monkeypatch):
    """A refined query drives retrieval; the agent still answers what the user asked."""
    _stub_scores(monkeypatch, [0.10, 0.90])  # reject attempt 1, accept attempt 2
    _stub_refinements(monkeypatch, ["drug a prior authorization criteria"])

    result = await build_agent_graph().ainvoke(_initial_state())

    assert result["retrieval_attempts"] == 2, "critic should have forced exactly one retry"

    # Search side moved.
    assert spy["retrieval_queries"] == [ORIGINAL, "drug a prior authorization criteria"]

    # Answer side stayed anchored. This is the assertion that fails on the old code,
    # where the second call received the refined search string instead.
    assert spy["agent_questions"] == [ORIGINAL, ORIGINAL]


async def test_trivial_refinement_falls_back_to_the_original_not_the_drifted_query(
    spy, monkeypatch
):
    """An empty refinement rewinds the search to the original question, not the last one.

    Needs two refinement rounds to discriminate: the old fallback returned state['question'],
    which by attempt 3 is already the drifted query, so the guard silently re-ran the same
    failing search instead of rewinding.
    """
    _stub_scores(monkeypatch, [0.10, 0.10, 0.90])
    _stub_refinements(monkeypatch, ["drug a prior authorization criteria", "   "])

    result = await build_agent_graph().ainvoke(_initial_state())

    assert result["retrieval_attempts"] == 3
    assert spy["retrieval_queries"] == [
        ORIGINAL,
        "drug a prior authorization criteria",
        ORIGINAL,  # old code re-ran "drug a prior authorization criteria" here
    ]
    assert spy["agent_questions"] == [ORIGINAL, ORIGINAL, ORIGINAL]
