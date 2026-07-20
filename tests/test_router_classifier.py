"""Trained router classifier — real inference plus the graph wiring.

The inference tests load the shipped weights and the MiniLM model, so they exercise the actual
production path. The graph test stubs the classifier to prove routing dispatches to it (and makes
no LLM call) when router_mode=classifier.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.router_classifier import classify_by_embedding


@pytest.mark.asyncio
async def test_classifier_returns_valid_category():
    assert await classify_by_embedding("What is the deductible for the PPO plan?") in (
        "factual", "comparative", "multihop",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected", [
    ("What is the copay for an MRI?", "factual"),
    ("Compare coverage for Drug A and Drug B.", "comparative"),
    ("Which drug requires a test dose, and what ICD-10 code applies to it?", "multihop"),
    # keyword-free phrasings — the classifier should read intent, not surface cues
    ("Is Humira or Ozempic cheaper?", "comparative"),
])
async def test_classifier_predicts_expected(question, expected):
    assert await classify_by_embedding(question) == expected


@pytest.mark.asyncio
async def test_router_mode_classifier_dispatches_to_the_model(monkeypatch):
    """router_mode=classifier must use the trained model and make no LLM call."""
    from app.agents.graph import build_agent_graph

    llm_calls = {"n": 0}

    async def fake_llm_classify(client, q):
        llm_calls["n"] += 1
        return "factual"

    async def fake_clf(question):
        return "comparative"

    async def search(session, query, top_k, document_id=None):
        return [{"chunk_id": "c", "content": "x", "page_number": 1,
                 "element_type": "T", "document_id": "d", "similarity": 0.5}]

    monkeypatch.setattr("app.agents.graph.settings.router_mode", "classifier")
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")
    monkeypatch.setattr("app.agents.graph.classify_by_embedding", fake_clf)
    monkeypatch.setattr("app.agents.graph.classify_query", fake_llm_classify)
    monkeypatch.setattr("app.agents.graph.similarity_search", search)
    monkeypatch.setattr("app.agents.graph.run_comparative_agent",
                        AsyncMock(return_value="answer"))

    state = {
        "original_question": "Compare A and B", "question": "Compare A and B",
        "query_type": "", "sub_questions": [], "source_chunks": [], "answer": "",
        "confidence": 0.0, "retrieval_attempts": 0, "document_id": None, "top_k": 4,
        "session": AsyncMock(), "client": AsyncMock(),
    }
    result = await build_agent_graph().ainvoke(state)

    assert result["query_type"] == "comparative"
    assert llm_calls["n"] == 0, "classifier mode must not call the LLM router"
