"""Ablation flags, exercised through the compiled graph.

Each flag must demonstrably change behaviour — that is the whole point of the day-17 ablation.
These drive the real graph with the collaborators stubbed and assert which nodes ran.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.graph import build_agent_graph

QUESTION = "Compare Drug A and Drug B, then say which needs a test dose."


def _chunk(cid="c1"):
    return {
        "chunk_id": cid,
        "content": "Drug A requires a test dose.",
        "page_number": 1,
        "element_type": "NarrativeText",
        "document_id": "00000000-0000-0000-0000-000000000010",
        "similarity": 0.7,
    }


def _state():
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
        "top_k": 4,
        "session": AsyncMock(),
        "client": AsyncMock(),
    }


@pytest.fixture
def spies(monkeypatch):
    """Stub every collaborator and record which ones were invoked."""
    calls = {"classify": 0, "decompose": 0, "score": 0, "refine": 0,
             "factual": 0, "comparative": 0, "multihop": 0}

    async def classify(client, q):
        calls["classify"] += 1
        return "multihop"  # the router, when on, would send this to the multihop path

    async def decompose(client, q):
        calls["decompose"] += 1
        return ["sub a", "sub b"]

    async def search(session, query, top_k, document_id=None):
        return [_chunk()]

    async def factual(client, q, chunks):
        calls["factual"] += 1
        return "factual answer"

    async def comparative(client, q, chunks):
        calls["comparative"] += 1
        return "comparative answer"

    async def multihop(client, q, chunks, sub_questions=None):
        calls["multihop"] += 1
        return "multihop answer"

    async def score(answer, chunks):
        calls["score"] += 1
        return 0.10  # low, so the retry loop fires whenever the critic is on

    async def refine(original, answer, client):
        calls["refine"] += 1
        return "refined query"

    monkeypatch.setattr("app.agents.graph.classify_query", classify)
    monkeypatch.setattr("app.agents.graph.decompose_question", decompose)
    monkeypatch.setattr("app.agents.graph.similarity_search", search)
    monkeypatch.setattr("app.agents.graph.run_factual_agent", factual)
    monkeypatch.setattr("app.agents.graph.run_comparative_agent", comparative)
    monkeypatch.setattr("app.agents.graph.run_multihop_agent", multihop)
    monkeypatch.setattr("app.agents.graph.score_answer", score)
    monkeypatch.setattr("app.agents.graph.generate_refined_query", refine)
    return calls


# ── router_mode ────────────────────────────────────────────────────────────

async def test_router_off_skips_classification_and_forces_factual(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "off")
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")  # isolate routing

    result = await build_agent_graph().ainvoke(_state())

    assert spies["classify"] == 0, "router off must not call the LLM classifier"
    assert result["query_type"] == "factual"
    assert spies["factual"] == 1 and spies["multihop"] == 0


async def test_router_on_classifies(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "llm")
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")

    result = await build_agent_graph().ainvoke(_state())

    assert spies["classify"] == 1
    assert result["query_type"] == "multihop" and spies["multihop"] == 1


# ── decompose_enabled ──────────────────────────────────────────────────────

async def test_decompose_disabled_takes_single_retrieval(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "llm")  # routes multihop
    monkeypatch.setattr("app.agents.graph.settings.decompose_enabled", False)
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")

    result = await build_agent_graph().ainvoke(_state())

    assert spies["decompose"] == 0, "decompose disabled must not split the question"
    assert spies["multihop"] == 1, "still answers via the multihop agent, just single-retrieval"
    assert result["sub_questions"] == []


async def test_decompose_enabled_splits(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "llm")
    monkeypatch.setattr("app.agents.graph.settings.decompose_enabled", True)
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")

    result = await build_agent_graph().ainvoke(_state())

    assert spies["decompose"] == 1
    assert result["sub_questions"] == ["sub a", "sub b"]


# ── critic_mode ────────────────────────────────────────────────────────────

async def test_critic_off_runs_once_no_retry(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "off")
    monkeypatch.setattr("app.agents.graph.settings.decompose_enabled", True)
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "off")

    result = await build_agent_graph().ainvoke(_state())

    assert spies["score"] == 0, "critic off must not score"
    assert spies["refine"] == 0, "critic off must not retry"
    assert result["retrieval_attempts"] == 1
    assert result["confidence"] == 1.0, "off reports the not-critiqued sentinel"


async def test_critic_cosine_retries_on_low_confidence(spies, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.router_mode", "off")
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "cosine")
    # score() returns 0.10 (< threshold), so it retries up to the attempt cap
    monkeypatch.setattr("app.agents.graph.settings.max_retrieval_attempts", 3)

    result = await build_agent_graph().ainvoke(_state())

    assert spies["score"] >= 1 and spies["refine"] >= 1, "low confidence must drive a retry"
    assert result["retrieval_attempts"] == 3, "retries until the attempt cap"
