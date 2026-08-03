"""NLI critic: scoring semantics, and its wiring through the graph.

The scoring tests exercise the real cross-encoder — the whole claim of day 19 is that this
scorer distinguishes cases the cosine scorer cannot, and a mocked model cannot demonstrate that.
They are skipped when the model is unavailable (no network on a cold image) rather than failing.

The graph tests stub the scorer, so they stay fast and assert only the wiring.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.graph import build_agent_graph

pytestmark = pytest.mark.asyncio


def _chunks(*texts):
    return [
        {"chunk_id": f"c{i}", "content": t, "page_number": 1,
         "element_type": "NarrativeText",
         "document_id": "00000000-0000-0000-0000-000000000010", "similarity": 0.7}
        for i, t in enumerate(texts)
    ]


SOURCE = ("Once approved, authorization for Zeltavir remains valid for 12 months from the date of "
          "approval. Reauthorization requires documented clinical benefit.")


@pytest.fixture(scope="module")
def nli_available():
    try:
        from app.agents.critic_agent import _get_nli_model
        _get_nli_model()
        return True
    except Exception:  # noqa: BLE001 — no network, or the checkpoint moved
        pytest.skip("NLI cross-encoder unavailable")


# ── scoring semantics (real model) ─────────────────────────────────────────

async def test_entailment_index_is_read_from_the_model(nli_available):
    """Hardcoding the column would invert the metric silently: every grounded answer would score
    near zero and the retry loop would fire on exactly the answers it should accept."""
    from app.agents.critic_agent import _entailment_index, _get_nli_model

    labels = _get_nli_model().config.id2label
    assert str(labels[_entailment_index()]).lower().startswith("entail")


async def test_supported_answer_scores_above_unsupported(nli_available):
    from app.agents.critic_agent import score_answer_nli

    supported = await score_answer_nli(
        "Zeltavir authorization is valid for 12 months.", _chunks(SOURCE))
    contradicted = await score_answer_nli(
        "Zeltavir authorization is valid for 6 months.", _chunks(SOURCE))

    assert supported > contradicted
    assert supported > 0.5 > contradicted


async def test_fluent_but_unsupported_answer_is_rejected(nli_available):
    """The case the cosine critic cannot catch, and the reason this scorer exists.

    This answer reuses the source's own vocabulary — 'documented clinical benefit', 'from the date
    of approval' — and is topically identical, so cosine similarity rates it highly. Only the
    number is wrong. Entailment is what notices.
    """
    from app.agents.critic_agent import score_answer, score_answer_nli

    hallucination = ("Zeltavir authorization requires documented clinical benefit and remains "
                     "valid for 24 months from the date of approval.")
    chunks = _chunks(SOURCE)

    cosine = await score_answer(hallucination, chunks)
    nli = await score_answer_nli(hallucination, chunks)

    assert cosine > 0.7, "precondition: cosine should be fooled by the vocabulary overlap"
    assert nli < 0.2, "entailment must reject an unsupported claim"


async def test_empty_and_missing_inputs_score_zero(nli_available):
    from app.agents.critic_agent import score_answer_nli

    assert await score_answer_nli("", _chunks(SOURCE)) == 0.0
    assert await score_answer_nli("Anything at all.", []) == 0.0


# ── graph wiring (stubbed scorer) ──────────────────────────────────────────

def _state(question="How long is authorization valid?"):
    return {
        "original_question": question, "question": question, "query_type": "",
        "sub_questions": [], "source_chunks": [], "answer": "", "confidence": 0.0,
        "retrieval_attempts": 0, "document_id": None, "top_k": 4,
        "session": AsyncMock(), "client": AsyncMock(),
    }


@pytest.fixture
def wiring(monkeypatch):
    calls = {"cosine": 0, "nli": 0, "refine": 0}

    async def search(session, query, top_k, document_id=None):
        return _chunks(SOURCE)

    async def factual(client, q, chunks):
        return "an answer"

    async def cosine(answer, chunks):
        calls["cosine"] += 1
        return 0.99

    async def nli(answer, chunks):
        calls["nli"] += 1
        return 0.10  # below critic_nli_threshold, so the retry loop must fire

    async def refine(original, answer, client):
        calls["refine"] += 1
        return "refined query"

    monkeypatch.setattr("app.agents.graph.settings.router_mode", "off")
    monkeypatch.setattr("app.agents.graph.similarity_search", search)
    monkeypatch.setattr("app.agents.graph.run_factual_agent", factual)
    monkeypatch.setattr("app.agents.graph.score_answer", cosine)
    monkeypatch.setattr("app.agents.graph.score_answer_nli", nli)
    monkeypatch.setattr("app.agents.graph.generate_refined_query", refine)
    return calls


async def test_nli_mode_uses_the_nli_scorer_not_cosine(wiring, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "nli")

    await build_agent_graph().ainvoke(_state())

    assert wiring["nli"] > 0, "critic_mode=nli must call the entailment scorer"
    assert wiring["cosine"] == 0, "critic_mode=nli must not call the cosine scorer"


async def test_cosine_mode_does_not_use_nli(wiring, monkeypatch):
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "cosine")

    await build_agent_graph().ainvoke(_state())

    assert wiring["cosine"] > 0 and wiring["nli"] == 0


async def test_nli_threshold_governs_retry_not_the_cosine_one(wiring, monkeypatch):
    """A score of 0.10 is below the NLI threshold, so the loop must retry.

    Guards the scale bug: the thresholds are not interchangeable, and reading the cosine
    threshold here would still retry — but reading the NLI threshold against a cosine score
    (0.99 vs 0.5) would wrongly accept. This asserts the pairing that makes both correct.
    """
    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "nli")
    monkeypatch.setattr("app.agents.graph.settings.critic_nli_threshold", 0.5)
    monkeypatch.setattr("app.agents.graph.settings.max_retrieval_attempts", 2)

    result = await build_agent_graph().ainvoke(_state())

    assert wiring["refine"] >= 1, "a sub-threshold NLI score must trigger refinement"
    assert result["retrieval_attempts"] == 2


async def test_nli_score_above_threshold_finishes_in_one_pass(wiring, monkeypatch):
    async def high(answer, chunks):
        wiring["nli"] += 1
        return 0.9

    monkeypatch.setattr("app.agents.graph.settings.critic_mode", "nli")
    monkeypatch.setattr("app.agents.graph.settings.critic_nli_threshold", 0.5)
    monkeypatch.setattr("app.agents.graph.score_answer_nli", high)

    result = await build_agent_graph().ainvoke(_state())

    assert wiring["refine"] == 0
    assert result["retrieval_attempts"] == 1
