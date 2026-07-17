import pytest

from app.agents.router_agent import classify_query, _parse_category
from app.agents.critic_agent import score_answer


def test_parse_category_from_json_object():
    """Providers that enforce the schema (NVIDIA NIM) return a JSON object."""
    assert _parse_category('{"category": "multihop"}') == "multihop"
    assert _parse_category('{"category":"comparative"}') == "comparative"


def test_parse_category_from_bare_word():
    """The local no-think qwen3-router variant returns the label directly."""
    assert _parse_category("factual") == "factual"
    assert _parse_category("  multihop\n") == "multihop"
    assert _parse_category('"comparative"') == "comparative"


def test_parse_category_rejects_unknown():
    """A value outside the enum is not silently coerced — the caller decides the fallback."""
    assert _parse_category("banana") is None
    assert _parse_category("") is None
    assert _parse_category('{"category": "sideways"}') is None


@pytest.mark.asyncio
async def test_classify_query_returns_valid_type(mock_llm):
    result = await classify_query(mock_llm, "What is the revenue?")
    assert result in ("factual", "comparative", "multihop")


@pytest.mark.asyncio
async def test_classify_query_parses_schema_json(make_llm_client):
    """End to end with a schema-enforcing provider's JSON object."""
    client = make_llm_client(content='{"category": "multihop"}')
    assert await classify_query(client, "What treats X, and what code applies?") == "multihop"


@pytest.mark.asyncio
async def test_classify_query_parses_bare_word(make_llm_client):
    """End to end with the local variant's bare-word output."""
    client = make_llm_client(content="comparative")
    assert await classify_query(client, "Compare A and B") == "comparative"


@pytest.mark.asyncio
async def test_classify_query_defaults_to_factual_on_garbage(make_llm_client):
    """Unparseable output must not raise or invent a category."""
    client = make_llm_client(content="something_invalid")
    assert await classify_query(client, "What?") == "factual"


@pytest.mark.asyncio
async def test_classify_query_defaults_to_factual_on_error(make_llm_client):
    """A routing failure must degrade to a default, never 500 the whole query.

    This is the failure the day-5 baseline hit: an empty/failed router response used to
    fall through silently. Now it is explicit and contained.
    """
    client = make_llm_client(content="factual")
    client.chat.completions.create.side_effect = RuntimeError("connection reset")
    assert await classify_query(client, "anything") == "factual"


@pytest.mark.asyncio
async def test_score_answer_returns_float(sample_chunks):
    """Critic score_answer should return a float between 0 and 1."""
    with pytest.MonkeyPatch.context() as mp:
        import numpy as np

        async def fake_embed(texts):
            return [np.random.rand(1536).tolist() for _ in texts]

        mp.setattr("app.agents.critic_agent.embed_texts", fake_embed)

        score = await score_answer("Revenue was $5.2 billion", sample_chunks)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
