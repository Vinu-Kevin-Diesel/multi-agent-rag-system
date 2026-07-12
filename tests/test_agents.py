import pytest

from app.agents.router_agent import classify_query
from app.agents.critic_agent import score_answer


@pytest.mark.asyncio
async def test_classify_query_returns_valid_type(mock_llm):
    """Router should return a valid query type."""
    result = await classify_query(mock_llm, "What is the revenue?")
    assert result in ("factual", "comparative", "multihop")


@pytest.mark.asyncio
async def test_classify_query_defaults_to_factual(make_llm_client):
    """Unparseable model output should default to factual."""
    client = make_llm_client(content="something_invalid")

    result = await classify_query(client, "What?")
    assert result == "factual"


@pytest.mark.asyncio
async def test_classify_query_reads_the_last_line(make_llm_client):
    """A reasoning model rambles, then states its verdict last. Take the verdict.

    The prompt asks for the category 'on the last line', and the body here deliberately
    name-drops the other two categories so a naive whole-text scan would pick the wrong one.
    """
    client = make_llm_client(
        content=(
            "This is not merely factual, and it is not quite comparative either.\n"
            "It requires chaining one fact into another.\n"
            "multihop"
        )
    )

    assert await classify_query(client, "What treats X, and what code applies?") == "multihop"


@pytest.mark.asyncio
async def test_classify_query_falls_back_to_reasoning_content(make_llm_client):
    """Thinking models can leave `content` empty and put the output in `reasoning_content`.

    Kimi K2.5 does this today; Qwen3 will do it locally. If extract_content stops handling
    it, every query silently classifies as factual and the router looks fine while being dead.
    """
    client = make_llm_client(content=None, reasoning_content="comparative")

    assert await classify_query(client, "Compare A and B") == "comparative"


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
