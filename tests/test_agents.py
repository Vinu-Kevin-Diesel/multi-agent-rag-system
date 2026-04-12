import pytest
from unittest.mock import AsyncMock

from app.agents.router_agent import classify_query
from app.agents.critic_agent import score_answer


@pytest.mark.asyncio
async def test_classify_query_returns_valid_type(mock_anthropic):
    """Router should return a valid query type."""
    result = await classify_query(mock_anthropic, "What is the revenue?")
    assert result in ("factual", "comparative", "multihop")


@pytest.mark.asyncio
async def test_classify_query_defaults_to_factual():
    """Invalid model output should default to factual."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text="something_invalid")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    result = await classify_query(mock_client, "What?")
    assert result == "factual"


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
