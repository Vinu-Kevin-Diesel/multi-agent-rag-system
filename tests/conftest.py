from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def _pin_ablation_flags(monkeypatch):
    """Pin the ablation flags to their documented defaults for every test.

    Settings are read from `.env`, which the app service passes into the test container — so a
    developer running with ROUTER_MODE=classifier locally would silently change what the graph
    tests exercise. (That happened: the real classifier routed a test question to multihop and
    hit the unmocked decomposer.) Tests that care about a flag monkeypatch it themselves; this
    just guarantees a known starting point.
    """
    monkeypatch.setattr(settings, "router_mode", "llm")
    monkeypatch.setattr(settings, "decompose_enabled", True)
    monkeypatch.setattr(settings, "critic_mode", "cosine")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def chat_completion(content: str | None = None, reasoning_content: str | None = None):
    """Build an OpenAI-shaped chat completion response.

    Deliberately plain objects, not AsyncMock: a bare AsyncMock auto-creates any attribute
    you touch, so a wrong-shaped mock fails silently and leaks a MagicMock into the code
    under test instead of erroring. SimpleNamespace raises AttributeError on a shape
    mismatch, which is what you want a mock to do.
    """
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        model_extra=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def make_llm_client():
    """Factory for a stub LLM client returning a canned chat completion.

    Matches the OpenAI-compatible surface the agents actually call:
    `client.chat.completions.create(...)`.
    """

    def _make(content: str | None = None, reasoning_content: str | None = None):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=chat_completion(content, reasoning_content)
        )
        return client

    return _make


@pytest.fixture
def mock_llm(make_llm_client):
    """Stub LLM client that classifies everything as 'factual'."""
    return make_llm_client(content="factual")


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "00000000-0000-0000-0000-000000000001",
            "content": "The revenue for Q3 2024 was $5.2 billion.",
            "page_number": 1,
            "element_type": "NarrativeText",
            "document_id": "00000000-0000-0000-0000-000000000010",
            "similarity": 0.92,
        },
        {
            "chunk_id": "00000000-0000-0000-0000-000000000002",
            "content": "Operating expenses increased by 12% year-over-year.",
            "page_number": 2,
            "element_type": "NarrativeText",
            "document_id": "00000000-0000-0000-0000-000000000010",
            "similarity": 0.85,
        },
    ]
