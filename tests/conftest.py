import pytest
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_anthropic():
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.content = [AsyncMock(text="factual")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


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
