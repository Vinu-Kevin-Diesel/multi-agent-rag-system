import pytest
from unittest.mock import patch, AsyncMock

from app.ingestion.layout_detection import DocumentRegion
from app.ingestion.ocr import apply_ocr
from app.ingestion.chunking import semantic_chunk, Chunk


@pytest.mark.asyncio
async def test_ocr_passes_through_text_regions():
    """Text regions should pass through OCR unchanged."""
    regions = [
        DocumentRegion(content="Hello world", element_type="NarrativeText", page_number=1),
        DocumentRegion(content="Another paragraph", element_type="Title", page_number=1),
    ]
    result = await apply_ocr(regions)
    assert len(result) == 2
    assert result[0].content == "Hello world"
    assert result[1].content == "Another paragraph"


@pytest.mark.asyncio
async def test_ocr_filters_empty_regions():
    """Empty regions should be dropped."""
    regions = [
        DocumentRegion(content="", element_type="NarrativeText", page_number=1),
        DocumentRegion(content="  ", element_type="NarrativeText", page_number=1),
        DocumentRegion(content="Valid text", element_type="NarrativeText", page_number=2),
    ]
    result = await apply_ocr(regions)
    assert len(result) == 1
    assert result[0].content == "Valid text"


@pytest.mark.asyncio
async def test_semantic_chunk_produces_chunks():
    """Semantic chunking should produce non-empty chunks from regions."""
    regions = [
        DocumentRegion(
            content="This is the first sentence. This is the second sentence. And a third one here.",
            element_type="NarrativeText",
            page_number=1,
        ),
    ]

    with patch("app.ingestion.chunking._get_encoder") as mock_encoder:
        import numpy as np
        mock_model = AsyncMock()
        mock_model.encode = lambda texts, **kwargs: np.random.rand(len(texts), 384)
        mock_encoder.return_value = mock_model

        chunks = await semantic_chunk(regions)

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].chunk_index == 0
