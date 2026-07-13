from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import patch, AsyncMock

from app.ingestion.layout_detection import DocumentRegion, detect_layout
from app.ingestion.chunking import semantic_chunk, Chunk


class FakeElement:
    """Stand-in for an unstructured element: stringifies to its text, carries a category."""

    def __init__(self, text: str, category: str = "NarrativeText", page_number: int | None = 1):
        self._text = text
        self.category = category
        self.metadata = SimpleNamespace(page_number=page_number, coordinates=None, filename="f.pdf")

    def __str__(self) -> str:
        return self._text


@pytest.mark.asyncio
async def test_detect_layout_keeps_text_regions():
    """Text elements become regions, preserving content, type and page."""
    elements = [
        FakeElement("Hello world", category="NarrativeText", page_number=1),
        FakeElement("Another paragraph", category="Title", page_number=2),
    ]

    with patch("app.ingestion.layout_detection.partition", return_value=elements):
        regions = await detect_layout(Path("doc.pdf"))

    assert [r.content for r in regions] == ["Hello world", "Another paragraph"]
    assert [r.element_type for r in regions] == ["NarrativeText", "Title"]
    assert [r.page_number for r in regions] == [1, 2]


@pytest.mark.asyncio
async def test_detect_layout_filters_empty_regions():
    """Blank elements are dropped before they reach chunking.

    This filter used to live in the (dead) OCR stage. Losing it would feed empty strings
    into the encoder and produce meaningless chunks, so it is pinned here.
    """
    elements = [
        FakeElement(""),
        FakeElement("   \n  "),
        FakeElement("Valid text", page_number=2),
    ]

    with patch("app.ingestion.layout_detection.partition", return_value=elements):
        regions = await detect_layout(Path("doc.pdf"))

    assert len(regions) == 1
    assert regions[0].content == "Valid text"
    assert regions[0].page_number == 2


@pytest.mark.asyncio
async def test_detect_layout_passes_configured_strategy_to_unstructured(monkeypatch):
    """The strategy setting must actually reach partition().

    It is the only switch that turns OCR on for scanned PDFs ('hi_res'/'ocr_only') or off
    ('fast'). If it silently stopped being forwarded, scanned documents would ingest as
    empty and the failure would look like a bad document rather than a bad config.
    """
    monkeypatch.setattr("app.ingestion.layout_detection.settings.ingestion_strategy", "hi_res")

    with patch(
        "app.ingestion.layout_detection.partition", return_value=[FakeElement("text")]
    ) as mock_partition:
        await detect_layout(Path("scan.pdf"))

    assert mock_partition.call_args.kwargs["strategy"] == "hi_res"


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
