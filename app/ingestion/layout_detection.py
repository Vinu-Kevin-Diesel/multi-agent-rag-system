"""Stage 1: Layout detection — identify structural regions in documents."""

from dataclasses import dataclass, field
from pathlib import Path

from unstructured.partition.auto import partition


@dataclass
class DocumentRegion:
    content: str
    element_type: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


async def detect_layout(file_path: Path) -> list[DocumentRegion]:
    """Partition a document into typed regions using the unstructured library.

    Handles PDFs, DOCX, images, HTML, plain text, and more.
    """
    elements = partition(filename=str(file_path))

    regions: list[DocumentRegion] = []
    for el in elements:
        content = str(el)
        if not content.strip():
            continue

        meta = el.metadata
        regions.append(
            DocumentRegion(
                content=content,
                element_type=el.category,
                page_number=getattr(meta, "page_number", None),
                metadata={
                    "coordinates": getattr(meta, "coordinates", None),
                    "filename": getattr(meta, "filename", None),
                },
            )
        )
    return regions
