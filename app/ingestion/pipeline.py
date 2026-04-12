"""3-stage ingestion pipeline orchestrator."""

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunking import semantic_chunk
from app.ingestion.layout_detection import detect_layout
from app.ingestion.ocr import apply_ocr
from app.models import Document, DocumentChunk
from app.utils.embeddings import embed_texts


class IngestionPipeline:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def run(self, file_path: Path, filename: str, content_type: str | None = None) -> Document:
        """Execute the full 3-stage ingestion pipeline.

        Stage 1: Layout detection
        Stage 2: OCR for image regions
        Stage 3: Semantic chunking + embedding
        """
        # Stage 1 — layout detection
        regions = await detect_layout(file_path)

        # Stage 2 — OCR
        regions = await apply_ocr(regions)

        # Stage 3 — semantic chunking
        chunks = await semantic_chunk(regions)

        # Compute embeddings in batch
        texts = [c.content for c in chunks]
        embeddings = await embed_texts(texts)

        # Persist
        page_numbers = {r.page_number for r in regions if r.page_number is not None}
        doc = Document(
            id=uuid.uuid4(),
            filename=filename,
            content_type=content_type,
            page_count=max(page_numbers) if page_numbers else None,
        )
        self.session.add(doc)

        for chunk, embedding in zip(chunks, embeddings):
            self.session.add(
                DocumentChunk(
                    document_id=doc.id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    element_type=chunk.element_type,
                    embedding=embedding,
                    metadata_=chunk.metadata,
                )
            )

        await self.session.commit()
        await self.session.refresh(doc)
        doc._chunk_count = len(chunks)
        return doc
