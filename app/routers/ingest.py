import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.ingestion import IngestionPipeline
from app.models import Document
from app.schemas import DocumentListItem, IngestResponse

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
):
    """Upload and ingest a document through the 3-stage pipeline.

    Accepts PDF, DOCX, images, HTML, and plain text.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pipeline = IngestionPipeline(session)
        doc = await pipeline.run(
            file_path=tmp_path,
            filename=file.filename,
            content_type=file.content_type,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    chunk_count = getattr(doc, "_chunk_count", 0)

    return IngestResponse(
        document_id=doc.id,
        filename=doc.filename,
        num_chunks=chunk_count,
        page_count=doc.page_count,
    )


@router.get("/documents", response_model=list[DocumentListItem])
async def list_documents(session: AsyncSession = Depends(get_session)):
    """List all ingested documents."""
    result = await session.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and all its chunks (cascades)."""
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(doc)
    await session.commit()
    return {"status": "deleted", "document_id": str(document_id)}
