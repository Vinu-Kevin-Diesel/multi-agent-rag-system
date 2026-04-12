import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.ingestion import IngestionPipeline
from app.schemas import IngestResponse

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
