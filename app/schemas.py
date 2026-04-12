from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    num_chunks: int
    page_count: int | None


class SourceChunk(BaseModel):
    chunk_id: UUID
    content: str
    page_number: int | None
    similarity: float


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    document_id: UUID | None = None


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    confidence: float
    sources: list[SourceChunk]
    retrieval_attempts: int


class DocumentListItem(BaseModel):
    id: UUID
    filename: str
    content_type: str | None
    page_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
