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


class AblationFlags(BaseModel):
    """The pipeline configuration actually in force, reported so an eval run can record it.

    The eval harness labels each run with a `--config` name, but that label is supplied by the
    caller. Without this, a run collected against the wrong flags is indistinguishable from a
    correct one — and an ablation study whose labels silently disagree with its configurations
    measures nothing. Reading it back from the running app makes the label verifiable.
    """

    router_mode: str
    router_model: str
    decompose_enabled: bool
    critic_mode: str
    critic_retry_enabled: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    flags: AblationFlags
