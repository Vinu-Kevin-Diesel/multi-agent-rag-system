"""pgvector HNSW similarity search."""

import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
from app.utils.embeddings import embed_single


async def similarity_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> list[dict]:
    """Retrieve the top-k most similar chunks to a query string.

    Returns dicts with keys: chunk_id, content, page_number, similarity, element_type.
    """
    query_embedding = await embed_single(query)

    # Build cosine similarity query using pgvector <=> operator
    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.element_type,
            DocumentChunk.document_id,
            (1 - distance_expr).label("similarity"),
        )
        .order_by(distance_expr)
        .limit(top_k)
    )

    if document_id is not None:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "chunk_id": row.id,
            "content": row.content,
            "page_number": row.page_number,
            "element_type": row.element_type,
            "document_id": row.document_id,
            "similarity": float(row.similarity),
        }
        for row in rows
    ]
