import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import get_agent_graph
from app.database import get_session
from app.dependencies import get_llm_client
from app.schemas import QueryRequest, QueryResponse, SourceChunk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """Query ingested documents using the multi-agent RAG system."""
    graph = get_agent_graph()
    model = get_llm_client()

    initial_state = {
        "original_question": request.question,
        "question": request.question,
        "query_type": "",
        "sub_questions": [],
        "source_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "retrieval_attempts": 0,
        "document_id": request.document_id,
        "top_k": request.top_k,
        "session": session,
        "client": model,
    }

    started = time.perf_counter()
    result = await graph.ainvoke(initial_state)
    elapsed = time.perf_counter() - started

    # The per-node [timing] lines break this down; this is the number a user actually feels.
    logger.info(
        "[timing] TOTAL %.2fs | type=%s attempts=%d confidence=%.3f chunks=%d",
        elapsed,
        result["query_type"],
        result["retrieval_attempts"],
        result["confidence"],
        len(result["source_chunks"]),
    )

    sources = [
        SourceChunk(
            chunk_id=c["chunk_id"],
            content=c["content"],
            page_number=c.get("page_number"),
            similarity=c["similarity"],
        )
        for c in result["source_chunks"]
    ]

    return QueryResponse(
        answer=result["answer"],
        query_type=result["query_type"],
        confidence=result["confidence"],
        sources=sources,
        retrieval_attempts=result["retrieval_attempts"],
    )
