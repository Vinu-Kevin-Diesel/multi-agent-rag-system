from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import build_agent_graph
from app.database import get_session
from app.dependencies import get_anthropic_client
from app.schemas import QueryRequest, QueryResponse, SourceChunk

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """Query ingested documents using the multi-agent RAG system.

    The router agent classifies the query, a specialized sub-agent generates
    an answer, and the critic agent validates it — re-retrieving if needed.
    """
    graph = build_agent_graph()
    client = get_anthropic_client()

    initial_state = {
        "question": request.question,
        "query_type": "",
        "source_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "retrieval_attempts": 0,
        "document_id": request.document_id,
        "top_k": request.top_k,
        "session": session,
        "client": client,
    }

    result = await graph.ainvoke(initial_state)

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
