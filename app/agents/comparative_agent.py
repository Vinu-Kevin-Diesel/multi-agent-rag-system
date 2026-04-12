"""Comparative agent — synthesizes answers by comparing across multiple sources."""

from app.config import settings
from app.agents.utils import extract_content

COMPARATIVE_SYSTEM_PROMPT = """You are a comparative analysis agent for a document intelligence system.
You compare information across multiple document chunks to answer questions that involve
contrasts, similarities, rankings, or multi-source synthesis.
Structure your answer clearly, highlighting differences and commonalities.
Cite the source chunks that support each point."""


async def run_comparative_agent(
    client,
    question: str,
    source_chunks: list[dict],
) -> str:
    """Generate a comparative analysis answer from multiple source chunks."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Doc {str(c.get('document_id', 'N/A'))[:8]} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": COMPARATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Source chunks:\n{context}\n\nComparative question: {question}"},
        ],
    )
    return extract_content(response)
