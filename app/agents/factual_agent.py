"""Factual agent — direct retrieval and answer generation for fact-based queries."""

from app.config import settings
from app.agents.utils import extract_content

FACTUAL_SYSTEM_PROMPT = """You are a precise factual question answering agent.
Answer the user's question using ONLY the provided source chunks.
If the answer cannot be found in the sources, say so explicitly.
Be concise and cite which source chunk(s) support your answer."""


async def run_factual_agent(
    client,
    question: str,
    source_chunks: list[dict],
) -> str:
    """Generate a factual answer grounded in retrieved chunks."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": FACTUAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Source chunks:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return extract_content(response)
