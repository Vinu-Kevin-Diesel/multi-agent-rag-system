"""Factual agent — direct retrieval and answer generation for fact-based queries."""

from anthropic import AsyncAnthropic

FACTUAL_SYSTEM_PROMPT = """You are a precise factual question answering agent.
Answer the user's question using ONLY the provided source chunks.
If the answer cannot be found in the sources, say so explicitly.
Be concise and cite which source chunk(s) support your answer."""


async def run_factual_agent(
    client: AsyncAnthropic,
    question: str,
    source_chunks: list[dict],
) -> str:
    """Generate a factual answer grounded in retrieved chunks."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=FACTUAL_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source chunks:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text
