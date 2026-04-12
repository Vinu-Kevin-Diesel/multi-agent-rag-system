"""Multi-hop agent — chain-of-thought reasoning over multiple retrieval steps."""

from anthropic import AsyncAnthropic

MULTIHOP_SYSTEM_PROMPT = """You are a multi-hop reasoning agent for a document intelligence system.
For complex questions requiring chained reasoning:
1. Break the question into sub-questions
2. Answer each sub-question using the provided source chunks
3. Chain the intermediate answers to produce the final answer

Show your reasoning chain clearly. Cite sources for each step."""


async def run_multihop_agent(
    client: AsyncAnthropic,
    question: str,
    source_chunks: list[dict],
) -> str:
    """Generate a multi-hop reasoned answer with explicit chain-of-thought."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=MULTIHOP_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source chunks:\n{context}\n\nMulti-hop question: {question}",
            }
        ],
    )
    return response.content[0].text
