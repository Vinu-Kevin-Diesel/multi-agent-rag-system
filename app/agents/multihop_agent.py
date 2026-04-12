"""Multi-hop agent — chain-of-thought reasoning over multiple retrieval steps."""

from app.config import settings
from app.agents.utils import extract_content

MULTIHOP_SYSTEM_PROMPT = """You are a multi-hop reasoning agent for a document intelligence system.
For complex questions requiring chained reasoning:
1. Break the question into sub-questions
2. Answer each sub-question using the provided source chunks
3. Chain the intermediate answers to produce the final answer

Show your reasoning chain clearly. Cite sources for each step."""


async def run_multihop_agent(
    client,
    question: str,
    source_chunks: list[dict],
) -> str:
    """Generate a multi-hop reasoned answer with explicit chain-of-thought."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": MULTIHOP_SYSTEM_PROMPT},
            {"role": "user", "content": f"Source chunks:\n{context}\n\nMulti-hop question: {question}"},
        ],
    )
    return extract_content(response)
