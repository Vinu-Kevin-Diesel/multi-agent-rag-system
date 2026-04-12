"""Router agent — classifies query type using Claude."""

from anthropic import AsyncAnthropic

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a document intelligence system.
Classify the user's question into exactly one category:

- "factual": Direct fact lookup from a single document (who, what, when, where)
- "comparative": Requires comparing information across multiple documents or sections
- "multihop": Requires chaining multiple pieces of information to arrive at an answer

Respond with ONLY the category name, nothing else."""


async def classify_query(client: AsyncAnthropic, question: str) -> str:
    """Return one of: factual, comparative, multihop."""
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    category = response.content[0].text.strip().lower()
    if category not in ("factual", "comparative", "multihop"):
        category = "factual"
    return category
