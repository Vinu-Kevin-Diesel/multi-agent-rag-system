"""Router agent — classifies query type using LLM."""

from app.config import settings
from app.agents.utils import extract_content

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a document intelligence system.
Classify the user's question into exactly one category:

- "factual": Direct fact lookup from a single document (who, what, when, where)
- "comparative": Requires comparing information across multiple documents or sections
- "multihop": Requires chaining multiple pieces of information to arrive at an answer

Respond with ONLY the category name, nothing else. Do not explain your reasoning."""


async def classify_query(client, question: str) -> str:
    """Return one of: factual, comparative, multihop."""
    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    text = extract_content(response)
    # Extract just the category from potentially verbose responses
    text_lower = text.strip().lower()
    for cat in ("factual", "comparative", "multihop"):
        if cat in text_lower:
            return cat
    return "factual"
