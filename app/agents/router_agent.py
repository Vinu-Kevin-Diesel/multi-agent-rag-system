"""Router agent — classifies query type using LLM."""

import logging
import re

from app.config import settings
from app.agents.utils import extract_content

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a document intelligence system.
Classify the user's question into exactly one category:

- "factual": Direct fact lookup from a single document (one thing to find: who, what, when, where)
- "comparative": Requires comparing or contrasting information between 2+ documents, sections, or entities
- "multihop": Requires chaining multiple pieces of information together (find A, then use A to find B)

Examples:
- "What is the deductible?" -> factual
- "Compare coverage for Drug A vs Drug B" -> comparative
- "What condition does Drug A treat, and what ICD-10 code applies?" -> multihop
- "If a patient has never taken Drug X, what steps must they complete AND how long does authorization last?" -> multihop

Respond with ONLY the single category word (factual, comparative, or multihop) on the last line of your response."""


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
    raw = extract_content(response)
    text = raw.strip().lower()

    # Log what the router actually saw (helpful for debugging misclassifications)
    logger.info("[router] Q=%r | raw_response=%r", question[:120], raw[:300])

    # Strategy 1: Look at the LAST line (reasoning models put their final answer there)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        for cat in ("multihop", "comparative", "factual"):
            if re.search(rf"\b{cat}\b", last_line):
                logger.info("[router] -> %s (from last line: %r)", cat, last_line[:80])
                return cat

    # Strategy 2: Fallback - check anywhere, but prefer more specific categories first
    for cat in ("multihop", "comparative", "factual"):
        if re.search(rf"\b{cat}\b", text):
            logger.info("[router] -> %s (fallback scan)", cat)
            return cat

    logger.warning("[router] -> factual (no match found in response)")
    return "factual"
