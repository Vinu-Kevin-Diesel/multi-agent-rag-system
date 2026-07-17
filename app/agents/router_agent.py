"""Router agent — classifies a query as factual / comparative / multihop.

Output is constrained by a JSON schema rather than parsed out of free text. The old approach
asked for a category "on the last line" and regex-scraped it, which failed silently against a
reasoning model: qwen3 spent the whole token budget thinking, generation was cut off before any
answer token, and the empty response fell through to a `factual` default — so the router worked
on easy questions and failed on exactly the hard ones that needed routing.

Two changes fix that: a schema-constrained response, and a thinking-disabled model for the call
(see settings.router_model). Parsing tolerates both shapes we see in practice — a JSON object
from providers that enforce the schema (NVIDIA NIM), and the bare category word from the local
qwen3-router variant.
"""

import json
import logging

from app.config import settings
from app.agents.utils import extract_content

logger = logging.getLogger(__name__)

CATEGORIES = ("factual", "comparative", "multihop")
DEFAULT_CATEGORY = "factual"

ROUTER_SYSTEM_PROMPT = """You classify a user's question into exactly one category:

- factual: a single fact to look up (who / what / when / where) from one place
- comparative: compare or contrast two or more things, sections, or entities
- multihop: chain facts together — find A, then use A to find B

Respond with only the category: factual, comparative, or multihop."""

ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "route",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": list(CATEGORIES)}},
            "required": ["category"],
            "additionalProperties": False,
        },
    },
}


def _parse_category(raw: str) -> str | None:
    """Extract a category from either a JSON object or a bare word. None if neither matches."""
    text = raw.strip()

    # Providers that enforce the schema return {"category": "..."}.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            value = str(obj.get("category", "")).strip().lower()
            if value in CATEGORIES:
                return value
    except (json.JSONDecodeError, ValueError):
        pass

    # The local no-think variant returns the label directly, e.g. `factual`.
    word = text.strip("\"' \n\t").lower()
    if word in CATEGORIES:
        return word

    # Defensive: a stray word wrapped in other text.
    for category in CATEGORIES:
        if category in word:
            return category

    return None


async def classify_query(client, question: str) -> str:
    """Return one of: factual, comparative, multihop. Defaults to factual on any failure."""
    try:
        response = await client.chat.completions.create(
            model=settings.effective_router_model,
            max_tokens=32,
            temperature=0,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=ROUTER_RESPONSE_FORMAT,
        )
        raw = extract_content(response)
    except Exception as exc:  # noqa: BLE001 — a routing failure must never 500 the query
        logger.warning("[router] call failed (%s) -> %s", exc, DEFAULT_CATEGORY)
        return DEFAULT_CATEGORY

    category = _parse_category(raw)
    if category:
        logger.info("[router] Q=%r -> %s", question[:100], category)
        return category

    logger.warning("[router] unparseable response %r -> %s", raw[:100], DEFAULT_CATEGORY)
    return DEFAULT_CATEGORY
