"""Query decomposition — break a multi-hop question into focused sub-questions.

Each sub-question drives an independent retrieval pass; the merged chunks go to the multi-hop
agent. Output is constrained by a JSON schema and parsed by `json.loads`, replacing the old
code-fence-stripping + `\\[.*?\\]` regex that failed silently on malformed output.

Runs on `settings.effective_router_model` — the same thinking-disabled model as the router.
Decomposition is a short structured-extraction task guided by the few-shot examples below, not
a reasoning task; on the local qwen3 the reasoning variant took 100s+ and produced no better
sub-questions. Sharing the router's model also avoids a model swap between the route and
decompose steps.
"""

import json
import logging

from app.config import settings
from app.agents.utils import extract_content

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 4

DECOMPOSE_SYSTEM_PROMPT = """You decompose a user's multi-hop question into the minimum
set of focused, independently searchable sub-questions.

RULES:
- Output ONLY a JSON array of strings. No prose, no markdown, no code fences.
- Each sub-question must correspond DIRECTLY to a piece of information the user asked for.
- Do NOT add sub-questions about related topics the user didn't ask about.
- Usually 2 sub-questions. Never more than 4.
- Each sub-question should be a short, self-contained search query (not a full sentence).

EXAMPLES:

User: "Which drug requires a test dose before authorization, and what ICD-10 code was added for that drug's indication?"
Output: ["which drug requires a test dose before authorization", "ICD-10 diagnosis code added to buprenorphine policy"]

User: "What is the deductible for in-network services, and what about out-of-network?"
Output: ["in-network deductible amount", "out-of-network deductible amount"]

User: "Who directed Inception and when did they win their first Oscar?"
Output: ["director of Inception movie", "Christopher Nolan first Oscar win year"]"""

# Providers that enforce it return {"sub_questions": [...]}. The local qwen3-router follows the
# few-shot examples instead and returns a bare array; _parse_sub_questions accepts both.
DECOMPOSE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "decompose",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_SUB_QUESTIONS,
                }
            },
            "required": ["sub_questions"],
            "additionalProperties": False,
        },
    },
}


def _parse_sub_questions(raw: str, original_question: str) -> list[str]:
    """Extract sub-questions from a bare JSON array or a {"sub_questions": [...]} object.

    Falls back to the original question (a valid single-hop retrieval) on anything unparseable,
    so a bad decomposition degrades to normal retrieval rather than an error.
    """
    text = raw.strip()
    # A defensive fence strip — one line, not the old multi-branch regex. The prompt forbids
    # fences and the local model obeys, but hosted models occasionally wrap output.
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("[decompose] unparseable %r; falling back to original question", raw[:150])
        return [original_question]

    items = data if isinstance(data, list) else data.get("sub_questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        logger.warning("[decompose] no sub-question list in %r; falling back", raw[:150])
        return [original_question]

    cleaned = [str(q).strip() for q in items if str(q).strip()]
    if not cleaned:
        return [original_question]
    return cleaned[:MAX_SUB_QUESTIONS]


async def decompose_question(client, question: str) -> list[str]:
    """Decompose a multi-hop question into up to 4 focused sub-questions."""
    try:
        response = await client.chat.completions.create(
            model=settings.effective_router_model,
            max_tokens=256,
            temperature=0,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose this question: {question}"},
            ],
            response_format=DECOMPOSE_RESPONSE_FORMAT,
        )
        raw = extract_content(response)
    except Exception as exc:  # noqa: BLE001 — a decompose failure must fall back, not 500
        logger.warning("[decompose] call failed (%s); falling back to original question", exc)
        return [question]

    sub_qs = _parse_sub_questions(raw, question)
    logger.info("[decompose] Q=%r -> %d sub-questions: %s", question[:100], len(sub_qs), sub_qs)
    return sub_qs
