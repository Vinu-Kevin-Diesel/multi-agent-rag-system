"""Query decomposition — break a multi-hop question into focused sub-questions.

Each sub-question will be used for an independent retrieval pass, and the
merged chunks will be passed to the multi-hop agent.
"""

import json
import logging
import re

from app.config import settings
from app.agents.utils import extract_content

logger = logging.getLogger(__name__)

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


async def decompose_question(client, question: str) -> list[str]:
    """Decompose a multi-hop question into 2-4 focused sub-questions."""
    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this question: {question}"},
        ],
    )
    raw = extract_content(response).strip()
    logger.info("[decompose] Q=%r | raw=%r", question[:120], raw[:300])

    # Strip code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)

    # Find the first JSON array
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not match:
        logger.warning("[decompose] No JSON array found; falling back to original question")
        return [question]

    try:
        sub_qs = json.loads(match.group(0))
        if not isinstance(sub_qs, list):
            raise ValueError("Not a list")
        sub_qs = [str(q).strip() for q in sub_qs if str(q).strip()]
        if not sub_qs:
            raise ValueError("Empty list")
        logger.info("[decompose] -> %d sub-questions: %s", len(sub_qs), sub_qs)
        return sub_qs[:4]  # cap at 4 to prevent blowup
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("[decompose] Parse failed (%s); falling back to original question", e)
        return [question]
