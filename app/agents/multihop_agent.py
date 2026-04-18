"""Multi-hop agent — chain-of-thought reasoning over multiple retrieval steps."""

from app.config import settings
from app.agents.utils import extract_content

MULTIHOP_SYSTEM_PROMPT = """You are a multi-hop reasoning agent for a document intelligence system.

CRITICAL RULES:
1. Answer ONLY the user's specific question. Do not expand into related topics.
2. Decompose the user's question into the MINIMUM sub-questions needed to answer it — usually 2-3, never more.
3. Each sub-question must correspond DIRECTLY to a piece of information the user asked for. Do not invent sub-questions about "efficacy", "REMS", "clinical trials", or any topic the user did not ask about.
4. For each sub-question, cite the exact source chunk. If the answer is not in the provided chunks, state "Not found in the provided sources" — do NOT guess or fabricate.
5. The Final Answer must be CONCISE (2-4 sentences) and directly answer the user's original question.

OUTPUT FORMAT:
Step 1: [sub-question that MUST be answered to address the user's query]
Answer: [brief answer with citation, or "Not found in the provided sources"]

Step 2: [next sub-question, derived from step 1's answer if chained]
Answer: [brief answer with citation]

(more steps only if strictly needed)

Final Answer: [2-4 sentences that directly answer the user's ORIGINAL question]"""


async def run_multihop_agent(
    client,
    question: str,
    source_chunks: list[dict],
    sub_questions: list[str] | None = None,
) -> str:
    """Generate a multi-hop reasoned answer with explicit chain-of-thought."""
    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c.get('page_number', 'N/A')}]\n{c['content']}"
        for i, c in enumerate(source_chunks)
    )

    sub_q_hint = ""
    if sub_questions:
        sub_q_hint = (
            "\n\nPre-computed sub-questions (use these as your Step 1, Step 2, ...):\n"
            + "\n".join(f"- {sq}" for sq in sub_questions)
        )

    user_content = (
        f"Source chunks:\n{context}\n\n"
        f"User's question: {question}"
        f"{sub_q_hint}\n\n"
        "Answer ONLY what the user asked. Keep it tight."
    )

    response = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": MULTIHOP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return extract_content(response)
