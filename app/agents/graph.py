"""LangGraph state graph — orchestrates routing, sub-agents, critic loop.

Flow:
    route → (factual/comparative: retrieve → agent)
          → (multihop:  decompose → multi_retrieve → multihop_agent)
    → critic → (refine → loop back) | done

State carries two distinct strings: `original_question` is what the user asked and is what
the agents must answer; `question` is the current search string, which the critic loop may
refine. They start equal and diverge on the first refinement.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.comparative_agent import run_comparative_agent
from app.agents.critic_agent import generate_refined_query, score_answer
from app.agents.decompose import decompose_question
from app.agents.factual_agent import run_factual_agent
from app.agents.multihop_agent import run_multihop_agent
from app.agents.router_agent import classify_query
from app.config import settings
from app.retrieval.vector_store import similarity_search

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    # What the user asked. Written once at entry; never mutated. Agents answer THIS.
    original_question: str
    # The current search string. The critic loop is free to refine it. Retrieval uses THIS.
    question: str
    query_type: str
    sub_questions: list[str]
    source_chunks: list[dict]
    answer: str
    confidence: float
    retrieval_attempts: int
    document_id: uuid.UUID | None
    top_k: int
    # injected dependencies
    session: Any
    client: Any


# ── Nodes ─────────────────────────────────────────────────────────────────

async def route_node(state: AgentState) -> dict:
    """Classify the query type."""
    query_type = await classify_query(state["client"], state["question"])
    return {"query_type": query_type}


async def retrieve_node(state: AgentState) -> dict:
    """Single-pass retrieval for factual/comparative queries."""
    chunks = await similarity_search(
        session=state["session"],
        query=state["question"],
        top_k=state["top_k"],
        document_id=state.get("document_id"),
    )
    return {
        "source_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


async def decompose_node(state: AgentState) -> dict:
    """Break the multi-hop question into focused sub-questions."""
    sub_qs = await decompose_question(state["client"], state["question"])
    return {"sub_questions": sub_qs}


async def multi_retrieve_node(state: AgentState) -> dict:
    """Iterative retrieval: run similarity_search for each sub-question, merge + dedupe."""
    sub_qs = state.get("sub_questions") or [state["question"]]
    top_k_total = state["top_k"]
    # Allocate a fair share of chunks per sub-question (min 3 each)
    per_sub = max(3, math.ceil(top_k_total / len(sub_qs)))

    all_chunks: list[dict] = []
    seen_ids: set[str] = set()

    for sq in sub_qs:
        chunks = await similarity_search(
            session=state["session"],
            query=sq,
            top_k=per_sub,
            document_id=state.get("document_id"),
        )
        for c in chunks:
            chunk_id = str(c.get("chunk_id", ""))
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(c)

    logger.info(
        "[multi_retrieve] %d sub-questions -> %d unique chunks (per_sub=%d)",
        len(sub_qs), len(all_chunks), per_sub,
    )

    return {
        "source_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


async def factual_node(state: AgentState) -> dict:
    answer = await run_factual_agent(
        state["client"], state["original_question"], state["source_chunks"]
    )
    return {"answer": answer}


async def comparative_node(state: AgentState) -> dict:
    answer = await run_comparative_agent(
        state["client"], state["original_question"], state["source_chunks"]
    )
    return {"answer": answer}


async def multihop_node(state: AgentState) -> dict:
    answer = await run_multihop_agent(
        state["client"],
        state["original_question"],
        state["source_chunks"],
        sub_questions=state.get("sub_questions"),
    )
    return {"answer": answer}


async def critic_node(state: AgentState) -> dict:
    """Score the answer against source chunks."""
    confidence = await score_answer(state["answer"], state["source_chunks"])
    return {"confidence": confidence}


async def refine_query_node(state: AgentState) -> dict:
    """Generate a refined query for re-retrieval.

    Guards against empty/trivial refinements by falling back to the original question.
    Clears sub_questions so the refined query will be re-decomposed if multihop.
    """
    refined = await generate_refined_query(state["question"], state["answer"], state["client"])
    if not refined or len(refined.strip()) < 5:
        return {"question": state["question"], "sub_questions": []}
    return {"question": refined, "sub_questions": []}


# ── Edge selectors ────────────────────────────────────────────────────────

def route_to_path(state: AgentState) -> str:
    """After classification, route to the standard or multihop retrieval path."""
    if state["query_type"] == "multihop":
        return "multihop_path"
    return "standard_path"


def route_to_standard_agent(state: AgentState) -> str:
    """Route standard (non-multihop) queries to factual or comparative agent."""
    if state["query_type"] == "comparative":
        return "comparative"
    return "factual"


def refine_to_path(state: AgentState) -> str:
    """After refine, loop back to the appropriate retrieval node based on query_type."""
    if state["query_type"] == "multihop":
        return "decompose"
    return "retrieve"


def should_retry(state: AgentState) -> str:
    """Decide whether to re-retrieve or finish."""
    if (
        state["confidence"] < settings.critic_similarity_threshold
        and state["retrieval_attempts"] < settings.max_retrieval_attempts
    ):
        return "refine"
    return "done"


# ── Graph construction ────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Construct the LangGraph state graph."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("multi_retrieve", multi_retrieve_node)
    graph.add_node("factual", factual_node)
    graph.add_node("comparative", comparative_node)
    graph.add_node("multihop", multihop_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refine", refine_query_node)

    # Entry
    graph.set_entry_point("route")

    # route → standard path (retrieve) OR multihop path (decompose)
    graph.add_conditional_edges(
        "route",
        route_to_path,
        {
            "standard_path": "retrieve",
            "multihop_path": "decompose",
        },
    )

    # Standard path: retrieve → factual or comparative
    graph.add_conditional_edges(
        "retrieve",
        route_to_standard_agent,
        {
            "factual": "factual",
            "comparative": "comparative",
        },
    )

    # Multihop path: decompose → multi_retrieve → multihop_agent
    graph.add_edge("decompose", "multi_retrieve")
    graph.add_edge("multi_retrieve", "multihop")

    # All agents → critic
    graph.add_edge("factual", "critic")
    graph.add_edge("comparative", "critic")
    graph.add_edge("multihop", "critic")

    # critic → refine or done
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"refine": "refine", "done": END},
    )

    # refine → loop back to appropriate retrieval node
    graph.add_conditional_edges(
        "refine",
        refine_to_path,
        {"retrieve": "retrieve", "decompose": "decompose"},
    )

    return graph.compile()
