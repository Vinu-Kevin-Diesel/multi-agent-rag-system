"""LangGraph state graph — orchestrates routing, sub-agents, critic loop."""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.comparative_agent import run_comparative_agent
from app.agents.critic_agent import generate_refined_query, score_answer
from app.agents.factual_agent import run_factual_agent
from app.agents.multihop_agent import run_multihop_agent
from app.agents.router_agent import classify_query
from app.config import settings
from app.retrieval.vector_store import similarity_search


class AgentState(TypedDict):
    question: str
    query_type: str
    source_chunks: list[dict]
    answer: str
    confidence: float
    retrieval_attempts: int
    document_id: uuid.UUID | None
    top_k: int
    # injected dependencies
    session: Any
    client: Any


async def route_node(state: AgentState) -> dict:
    """Classify the query type."""
    query_type = await classify_query(state["client"], state["question"])
    return {"query_type": query_type}


async def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant chunks from pgvector."""
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


async def factual_node(state: AgentState) -> dict:
    answer = await run_factual_agent(state["client"], state["question"], state["source_chunks"])
    return {"answer": answer}


async def comparative_node(state: AgentState) -> dict:
    answer = await run_comparative_agent(state["client"], state["question"], state["source_chunks"])
    return {"answer": answer}


async def multihop_node(state: AgentState) -> dict:
    answer = await run_multihop_agent(state["client"], state["question"], state["source_chunks"])
    return {"answer": answer}


async def critic_node(state: AgentState) -> dict:
    """Score the answer against source chunks."""
    confidence = await score_answer(state["answer"], state["source_chunks"])
    return {"confidence": confidence}


async def refine_query_node(state: AgentState) -> dict:
    """Generate a refined query for re-retrieval.

    Guards against empty/trivial refinements by falling back to the original question.
    """
    refined = await generate_refined_query(state["question"], state["answer"], state["client"])
    # Sanity check: if the refined query is empty or too short, keep the original
    if not refined or len(refined.strip()) < 5:
        return {"question": state["question"]}
    return {"question": refined}


def should_retry(state: AgentState) -> str:
    """Decide whether to re-retrieve or finish."""
    if (
        state["confidence"] < settings.critic_similarity_threshold
        and state["retrieval_attempts"] < settings.max_retrieval_attempts
    ):
        return "refine"
    return "done"


def route_to_agent(state: AgentState) -> str:
    """Route to the appropriate sub-agent based on query type."""
    return state["query_type"]


def build_agent_graph() -> StateGraph:
    """Construct the LangGraph state graph.

    Flow:
        route → retrieve → sub-agent → critic → (refine → retrieve → ...) | done
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("factual", factual_node)
    graph.add_node("comparative", comparative_node)
    graph.add_node("multihop", multihop_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refine", refine_query_node)

    # Entry point
    graph.set_entry_point("route")

    # Route → retrieve
    graph.add_edge("route", "retrieve")

    # Retrieve → sub-agent (conditional on query type)
    graph.add_conditional_edges(
        "retrieve",
        route_to_agent,
        {
            "factual": "factual",
            "comparative": "comparative",
            "multihop": "multihop",
        },
    )

    # Sub-agents → critic
    graph.add_edge("factual", "critic")
    graph.add_edge("comparative", "critic")
    graph.add_edge("multihop", "critic")

    # Critic → conditional (retry or done)
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "refine": "refine",
            "done": END,
        },
    )

    # Refine → retrieve (loop back)
    graph.add_edge("refine", "retrieve")

    return graph.compile()
