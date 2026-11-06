"""The fixed agent graph: every edge is unconditional except the final check."""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.agent import tools
from insurance_rag.agent.budget import spend
from insurance_rag.agent.state import AgentAnswer, AgentState
from insurance_rag.config import settings
from insurance_rag.generation.chain import (
    MAX_CONTEXT_TOKENS,
    REFUSAL,
    SYSTEM,
    USER,
    cited,
    format_context,
)
from insurance_rag.providers import chat_model
from insurance_rag.ratelimit import with_retry
from insurance_rag.retrieval.search import search_corpus

__all__ = ["run_agent"]

EXTRA_RULE = (
    "\n7. Some excerpts are neighbouring provisions added for context; "
    "rely on them only if they bear on the question."
)


def _step(state: AgentState, **update) -> dict:
    """The node's update plus one more step on the counter."""
    return {**update, "steps": state["steps"] + 1}


@lru_cache(maxsize=1)
def _chain():
    from langchain_core.prompts import ChatPromptTemplate

    system = SYSTEM.replace("{refusal}", REFUSAL) + EXTRA_RULE
    model = chat_model(settings.generation_provider, settings.generation_model, max_tokens=1024)
    return ChatPromptTemplate.from_messages([("system", system), ("human", USER)]) | model


def _generate(context: str, question: str) -> tuple[str, int]:
    """The one Groq call: the answer text and the tokens it cost."""
    message = with_retry(lambda: _chain().invoke({"context": context, "question": question}))
    return message.content.strip(), (message.usage_metadata or {}).get("total_tokens", 0)


def plan(state: AgentState) -> dict:
    """No decomposition yet; the Planner agent replaces this node."""
    return _step(state, sub_questions=[])


def gather(state: AgentState) -> dict:
    """Search the question and each sub-question, then add the neighbours of every hit."""
    k = settings.rerank_top_k
    pools = {"main": search_corpus(state["question"], k=k)}
    for i, sub in enumerate(state["sub_questions"]):
        pools[f"sub:{i}"] = search_corpus(sub, k=k)
    seeds = [chunk for pool in pools.values() for chunk in pool]
    pools["neighbors"] = tools.expand(seeds)
    return _step(state, pools=pools, checked=list(pools))


def exclusion_check(state: AgentState) -> dict:
    """The mandatory pass for the clause that limits the others; it runs on every question."""
    found = [hit.chunk for hit in tools.exclusion_pass(state["question"])]
    return _step(state, pools={"exclusion": found}, checked=state["checked"] + ["exclusion"])


def assemble(state: AgentState) -> dict:
    """Fuse the pools into one context; the retry gets the full single-shot budget."""
    limit = MAX_CONTEXT_TOKENS if state["retries"] else settings.agent_context_tokens
    return _step(state, context=tools.assemble(state["pools"], limit=limit))


def synthesize(state: AgentState) -> dict:
    """Write the answer from the assembled excerpts."""
    text, used = _generate(format_context(state["context"]), state["question"])
    return _step(state, answer=text, tokens=spend(state["tokens"], used))


def retry(state: AgentState) -> dict:
    """Count the one allowed retry."""
    return _step(state, retries=state["retries"] + 1)


def refuse(state: AgentState) -> dict:
    """Replace whatever the model wrote with exactly the refusal."""
    return _step(state, answer=REFUSAL)


def route(state: AgentState) -> str:
    """Final check: end, retry once, or refuse. Deterministic, never a second LLM call."""
    answered = REFUSAL not in state["answer"] and bool(cited(state["answer"], state["context"]))
    if answered and (state["retries"] or state["pools"].get("exclusion")):
        return "end"
    return "refuse" if state["retries"] else "retry"


@lru_cache(maxsize=1)
def _graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    nodes = (plan, gather, exclusion_check, assemble, synthesize, retry, refuse)
    for node in nodes:
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    for before, after in (
        ("plan", "gather"),
        ("gather", "exclusion_check"),
        ("exclusion_check", "assemble"),
        ("assemble", "synthesize"),
        ("retry", "assemble"),
        ("refuse", END),
    ):
        graph.add_edge(before, after)
    outcomes = {"end": END, "retry": "retry", "refuse": "refuse"}
    graph.add_conditional_edges("synthesize", route, outcomes)
    return graph.compile()


def run_agent(question: str) -> AgentAnswer:
    """Run the graph for one question; the step cap and token budget are enforced inside."""
    start: AgentState = {
        "question": question, "sub_questions": [], "pools": {}, "checked": [],
        "context": [], "answer": "", "retries": 0, "steps": 0, "tokens": 0,
    }
    limit = settings.max_agent_steps + 1  # LangGraph counts the start as a step
    final = _graph().invoke(start, {"recursion_limit": limit})
    text, context = final["answer"], final["context"]
    used = [] if text == REFUSAL else cited(text, context)
    return AgentAnswer(
        question=question,
        text=text,
        chunks=[] if text == REFUSAL else used or context,
        retrieved=context,
        checked=final["checked"],
        steps=final["steps"],
        tokens=final["tokens"],
        cited_nothing=text != REFUSAL and not used,
    )
