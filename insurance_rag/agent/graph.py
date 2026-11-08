"""The fixed agent graph: guard, plan, gather, assemble, answer, final check, guard."""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.agent import policy, tools, web
from insurance_rag.agent.budget import spend
from insurance_rag.agent.decompose import decompose
from insurance_rag.agent.state import AgentAnswer, AgentState
from insurance_rag.agent.web_summary import summarize
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
from insurance_rag.schema import Chunk

__all__ = ["run_agent", "retrieve"]

GUARD_NODES = 2  # guard_input and guard_output sit outside the agent's step cap


def _step(state: AgentState, **update) -> dict:
    """The node's update plus one more step on the counter."""
    return {**update, "steps": state["steps"] + 1}


STYLE = """
7. Write for a reader with no legal training, in everyday words. Open with a one-sentence answer.
   Add at most two short sentences for a condition or exception that changes the answer. Add
   nothing the question did not ask. Explain a defined term only if the answer uses it. Keep
   every citation."""


@lru_cache(maxsize=2)
def _chain(plain: bool):
    from langchain_core.prompts import ChatPromptTemplate

    system = (SYSTEM + (STYLE if plain else "")).replace("{refusal}", REFUSAL)
    model = chat_model(settings.generation_provider, settings.generation_model, max_tokens=1024)
    return ChatPromptTemplate.from_messages([("system", system), ("human", USER)]) | model


def _generate(context: str, question: str) -> tuple[str, int]:
    """The one Groq call; the plain-language rule rides with the guardrails (eval rows omit it)."""
    chain = _chain(settings.enable_guardrails)
    message = with_retry(lambda: chain.invoke({"context": context, "question": question}))
    return message.content.strip(), (message.usage_metadata or {}).get("total_tokens", 0)


def guard_input(state: AgentState) -> dict:
    """Redact personal data and stop questions the product must not answer; no model call."""
    if not settings.enable_guardrails:
        return {}
    intake = policy.check_input(state["question"])
    update: dict = {"question": intake.question, "guard": intake.notes}
    if intake.blocked:
        update |= {"answer": intake.answer, "blocked": intake.blocked}
    return update


def after_guard_input(state: AgentState) -> str:
    """A stopped question skips the agent and goes straight to the output guard."""
    return "guard_output" if state["blocked"] else "plan"


def plan(state: AgentState) -> dict:
    """The Planner agent splits the question; `agent_decompose=false` skips it."""
    subs = decompose(state["question"]) if settings.agent_decompose else []
    return _step(state, sub_questions=subs)


def gather(state: AgentState) -> dict:
    """Search the question and each sub-question."""
    k = settings.rerank_top_k
    pools = {"main": search_corpus(state["question"], k=k)}
    for i, sub in enumerate(state["sub_questions"]):
        pools[f"sub:{i}"] = search_corpus(sub, k=k)
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
    spend(used)
    return _step(state, answer=text, tokens=state["tokens"] + used)


def retry(state: AgentState) -> dict:
    """Count the one allowed retry."""
    return _step(state, retries=state["retries"] + 1)


def refuse(state: AgentState) -> dict:
    """Replace whatever the model wrote with exactly the refusal."""
    return _step(state, answer=REFUSAL)


def web_supplement(state: AgentState) -> dict:
    """Public web results, and optionally a plain summary, shown beneath the refusal only."""
    results = web.web_supplement(state["question"])
    summary = summarize(state["question"], results) if settings.web_summary else ""
    return _step(state, web=results, web_summary=summary)


def guard_output(state: AgentState) -> dict:
    """Screen the answer and summary and attach the disclaimer; a flagged answer is replaced."""
    if not settings.enable_guardrails:
        return {}
    excerpts = " ".join(chunk.text for chunk in state["context"])
    retried = ["retried"] if state["retries"] and state["answer"] != REFUSAL else []
    release = policy.check_output(
        state["answer"], state["web_summary"], excerpts, state["blocked"], state["guard"] + retried
    )
    return {
        "answer": release.answer, "web_summary": release.summary, "blocked": release.blocked,
        "guard": release.notes, "disclaimer": release.disclaimer,
    }


def after_refuse(state: AgentState) -> str:
    """The web tier runs after a refusal and only when the flag is on."""
    return "web" if settings.enable_web_fallback else "end"


def route(state: AgentState) -> str:
    """Final check: end, retry once, or refuse. Deterministic, never a second LLM call."""
    answered = REFUSAL not in state["answer"] and bool(cited(state["answer"], state["context"]))
    if answered and (state["retries"] or state["pools"].get("exclusion")):
        return "end"
    return "refuse" if state["retries"] else "retry"


@lru_cache(maxsize=2)
def _graph(retrieval_only: bool = False):
    """The graph; `retrieval_only` stops after the assembler, so nothing calls a model."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    for node in (plan, gather, exclusion_check, assemble):
        graph.add_node(node.__name__, node)
    for before, after in (("plan", "gather"), ("gather", "exclusion_check"),
                          ("exclusion_check", "assemble")):
        graph.add_edge(before, after)
    if retrieval_only:
        graph.add_edge(START, "plan")
        graph.add_edge("assemble", END)
        return graph.compile()

    for node in (guard_input, synthesize, retry, refuse, web_supplement, guard_output):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "guard_input")
    for before, after in (("assemble", "synthesize"), ("retry", "assemble"),
                          ("web_supplement", "guard_output"), ("guard_output", END)):
        graph.add_edge(before, after)
    graph.add_conditional_edges(
        "guard_input", after_guard_input, {"plan": "plan", "guard_output": "guard_output"}
    )
    outcomes = {"end": "guard_output", "retry": "retry", "refuse": "refuse"}
    graph.add_conditional_edges("synthesize", route, outcomes)
    graph.add_conditional_edges(
        "refuse", after_refuse, {"web": "web_supplement", "end": "guard_output"}
    )
    return graph.compile()


def _run(question: str, *, retrieval_only: bool = False) -> AgentState:
    start: AgentState = {
        "question": question, "sub_questions": [], "pools": {}, "checked": [],
        "context": [], "answer": "", "retries": 0, "steps": 0, "tokens": 0,
        "web": [], "web_summary": "", "blocked": "", "guard": [], "disclaimer": "",
    }
    limit = settings.max_agent_steps + 1 + GUARD_NODES  # LangGraph counts the start as a step
    return _graph(retrieval_only).invoke(start, {"recursion_limit": limit})


def retrieve(question: str) -> list[Chunk]:
    """The context the agent assembles before it writes anything: the free retrieval row."""
    return _run(question, retrieval_only=True)["context"]


def run_agent(question: str) -> AgentAnswer:
    """Run the graph for one question; the step cap and token budget are enforced inside."""
    final = _run(question)
    text, context = final["answer"], final["context"]
    declined = text == REFUSAL or bool(final["blocked"])
    used = [] if declined else cited(text, context)
    return AgentAnswer(
        question=final["question"],
        text=text,
        chunks=[] if declined else used or context,
        retrieved=context,
        checked=final["checked"],
        steps=final["steps"],
        tokens=final["tokens"],
        cited_nothing=not declined and not used,
        web=final["web"],
        web_summary=final["web_summary"],
        blocked=final["blocked"],
        guard=final["guard"],
        disclaimer=final["disclaimer"],
    )
