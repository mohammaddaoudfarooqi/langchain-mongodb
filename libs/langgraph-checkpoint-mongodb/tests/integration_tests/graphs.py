"""Graph factories shared by the new-code tests and the legacy (0.5.0) writer.

This module must only import LangGraph / LangChain (never the saver), so the
legacy writer can import it under the published package.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class MixedState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    counter: int
    notes: dict
    log: Annotated[list[str], operator.add]


def mixed_graph() -> StateGraph:
    """agent loops 3 times, then review (interrupt_before) and finish."""

    def agent(state: MixedState) -> dict:
        n = state.get("counter", 0) + 1
        return {
            "messages": [AIMessage(f"step {n}", id=f"ai-{n}")],
            "counter": n,
            "notes": {**state.get("notes", {}), f"k{n}": "v" * 2000},
            "log": [f"agent{n}"],
        }

    def review(state: MixedState) -> dict:
        return {"log": ["review"], "messages": [HumanMessage("ok", id="h-review")]}

    def finish(state: MixedState) -> dict:
        return {"log": ["finish"]}

    g = StateGraph(MixedState)
    g.add_node("agent", agent)
    g.add_node("review", review)
    g.add_node("finish", finish)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent", lambda s: "agent" if s["counter"] < 3 else "review"
    )
    g.add_edge("review", "finish")
    g.add_edge("finish", END)
    return g


MIXED_INPUT: dict[str, Any] = {"messages": [HumanMessage("go", id="h-0")]}


# ---------------------------------------------------------------- large-state shape

from langgraph.types import interrupt  # noqa: E402


class LargeState(TypedDict, total=False):
    variables: dict
    conversation_history: Annotated[list[AnyMessage], add_messages]
    agent_a_conversation_history: Annotated[list[AnyMessage], add_messages]
    status: str


def large_state_graph(
    history_bytes: int = 6_000_000,
    tool_bytes: int = 12_000_000,
    huge_bytes: int = 0,
    messages: int = 30,
) -> StateGraph:
    """planner -> tool -> agent_a -> [huge] -> approval (interrupt) -> finish.

    ``planner`` builds ~history_bytes of conversation history; ``tool`` puts a
    ~tool_bytes result into ``variables``; ``huge`` (only if huge_bytes) adds
    a single oversized value to ``variables``.
    """
    per = max(1, history_bytes // messages)

    def planner(state: LargeState) -> dict:
        msgs: list[AnyMessage] = [
            (HumanMessage if i % 2 == 0 else AIMessage)(
                f"{i}:" + chr(97 + i % 26) * per, id=f"hist-{i}"
            )
            for i in range(messages)
        ]
        return {"conversation_history": msgs, "status": "planned"}

    def tool(state: LargeState) -> dict:
        return {
            "variables": {
                **state.get("variables", {}),
                "tool_result": "T" * tool_bytes,
                "rows": 42,
            },
            "status": "tool_done",
        }

    def agent_a(state: LargeState) -> dict:
        return {
            "agent_a_conversation_history": [
                AIMessage("agent A summary", id="agent-a-1")
            ],
            "status": "agent_a_done",
        }

    def huge(state: LargeState) -> dict:
        return {"variables": {**state["variables"], "huge": b"\x01" * huge_bytes}}

    def approval(state: LargeState) -> dict:
        decision = interrupt(
            {
                "question": "approve?",
                "tool_bytes": len(state["variables"]["tool_result"]),
            }
        )
        return {"status": f"approved:{decision}"}

    def finish(state: LargeState) -> dict:
        return {
            "conversation_history": [AIMessage("final answer", id="final")],
            "status": "done",
        }

    g = StateGraph(LargeState)
    for name, fn in (
        ("planner", planner),
        ("tool", tool),
        ("agent_a", agent_a),
        ("approval", approval),
        ("finish", finish),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "tool")
    g.add_edge("tool", "agent_a")
    if huge_bytes:
        g.add_node("huge", huge)
        g.add_edge("agent_a", "huge")
        g.add_edge("huge", "approval")
    else:
        g.add_edge("agent_a", "approval")
    g.add_edge("approval", "finish")
    g.add_edge("finish", END)
    return g


def large_state_graph_small() -> StateGraph:
    """Same shape, small enough for the old single-document format."""
    return large_state_graph(history_bytes=200_000, tool_bytes=300_000)


LARGE_STATE_INPUT: dict[str, Any] = {"variables": {"customer": "acme"}, "status": "new"}
