"""PR5g / TC-SMOKE-001: a 20-step agent loop with tool calls, an interrupt and
a resume (fresh saver) against a real server."""

from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver


@tool
def lookup(n: int) -> str:
    """Return a payload for item n."""
    return f"result-{n}:" + "r" * 3000


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: str


def _graph() -> StateGraph:
    def agent(state: AgentState) -> dict:
        calls = sum(isinstance(m, ToolMessage) for m in state["messages"])
        if calls < 9:  # 9 tool round trips = 18 agent/tool steps
            return {
                "messages": [
                    AIMessage(
                        "",
                        id=f"ai-{calls}",
                        tool_calls=[
                            {
                                "name": "lookup",
                                "args": {"n": calls},
                                "id": f"call-{calls}",
                            }
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage("all done", id="ai-final")]}

    def approve(state: AgentState) -> dict:
        if len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 5:
            return {"approved": interrupt("approve halfway?")}
        return {}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([lookup]))
    g.add_node("approve", approve)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_edge("tools", "approve")
    g.add_edge("approve", "agent")
    return g


def test_tc_smoke_001_agent_loop(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    cfg: Any = {"configurable": {"thread_id": "smoke"}, "recursion_limit": 100}
    first = (
        _graph()
        .compile(checkpointer=make_saver())
        .invoke({"messages": [HumanMessage("start", id="h0")]}, cfg)
    )
    assert first["__interrupt__"][0].value == "approve halfway?"

    final = (
        _graph().compile(checkpointer=make_saver()).invoke(Command(resume="yes"), cfg)
    )
    tool_msgs = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert [str(m.content).split(":")[0] for m in tool_msgs] == [
        f"result-{i}" for i in range(9)
    ]
    assert final["messages"][-1].content == "all done"
    assert final["approved"] == "yes"
    state = _graph().compile(checkpointer=make_saver()).get_state(cfg)
    assert state.next == ()
    assert state.values == final
    steps = client[db_name]["checkpoints"].count_documents({"thread_id": "smoke"})
    assert steps >= 20
    assert all(
        d.get("layout") == 2
        for d in client[db_name]["checkpoints"].find({"thread_id": "smoke"})
    )
