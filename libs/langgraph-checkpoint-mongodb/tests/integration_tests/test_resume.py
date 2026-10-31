"""PR5d: real graphs, real server, fresh saver for every read/resume.

TC-035, TC-FORK, TC-SUBGRAPH, TC-BIG.
"""

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

MB = 1_000_000


class AB(TypedDict, total=False):
    a: dict
    b: list


def test_tc035_two_node_graph_cold_read(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    g = StateGraph(AB)
    g.add_node("setA", lambda s: {"a": {"k": "A" * 2000}})
    g.add_node("setB", lambda s: {"b": ["B", 1, None]})
    g.add_edge(START, "setA")
    g.add_edge("setA", "setB")
    g.add_edge("setB", END)
    cfg: Any = {"configurable": {"thread_id": "tc035"}}
    g.compile(checkpointer=make_saver()).invoke({}, cfg)

    state = g.compile(checkpointer=make_saver()).get_state(cfg)
    assert state.values == {"a": {"k": "A" * 2000}, "b": ["B", 1, None]}
    latest = client[db_name]["checkpoints"].find_one(
        {"thread_id": "tc035"}, sort=[("checkpoint_id", -1)]
    )
    assert latest is not None and latest["layout"] == 2
    # every historical state also reads back
    history = list(g.compile(checkpointer=make_saver()).get_state_history(cfg))
    assert len(history) == 4
    assert history[-1].values == {}


class Fork(TypedDict, total=False):
    x: dict
    y: list


def test_tc_fork_branches_read_their_own_value(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    g = StateGraph(Fork)
    g.add_node("one", lambda s: {"x": {"side": "BASE"}, "y": [1]})
    g.add_node("two", lambda s: {"y": s["y"] + [2]})
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", END)
    cfg: Any = {"configurable": {"thread_id": "fork"}}
    g.compile(checkpointer=make_saver()).invoke({}, cfg)

    graph = g.compile(checkpointer=make_saver())
    ancestor = next(
        s for s in graph.get_state_history(cfg) if s.next == ("two",)
    ).config
    left = graph.update_state(ancestor, {"x": {"side": "LEFT"}})
    right = g.compile(checkpointer=make_saver()).update_state(
        ancestor, {"x": {"side": "RIGHT"}}
    )

    fresh = g.compile(checkpointer=make_saver())
    assert fresh.get_state(left).values["x"] == {"side": "LEFT"}
    assert fresh.get_state(right).values["x"] == {"side": "RIGHT"}
    # both branches can continue independently
    assert fresh.invoke(None, left)["x"] == {"side": "LEFT"}
    assert g.compile(checkpointer=make_saver()).get_state(right).values["x"] == {
        "side": "RIGHT"
    }
    # same version prefix, distinct versions -> distinct blobs (no collision)
    xs = list(
        client[db_name]["checkpoint_blobs"].find({"thread_id": "fork", "channel": "x"})
    )
    assert len({b["version"] for b in xs}) == len(xs) == 3


class Inner(TypedDict, total=False):
    inner: list
    answer: str


class Outer(TypedDict, total=False):
    inner: list
    answer: str
    done: bool


def _subgraph_parent() -> StateGraph:
    def collect(state: Inner) -> dict:
        return {"inner": (state.get("inner") or []) + ["collected", "z" * 3000]}

    def ask(state: Inner) -> dict:
        return {"answer": interrupt({"question": "approve?"})}

    sub = StateGraph(Inner)
    sub.add_node("collect", collect)
    sub.add_node("ask", ask)
    sub.add_edge(START, "collect")
    sub.add_edge("collect", "ask")
    sub.add_edge("ask", END)

    parent = StateGraph(Outer)
    parent.add_node("sub", sub.compile())
    parent.add_node("finish", lambda s: {"done": True})
    parent.add_edge(START, "sub")
    parent.add_edge("sub", "finish")
    parent.add_edge("finish", END)
    return parent


def test_tc_subgraph_state_survives_resume(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    cfg: Any = {"configurable": {"thread_id": "sub"}}
    first = (
        _subgraph_parent()
        .compile(checkpointer=make_saver())
        .invoke({"inner": ["start"]}, cfg)
    )
    assert "__interrupt__" in first

    state = (
        _subgraph_parent()
        .compile(checkpointer=make_saver())
        .get_state(cfg, subgraphs=True)
    )
    sub_state = state.tasks[0].state
    assert sub_state is not None
    assert sub_state.values["inner"] == ["start", "collected", "z" * 3000]  # type: ignore[index]
    namespaces = client[db_name]["checkpoint_blobs"].distinct(
        "checkpoint_ns", {"thread_id": "sub"}
    )
    assert any(ns.startswith("sub:") for ns in namespaces)

    out = (
        _subgraph_parent()
        .compile(checkpointer=make_saver())
        .invoke(Command(resume="yes"), cfg)
    )
    assert out["answer"] == "yes"
    assert out["done"] is True
    assert out["inner"] == ["start", "collected", "z" * 3000]


class Big(TypedDict, total=False):
    six: str
    twelve: bytes
    tail: list


def test_tc_big_state_over_16mb_total(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    g = StateGraph(Big)
    g.add_node("six", lambda s: {"six": "6" * (6 * MB)})
    g.add_node("twelve", lambda s: {"twelve": b"\x0c" * (12 * MB)})
    g.add_node("tail", lambda s: {"tail": [len(s["six"]), len(s["twelve"])]})
    g.add_edge(START, "six")
    g.add_edge("six", "twelve")
    g.add_edge("twelve", "tail")
    g.add_edge("tail", END)
    cfg: Any = {"configurable": {"thread_id": "big"}}
    g.compile(checkpointer=make_saver(), interrupt_before=["tail"]).invoke({}, cfg)

    out = g.compile(checkpointer=make_saver()).invoke(None, cfg)
    assert out["tail"] == [6 * MB, 12 * MB]
    assert out["six"] == "6" * (6 * MB)
    assert out["twelve"] == b"\x0c" * (12 * MB)
