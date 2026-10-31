"""V1 / INV-006: pre-v4 checkpoints keep their pending sends.

Before checkpoint format v4, ``Send`` packets lived only as ``__pregel_tasks``
writes on the *parent* checkpoint. LangGraph core only migrates an
in-checkpoint ``pending_sends`` list, so the saver must surface the parent's
TASKS writes itself (mirroring the Postgres saver's ``_migrate_pending_sends``).
"""

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.constants import TASKS
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver


class State(TypedDict):
    items: list[int]
    results: Annotated[list[int], operator.add]


def _graph() -> StateGraph:
    def gen(state: State) -> dict:
        return {}

    def fan(state: State) -> list[Send]:
        return [Send("worker", i) for i in state["items"]]

    def worker(i: int) -> dict:
        return {"results": [i * 10]}

    b = StateGraph(State)
    b.add_node("gen", gen)
    b.add_node("worker", worker)  # type: ignore[arg-type]
    b.add_edge(START, "gen")
    b.add_conditional_edges("gen", fan, ["worker"])
    b.add_edge("worker", END)
    return b


def _downgrade_thread_to_v3(saver: MongoDBSaver, client: MongoClient, db: str) -> None:
    """Rewrite every checkpoint of thread "t" as a pre-v4 checkpoint."""
    coll = client[db]["checkpoints"]
    for d in coll.find({"thread_id": "t"}):
        ck: Any = saver.serde.loads_typed((d["type"], d["checkpoint"]))
        ck["v"] = 3
        ck["channel_values"].pop(TASKS, None)
        ck["channel_versions"].pop(TASKS, None)
        t, data = saver.serde.dumps_typed(ck)
        update: dict[str, Any] = {"$set": {"type": t, "checkpoint": data}}
        if d.get("layout") == 2:  # per-channel manifest: also drop the TASKS ref
            update["$pull"] = {
                "blob_refs": {"channel": TASKS},
                "inline_channels": TASKS,
            }
        coll.update_one({"_id": d["_id"]}, update)


def test_pre_v4_pending_sends_are_scheduled(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    cfg: Any = {"configurable": {"thread_id": "t"}}
    saver = make_saver()
    g = _graph().compile(checkpointer=saver, interrupt_before=["worker"])
    g.invoke({"items": [1, 2, 3], "results": []}, cfg)

    _downgrade_thread_to_v3(saver, client, db_name)

    fresh = make_saver()
    g2 = _graph().compile(checkpointer=fresh, interrupt_before=["worker"])
    assert g2.get_state(cfg).next == ("worker", "worker", "worker")
    out = g2.invoke(None, cfg)
    assert sorted(out["results"]) == [10, 20, 30]


def test_v4_checkpoint_is_not_touched(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    cfg: Any = {"configurable": {"thread_id": "t"}}
    g = _graph().compile(checkpointer=make_saver(), interrupt_before=["worker"])
    g.invoke({"items": [4], "results": []}, cfg)
    g2 = _graph().compile(checkpointer=make_saver(), interrupt_before=["worker"])
    assert g2.get_state(cfg).next == ("worker",)
    assert g2.invoke(None, cfg)["results"] == [40]
