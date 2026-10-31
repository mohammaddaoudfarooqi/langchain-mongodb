"""PR4 / TC-043: a DeltaChannel graph with two parallel writers replays the
same value from storage as the live run produced."""

import os
import time
import uuid
from collections.abc import Iterator, Sequence
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)


def concat(state: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    out = list(state or [])
    for w in writes:
        out.extend(w)
    return out


class State(TypedDict):
    log: Annotated[list[str], DeltaChannel(concat, snapshot_frequency=1000)]
    round: int


def build() -> StateGraph:
    def start(state: State) -> dict:
        return {"round": state.get("round", 0) + 1}

    def a(state: State) -> dict:
        time.sleep(0.15)  # finish after "b" so storage order != task order
        return {"log": [f"a{state['round']}"]}

    def b(state: State) -> dict:
        return {"log": [f"b{state['round']}"]}

    def join(state: State) -> dict:
        return {}

    g = StateGraph(State)
    g.add_node("start", start)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("join", join, defer=True)
    g.add_edge(START, "start")
    g.add_edge("start", "a")
    g.add_edge("start", "b")
    g.add_edge("a", "join")
    g.add_edge("b", "join")
    g.add_conditional_edges("join", lambda s: "start" if s["round"] < 3 else END)
    return g


@pytest.fixture
def db() -> Iterator[tuple[MongoClient, str]]:
    client: MongoClient = MongoClient(MONGODB_URI)
    name = f"ckpt_test_py_{uuid.uuid4().hex[:12]}"
    yield client, name
    client.drop_database(name)
    client.close()


def test_tc043_delta_channel_parallel_writers(db: tuple[MongoClient, str]) -> None:
    client, name = db
    cfg: Any = {"configurable": {"thread_id": "delta"}}
    live = (
        build()
        .compile(checkpointer=MongoDBSaver(client, name))
        .invoke({"log": [], "round": 0}, cfg)
    )
    assert live["log"] == ["a1", "b1", "a2", "b2", "a3", "b3"]
    # A fresh saver replays the channel from stored writes.
    fresh = build().compile(checkpointer=MongoDBSaver(client, name))
    assert fresh.get_state(cfg).values["log"] == live["log"]
