"""PR9 demo: DeltaChannel replay through the get_delta_channel_history override."""

import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import MongoDBSaver
from tests.integration_tests._demo import demo_db


def concat(state: Any, writes: Sequence[list]) -> list:
    return list(state or []) + [x for w in writes for x in w]


class S(TypedDict, total=False):
    log: Annotated[list, DeltaChannel(concat, snapshot_frequency=1000)]
    n: Annotated[int, operator.add]


g = StateGraph(S)
g.add_node("step", lambda state: {"log": [f"s{state.get('n', 0)}"], "n": 1})
g.add_edge(START, "step")
g.add_conditional_edges("step", lambda state: "step" if state["n"] < 200 else END)
cfg: Any = {"configurable": {"thread_id": "d"}, "recursion_limit": 500}
with demo_db() as (client, db):
    # durability="sync": LangGraph 1.2.11 can deadlock on long DeltaChannel
    # runs with the default async durability (reproducible with InMemorySaver)
    g.compile(checkpointer=MongoDBSaver(client, db)).invoke(
        {"log": [], "n": 0}, cfg, durability="sync"
    )
    saver = MongoDBSaver(client, db)
    tup = saver.get_tuple(cfg)
    assert tup is not None
    hist = saver.get_delta_channel_history(config=tup.config, channels=["log"])
    print(
        "writes replayed:", len(hist["log"]["writes"]), "seed:", hist["log"].get("seed")
    )
    print(
        "state:",
        len(g.compile(checkpointer=saver).get_state(cfg).values["log"]),
        "entries",
    )
