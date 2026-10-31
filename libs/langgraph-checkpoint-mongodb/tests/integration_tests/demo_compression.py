"""PR6 demo: compressed blobs (msgpack+gzip) read back by a plain saver."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import CompressionConfig, MongoDBSaver
from tests.integration_tests._demo import demo_db


class S(TypedDict, total=False):
    report: str
    note: str


g = StateGraph(S)
g.add_node(
    "write",
    lambda state: {"report": "quarterly numbers " * 200_000, "note": "x" * 5000},
)
g.add_edge(START, "write")
g.add_edge("write", END)
cfg: Any = {"configurable": {"thread_id": "c"}}
with demo_db() as (client, db):
    g.compile(
        checkpointer=MongoDBSaver(client, db, compression=CompressionConfig())
    ).invoke({}, cfg)
    for d in client[db]["checkpoint_blobs"].find():
        print(f"{d['channel']:7} type={d['type']:13} stored={len(d['value'])} bytes")
    state = g.compile(checkpointer=MongoDBSaver(client, db)).get_state(
        cfg
    )  # no compression configured
    print("plain saver reads it back:", len(state.values["report"]), "chars")
