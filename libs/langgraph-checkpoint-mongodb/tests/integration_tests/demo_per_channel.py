"""PR5 demo: >16 MB of state, only changed channels rewritten, cold resume."""

from typing import Any, TypedDict

import bson
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import MongoDBSaver
from tests.integration_tests._demo import demo_db


class S(TypedDict, total=False):
    history: str
    tool_result: str
    step: int


g = StateGraph(S)
g.add_node("history", lambda state: {"history": "h" * 6_000_000, "step": 1})
g.add_node("tool", lambda state: {"tool_result": "t" * 12_000_000, "step": 2})
g.add_node("tick", lambda state: {"step": state["step"] + 1})
g.add_edge(START, "history")
g.add_edge("history", "tool")
g.add_edge("tool", "tick")
g.add_edge("tick", END)
cfg: Any = {"configurable": {"thread_id": "large"}}
with demo_db() as (client, db):
    g.compile(checkpointer=MongoDBSaver(client, db), interrupt_before=["tick"]).invoke(
        {}, cfg
    )
    out = g.compile(checkpointer=MongoDBSaver(client, db)).invoke(
        None, cfg
    )  # fresh saver
    print(
        "resumed:", {k: (len(v) if isinstance(v, str) else v) for k, v in out.items()}
    )
    for d in client[db]["checkpoint_blobs"].find(
        {}, {"channel": 1, "version": 1, "value": 1}
    ):
        print(
            f"blob {d['channel']:12} v={d['version'][:6]}..{d['version'][-4:]} {len(d['value']):>10} bytes"
        )
    for d in client[db]["checkpoints"].find().sort("checkpoint_id", 1):
        print(
            f"manifest {len(bson.encode(d)):>6} bytes  refs={[r['channel'] for r in d['blob_refs']]} inline={d['inline_channels']}"
        )
