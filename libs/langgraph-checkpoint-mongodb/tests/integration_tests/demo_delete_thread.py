"""PR8 demo: delete_thread removes every namespace from all collections."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import MongoDBSaver, RetentionConfig
from tests.integration_tests._demo import demo_db


class Inner(TypedDict, total=False):
    data: list


sub = StateGraph(Inner)
sub.add_node("inner", lambda state: {"data": ["sub"] * 600})
sub.add_edge(START, "inner")
sub.add_edge("inner", END)
outer = StateGraph(Inner)
outer.add_node("sub", sub.compile(checkpointer=True))
outer.add_edge(START, "sub")
outer.add_edge("sub", END)
colls = ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "threads")
with demo_db() as (client, db):
    saver = MongoDBSaver(client, db, retention=RetentionConfig())
    cfg: Any = {"configurable": {"thread_id": "t"}}
    outer.compile(checkpointer=saver).invoke({"data": []}, cfg)
    print(
        "before:", {c: client[db][c].count_documents({"thread_id": "t"}) for c in colls}
    )
    print("namespaces:", client[db]["checkpoints"].distinct("checkpoint_ns"))
    saver.delete_thread("t")
    print(
        "after: ", {c: client[db][c].count_documents({"thread_id": "t"}) for c in colls}
    )
    print("transaction used:", saver._supports_transactions)
