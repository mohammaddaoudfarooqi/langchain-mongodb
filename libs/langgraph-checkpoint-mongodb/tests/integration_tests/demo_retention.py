"""PR7 demo: sweep_expired deletes an idle thread; late writers get ThreadExpiredError."""

from datetime import UTC, datetime, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import (
    MongoDBSaver,
    RetentionConfig,
    ThreadExpiredError,
)
from tests.integration_tests._demo import demo_db


class S(TypedDict, total=False):
    items: list


g = StateGraph(S)
g.add_node("add", lambda state: {"items": (state.get("items") or []) + ["x"]})
g.add_edge(START, "add")
g.add_edge("add", END)
with demo_db() as (client, db):
    saver = MongoDBSaver(client, db, retention=RetentionConfig())
    graph = g.compile(checkpointer=saver)
    for t in ("idle", "active"):
        graph.invoke({}, {"configurable": {"thread_id": t}})
    old = datetime.now(tz=UTC) - timedelta(days=30)
    client[db]["threads"].update_one(
        {"thread_id": "idle"}, {"$set": {"last_active": old}}
    )
    print("sweep:", saver.sweep_expired(older_than=timedelta(days=7)))
    print(
        "threads:",
        [
            (d["thread_id"], d.get("tombstone", False))
            for d in client[db]["threads"].find()
        ],
    )
    try:
        graph.invoke({}, {"configurable": {"thread_id": "idle"}})
    except ThreadExpiredError as e:
        print("late write:", e.code, e.thread_id)
    print(
        "active thread intact:",
        graph.get_state({"configurable": {"thread_id": "active"}}).values,
    )
