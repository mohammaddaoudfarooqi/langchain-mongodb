"""PR1 demo: a >16 MiB channel raises CheckpointSizeError; nothing is written."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.mongodb import CheckpointSizeError, MongoDBSaver
from tests.integration_tests._demo import demo_db


class S(TypedDict, total=False):
    blob: bytes


g = StateGraph(S)
g.add_node("big", lambda state: {"blob": b"x" * (17 * 1024 * 1024)})
g.add_edge(START, "big")
g.add_edge("big", END)
with demo_db() as (client, db):
    seen: list[Any] = []
    graph = g.compile(
        checkpointer=MongoDBSaver(client, db, on_document_size=seen.append)
    )
    try:
        graph.invoke({}, {"configurable": {"thread_id": "t"}}, durability="sync")
    except CheckpointSizeError as e:
        print(f"{e.code}: collection={e.collection} bytes={e.bytes} limit={e.limit}")
        print("largest channels:", e.largest_channels)
    print(
        "documents with the oversized value:",
        client[db]["checkpoint_writes"].count_documents({"channel": "blob"}),
    )
    print(
        "on_document_size calls (documents written):",
        [(i.collection, i.bytes) for i in seen],
    )
