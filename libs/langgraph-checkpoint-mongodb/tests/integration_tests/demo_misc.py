"""V1 + PR3 + PR4 demo: pre-v4 sends kept, unique indexes, write ordering."""

from typing import Any

from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.mongodb import MongoDBSaver
from tests.integration_tests._demo import demo_db

with demo_db() as (client, db):
    client[db]["checkpoints"].create_index(
        [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        name="legacy_idx",
    )
    saver = MongoDBSaver(client, db)
    print(
        "setup_errors:",
        [(e.code, e.collection, e.index_name) for e in saver.setup_errors],
    )
    for name in ("checkpoint_writes", "checkpoint_blobs"):
        print(
            name,
            [(i["name"], i.get("unique")) for i in client[db][name].list_indexes()],
        )
    cfg: Any = saver.put(
        {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {},
        {},
    )
    saver.put_writes(cfg, [("ch", "from ~b")], "task-1", "~b")
    saver.put_writes(cfg, [("ch", "from ~a")], "task-2", "~a")
    tup = saver.get_tuple(cfg)
    assert tup is not None
    print("pending writes (task_path order):", [w[2] for w in tup.pending_writes or []])
