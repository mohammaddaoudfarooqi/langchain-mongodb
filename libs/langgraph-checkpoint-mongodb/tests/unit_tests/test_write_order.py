"""PR4 / TC-040: pending writes come back ordered by (task_path, task_id, idx)."""

from collections.abc import Callable
from typing import Any

from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver


def test_tc040_pending_writes_sorted(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    cfg: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    from langgraph.checkpoint.base import empty_checkpoint

    saved = saver.put(cfg, empty_checkpoint(), {}, {})
    ckpt_id = saved["configurable"]["checkpoint_id"]
    expected = [
        ("", "task-z", 0),
        ("~a", "task-b", 0),
        ("~a", "task-b", 1),
        ("~a", "task-c", -1),  # negative special idx sorts by value within task
        ("~a", "task-c", 0),
        ("~b", "task-a", 0),
    ]
    docs = []
    for path, task, idx in reversed(expected):
        t, v = saver.serde.dumps_typed(f"{path}|{task}|{idx}")
        docs.append(
            {
                "thread_id": "t",
                "checkpoint_ns": "",
                "checkpoint_id": ckpt_id,
                "task_id": task,
                "task_path": path,
                "idx": idx,
                "channel": "ch",
                "type": t,
                "value": v,
            }
        )
    client[db_name]["checkpoint_writes"].insert_many(docs)
    tup = saver.get_tuple(saved)
    assert tup is not None
    got = [w[2] for w in tup.pending_writes or []]
    assert got == [f"{p}|{t}|{i}" for p, t, i in expected]
    listed = next(iter(saver.list(saved)))
    assert [w[2] for w in listed.pending_writes or []] == got


def test_put_writes_conflict_key_matches_unique_index(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    """Same (task_id, idx) with a different task_path must not raise E11000.

    ``update_state`` on an ancestor re-writes the ancestor task's writes with
    ``task_path=""``. The unique key is (thread, ns, checkpoint, task_id, idx),
    so the upsert filter must be exactly that key (first write wins, as in
    the Postgres saver's ON CONFLICT DO NOTHING).
    """
    saver = make_saver()
    cfg: Any = {
        "configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "c1"}
    }
    saver.put_writes(cfg, [("a", 1), ("b", 2)], "task", "~__pregel_pull, step")
    saver.put_writes(cfg, [("a", 10)], "task", "")
    docs = list(
        client[db_name]["checkpoint_writes"].find(
            {}, {"_id": 0, "channel": 1, "task_path": 1, "idx": 1}
        )
    )
    assert sorted((d["idx"], d["channel"], d["task_path"]) for d in docs) == [
        (0, "a", "~__pregel_pull, step"),
        (1, "b", "~__pregel_pull, step"),
    ]
    # special channels ($set) still replace, task_path included
    saver.put_writes(cfg, [("__error__", "boom")], "task", "~p")
    saver.put_writes(cfg, [("__error__", "boom2")], "task", "~q")
    err = client[db_name]["checkpoint_writes"].find_one({"channel": "__error__"})
    assert err is not None and err["task_path"] == "~q"
    assert saver.serde.loads_typed((err["type"], err["value"])) == "boom2"
