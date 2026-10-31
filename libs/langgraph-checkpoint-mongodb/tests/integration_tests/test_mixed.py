"""PR5e: old-format threads written by the published 0.5.0 package next to new
threads (TC-039, TC-UPGRADE; INV-001/002)."""

from collections.abc import Callable
from typing import Any

from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

from .graphs import MIXED_INPUT, mixed_graph
from .legacy import legacy_write


def test_tc039_tc_upgrade_old_and_new_threads(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    report = legacy_write(
        "mixed_graph",
        db_name,
        "old",
        input_name="MIXED_INPUT",
        interrupt_before=("review",),
    )
    assert report["next"] == ["review"]
    assert all(isinstance(v, int) for v in report["versions"].values())
    coll = client[db_name]["checkpoints"]
    old_before = {d["checkpoint_id"]: d for d in coll.find({"thread_id": "old"})}
    assert old_before and not any("layout" in d for d in old_before.values())

    old_cfg: Any = {"configurable": {"thread_id": "old"}}
    new_cfg: Any = {"configurable": {"thread_id": "new"}}
    mixed_graph().compile(
        checkpointer=make_saver(), interrupt_before=["review"]
    ).invoke(MIXED_INPUT, new_cfg)

    # TC-UPGRADE: resuming the integer-version thread on the new code works
    # (no str/int TypeError) and both threads end in the same state.
    out_old = mixed_graph().compile(checkpointer=make_saver()).invoke(None, old_cfg)
    out_new = mixed_graph().compile(checkpointer=make_saver()).invoke(None, new_cfg)
    assert out_old == out_new
    assert out_old["log"] == ["agent1", "agent2", "agent3", "review", "finish"]

    # INV-001: documents written by 0.5.0 are untouched and read as before
    for doc in old_before.values():
        assert coll.find_one({"_id": doc["_id"]}) == doc
    # INV-002 / REQ-E-039: the old thread kept integer versions + old format
    for doc in coll.find({"thread_id": "old"}):
        assert "layout" not in doc and "blob_refs" not in doc
    old_tuple = make_saver().get_tuple(old_cfg)
    assert old_tuple is not None
    assert all(
        isinstance(v, int) for v in old_tuple.checkpoint["channel_versions"].values()
    )
    assert (
        client[db_name]["checkpoint_blobs"].count_documents({"thread_id": "old"}) == 0
    )
    # the new thread uses layout 2 with string versions
    assert all(d.get("layout") == 2 for d in coll.find({"thread_id": "new"}))
    new_tuple = make_saver().get_tuple(new_cfg)
    assert new_tuple is not None
    assert all(
        isinstance(v, str) for v in new_tuple.checkpoint["channel_versions"].values()
    )
    # full history of both threads reads back
    g = mixed_graph().compile(checkpointer=make_saver())
    assert len(list(g.get_state_history(old_cfg))) == len(
        list(g.get_state_history(new_cfg))
    )
