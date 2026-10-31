"""PR7: thread-scoped retention (TC-060..064, REQ-NF-001, INV-009)."""

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import (
    CheckpointIntegrityError,
    MongoDBSaver,
    RetentionConfig,
    ThreadExpiredError,
)
from langgraph.checkpoint.mongodb import saver as saver_module

WRITE_COMMANDS = {"insert", "update", "delete", "bulkWrite", "findAndModify"}


def _cfg(thread: str) -> Any:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def _put(
    saver: MongoDBSaver, thread: str, parent: Any = None, value: Any = None
) -> Any:
    v = saver.get_next_version(None, None)
    ck = empty_checkpoint()
    ck["channel_values"] = {"data": value if value is not None else [thread], "n": 1}
    ck["channel_versions"] = {"data": v, "n": v}
    return saver.put(parent or _cfg(thread), ck, {}, dict(ck["channel_versions"]))


def _age(client: MongoClient, db: str, thread: str, hours: float) -> None:
    client[db]["threads"].update_one(
        {"thread_id": thread},
        {"$set": {"last_active": datetime.now(tz=UTC) - timedelta(hours=hours)}},
    )


def test_tc020_threads_indexes_only_with_retention(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    make_saver()
    assert "threads" not in client[db_name].list_collection_names()
    make_saver(retention=RetentionConfig(tombstone_seconds=3600))
    ix = {tuple(i["key"].items()): i for i in client[db_name]["threads"].list_indexes()}
    assert ix[(("thread_id", 1),)]["unique"] is True
    assert ix[(("tombstoned_at", 1),)]["expireAfterSeconds"] == 3600
    assert (("last_active", 1),) in ix


def test_tc060_put_and_put_writes_advance_last_active(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    off = make_saver()
    cfg = _put(off, "plain")
    off.put_writes(cfg, [("a", 1)], "task")
    assert "threads" not in client[db_name].list_collection_names()

    saver = make_saver(retention=RetentionConfig())
    cfg = _put(saver, "t1")
    rec = client[db_name]["threads"].find_one({"thread_id": "t1"})
    assert rec is not None and rec["generation"] == 0 and "tombstone" not in rec
    first = rec["last_active"]
    _age(client, db_name, "t1", 5)
    saver.put_writes(cfg, [("a", 1)], "task")
    rec2 = client[db_name]["threads"].find_one({"thread_id": "t1"})
    assert rec2 is not None and rec2["last_active"] >= first
    # empty writes stay a no-op and do not touch threads
    _age(client, db_name, "t1", 5)
    saver.put_writes(cfg, [], "task")
    rec3 = client[db_name]["threads"].find_one({"thread_id": "t1"})
    assert rec3 is not None and rec3["last_active"] < first
    # old-format (integer version) threads are tracked too
    ck = empty_checkpoint()
    ck["channel_values"] = {"x": 1}
    ck["channel_versions"] = {"x": 1}
    saver.put(_cfg("old"), ck, {}, {"x": 1})
    assert client[db_name]["threads"].count_documents({"thread_id": "old"}) == 1


def test_tc061_sweep_deletes_expired_thread_and_keeps_tombstone(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver(retention=RetentionConfig())
    for t in ("old", "fresh"):
        cfg = _put(saver, t)
        saver.put_writes(cfg, [("data", [1])], "task")
        _put(saver, t, parent=cfg, value=["second"])
    _age(client, db_name, "old", 48)
    result = saver.sweep_expired(older_than=timedelta(hours=24))
    assert result["threads"] == 1
    assert result["documents"] == 2 + 1 + 2  # manifests + writes + blobs
    for coll in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        assert client[db_name][coll].count_documents({"thread_id": "old"}) == 0
        assert client[db_name][coll].count_documents({"thread_id": "fresh"}) > 0
    rec = client[db_name]["threads"].find_one({"thread_id": "old"})
    assert rec is not None and rec["tombstone"] is True and rec["generation"] == 1
    assert isinstance(rec["tombstoned_at"], datetime)
    # a purged tombstone has no last_active (that marks the purge as complete)
    assert "last_active" not in rec and "purged_at" not in rec
    assert set(rec) == {"_id", "thread_id", "generation", "tombstone", "tombstoned_at"}
    # writes to a swept thread are refused, nothing is written
    with pytest.raises(ThreadExpiredError) as ei:
        _put(saver, "old")
    assert ei.value.thread_id == "old" and ei.value.code == "ThreadExpiredError"
    with pytest.raises(ThreadExpiredError):
        saver.put_writes(
            {
                "configurable": {
                    "thread_id": "old",
                    "checkpoint_ns": "",
                    "checkpoint_id": "x",
                }
            },
            [("a", 1)],
            "task",
        )
    assert client[db_name]["checkpoints"].count_documents({"thread_id": "old"}) == 0
    assert (
        client[db_name]["checkpoint_writes"].count_documents({"thread_id": "old"}) == 0
    )
    # a second sweep has nothing to do
    assert saver.sweep_expired(older_than=timedelta(hours=24)) == {
        "threads": 0,
        "documents": 0,
    }


def test_tc062_sweep_vs_put_concurrency(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    sweeper = make_saver(retention=RetentionConfig())
    writer = make_saver(retention=RetentionConfig())
    outcomes = {"committed": 0, "expired": 0}
    # jitter the writer by up to a few round trips so both orders happen
    t0 = time.perf_counter()
    for _ in range(5):
        client.admin.command("ping")
    rtt = (time.perf_counter() - t0) / 5
    rng = random.Random(62)
    for i in range(50):
        delay = rng.uniform(0, 6 * rtt)
        thread = f"race-{i}"
        parent = _put(writer, thread)
        _age(client, db_name, thread, 48)
        errors: list[BaseException] = []
        new_cfg: list[Any] = []
        barrier = threading.Barrier(2)

        def do_sweep(barrier: threading.Barrier = barrier) -> None:
            barrier.wait()
            sweeper.sweep_expired(older_than=timedelta(hours=1))

        def do_put(
            thread: str = thread,
            parent: Any = parent,
            delay: float = delay,
            barrier: threading.Barrier = barrier,
            new_cfg: list[Any] = new_cfg,
            errors: list[BaseException] = errors,
        ) -> None:
            barrier.wait()
            time.sleep(delay)
            try:
                new_cfg.append(_put(writer, thread, parent=parent, value=["new"]))
            except ThreadExpiredError as e:
                errors.append(e)

        ts = [threading.Thread(target=do_sweep), threading.Thread(target=do_put)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        rec = client[db_name]["threads"].find_one({"thread_id": thread})
        assert rec is not None
        if new_cfg:
            outcomes["committed"] += 1
            assert "tombstone" not in rec, f"iteration {i}: committed write but swept"
            tup = writer.get_tuple(new_cfg[0])
            assert tup is not None and tup.checkpoint["channel_values"]["data"] == [
                "new"
            ]
        else:
            outcomes["expired"] += 1
            assert len(errors) == 1 and rec.get("tombstone") is True
            assert (
                client[db_name]["checkpoints"].count_documents({"thread_id": thread})
                == 0
            )
        # never a manifest without its blobs
        for doc in client[db_name]["checkpoints"].find({"thread_id": thread}):
            try:
                writer.get_tuple(
                    {
                        "configurable": {
                            "thread_id": thread,
                            "checkpoint_ns": "",
                            "checkpoint_id": doc["checkpoint_id"],
                        }
                    }
                )
            except CheckpointIntegrityError as e:  # pragma: no cover
                raise AssertionError(f"iteration {i}: manifest without blobs") from e
    print("TC-062 outcomes:", outcomes)
    assert outcomes["committed"] + outcomes["expired"] == 50
    assert outcomes["committed"] > 0 and outcomes["expired"] > 0, outcomes


def test_tc063_interrupted_sweep_is_finished_by_rerun(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = make_saver(retention=RetentionConfig())
    for t in ("a", "b"):
        _put(saver, t)
        _age(client, db_name, t, 48)

    def killed(*a: Any, **k: Any) -> Any:
        raise RuntimeError("sweeper killed after the claim")

    monkeypatch.setattr(saver, "_purge_thread_data", killed)
    with pytest.raises(RuntimeError):
        saver.sweep_expired(older_than=timedelta(hours=1))
    assert client[db_name]["threads"].count_documents({"tombstone": True}) == 1
    # claimed but not purged: tombstone set, last_active still present
    assert (
        client[db_name]["threads"].count_documents(
            {"tombstone": True, "last_active": {"$exists": True}}
        )
        == 1
    )
    monkeypatch.undo()
    result = saver.sweep_expired(older_than=timedelta(hours=1))
    assert result == {"threads": 2, "documents": 2 * (1 + 1)}  # manifest + 1 blob each
    for coll in ("checkpoints", "checkpoint_blobs"):
        assert client[db_name][coll].count_documents({}) == 0


def test_tc064_ttl_option_logs_one_deprecation_warning(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(saver_module, "_TTL_DEPRECATION_LOGGED", False)
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.mongodb"):
        s1 = make_saver(ttl=3600)
        make_saver(ttl=3600)
    msgs = [r.getMessage() for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(msgs) == 1 and "sweep_expired" in msgs[0]
    # INV-009: created_at is still written exactly as before
    cfg = _put(s1, "ttl")
    s1.put_writes(cfg, [("a", 1)], "task")
    for coll in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        doc = client[db_name][coll].find_one({"thread_id": "ttl"})
        assert doc is not None and isinstance(doc["created_at"], datetime), coll


def test_retention_write_path_round_trips(
    monitored_client: MongoClient, db_name: str, recorder: Any, max_wire_version: int
) -> None:
    """REQ-NF-001: retention adds no sequential round trip to layout-2 put on
    8.0+. Older servers touch the thread first, then write blobs."""
    saver = MongoDBSaver(monitored_client, db_name, retention=RetentionConfig())
    recorder.clear()
    _put(saver, "rt")
    events = [e for e in recorder.events if e.command_name in WRITE_COMMANDS]
    if saver._use_client_bulk_write:
        assert [e.command_name for e in events] == ["bulkWrite"]
        assert [n["ns"].split(".", 1)[1] for n in events[0].command["nsInfo"]] == [
            "threads",
            "checkpoint_blobs",
            "checkpoints",
        ]
    else:
        assert [e.command["update"] for e in events] == [
            "threads",
            "checkpoint_blobs",
            "checkpoints",
        ]
    # the pre-8.0 path also works on newer servers
    saver._use_client_bulk_write = False
    recorder.clear()
    _put(saver, "rt2")
    events = [e for e in recorder.events if e.command_name in WRITE_COMMANDS]
    assert events[-1].command["update"] == "checkpoints" and len(events) == 3


def test_retention_on_old_format_and_client_bulk_expired(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver(retention=RetentionConfig())
    for path in (False, True):
        if path and not saver._use_client_bulk_write:
            continue
        saver._use_client_bulk_write = path
        t = f"exp-{path}"
        _put(saver, t)
        _age(client, db_name, t, 48)
        saver.sweep_expired(older_than=timedelta(hours=1))
        with pytest.raises(ThreadExpiredError):
            _put(saver, t)
        ck = empty_checkpoint()
        ck["channel_values"] = {"x": 1}
        ck["channel_versions"] = {"x": 1}
        with pytest.raises(ThreadExpiredError):
            saver.put(_cfg(t), ck, {}, {"x": 1})
        # nothing may outlive the purge: no sweep would ever remove it
        for coll in ("checkpoints", "checkpoint_blobs"):
            assert client[db_name][coll].count_documents({"thread_id": t}) == 0
