"""PR5: per-channel storage against a real server (TC-030, 034, 036, NF-002)."""

from collections.abc import Callable
from typing import Any

import bson
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import Checkpoint, create_checkpoint, empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import CheckpointIntegrityError, MongoDBSaver, _core

MB = 1_000_000


def _cfg(thread: str = "t", ns: str = "", ckpt: str | None = None) -> Any:
    c: dict[str, Any] = {"thread_id": thread, "checkpoint_ns": ns}
    if ckpt:
        c["checkpoint_id"] = ckpt
    return {"configurable": c}


def _step(
    saver: MongoDBSaver,
    parent_cfg: Any,
    prev: Checkpoint | None,
    values: dict[str, Any],
    changed: list[str],
) -> tuple[Any, Checkpoint]:
    """Emulate one LangGraph step: bump versions of ``changed`` channels."""
    ck = create_checkpoint(prev, None, 1) if prev else empty_checkpoint()
    versions = dict(prev["channel_versions"]) if prev else {}
    current_max = max(versions.values()) if versions else None
    new_versions = {}
    for ch in changed:
        versions[ch] = saver.get_next_version(current_max, None)
        new_versions[ch] = versions[ch]
    ck["channel_versions"] = versions
    ck["channel_values"] = dict(values)
    return saver.put(parent_cfg, ck, {"step": 1}, new_versions), ck


def test_tc030_manifest_and_blobs(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    values = {
        "messages": [HumanMessage("hi"), AIMessage("hello")],
        "count": 3,
        "flag": True,
        "label": "short",
        "doc": "x" * 5000,
        "nothing": None,
    }
    saved, ck = _step(saver, _cfg(), None, values, list(values))
    m = client[db_name]["checkpoints"].find_one({"checkpoint_id": ck["id"]})
    assert m is not None
    assert m["layout"] == 2
    assert {r["channel"] for r in m["blob_refs"]} == {"messages", "doc"}
    assert set(m["inline_channels"]) == {"count", "flag", "label", "nothing"}
    for r in m["blob_refs"]:
        assert r["version"] == ck["channel_versions"][r["channel"]]
    blobs = list(client[db_name]["checkpoint_blobs"].find({}, {"_id": 0}))
    assert {
        (b["thread_id"], b["checkpoint_ns"], b["channel"], b["version"]) for b in blobs
    } == {("t", "", r["channel"], r["version"]) for r in m["blob_refs"]}
    assert all(
        set(b) == {"thread_id", "checkpoint_ns", "channel", "version", "type", "value"}
        for b in blobs
    )
    # round trip through a fresh saver
    tup = make_saver().get_tuple(saved)
    assert tup is not None
    assert tup.checkpoint["channel_values"] == values
    assert tup.checkpoint == ck


class _FailingUpdates:
    """Collection proxy whose ``update_one`` raises (simulated crash)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def update_one(self, *a: Any, **k: Any) -> Any:
        raise ConnectionError("simulated crash before manifest")


@pytest.mark.parametrize("client_bulk", [False, True])
def test_tc034_crash_between_blobs_and_manifest(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    max_wire_version: int,
    client_bulk: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if client_bulk and max_wire_version < 25:
        pytest.skip("client bulkWrite needs MongoDB 8.0+")
    saver = make_saver()
    saver._use_client_bulk_write = client_bulk
    v = saver.get_next_version(None, None)
    ck = empty_checkpoint()
    ck["channel_values"] = {"a": [1, 2], "b": {"k": "v"}}
    ck["channel_versions"] = {"a": v, "b": v}
    if client_bulk:
        real = saver.client.bulk_write

        def crash(models: list[Any], **kw: Any) -> Any:
            real(models[:-1], **kw)  # blobs land, manifest does not
            raise ConnectionError("simulated crash before manifest")

        monkeypatch.setattr(saver.client, "bulk_write", crash)
    else:
        monkeypatch.setattr(
            saver, "checkpoint_collection", _FailingUpdates(saver.checkpoint_collection)
        )
    with pytest.raises(ConnectionError):
        saver.put(_cfg(), ck, {}, {"a": v, "b": v})
    assert client[db_name]["checkpoints"].count_documents({}) == 0
    assert client[db_name]["checkpoint_blobs"].count_documents({}) == 2
    # retry with a healthy saver succeeds; blob inserts are idempotent
    monkeypatch.undo()
    retry = make_saver()
    saved = retry.put(_cfg(), ck, {}, {"a": v, "b": v})
    assert client[db_name]["checkpoint_blobs"].count_documents({}) == 2
    tup = retry.get_tuple(saved)
    assert tup is not None and tup.checkpoint == ck


def test_tc036_missing_blob_raises(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    saved, ck = _step(saver, _cfg(), None, {"a": [1], "b": [2]}, ["a", "b"])
    gone = client[db_name]["checkpoint_blobs"].find_one_and_delete({"channel": "b"})
    assert gone is not None
    with pytest.raises(CheckpointIntegrityError) as ei:
        saver.get_tuple(saved)
    e = ei.value
    assert (e.thread_id, e.checkpoint_ns, e.checkpoint_id) == ("t", "", ck["id"])
    assert e.missing == [{"channel": "b", "version": gone["version"]}]
    with pytest.raises(CheckpointIntegrityError):
        list(saver.list(_cfg()))


def test_unchanged_channels_are_referenced_not_rewritten(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    values = {"a": [1], "b": [2], "small": 1}
    cfg1, ck1 = _step(saver, _cfg(), None, values, ["a", "b", "small"])
    values2 = {**values, "a": [1, 1], "small": 2}
    cfg2, ck2 = _step(saver, cfg1, ck1, values2, ["a", "small"])
    assert client[db_name]["checkpoint_blobs"].count_documents({}) == 3
    t1, t2 = saver.get_tuple(cfg1), saver.get_tuple(cfg2)
    assert t1 and t1.checkpoint["channel_values"] == values
    assert t2 and t2.checkpoint["channel_values"] == values2
    assert t2.parent_config == cfg1


def test_tc_nf002_write_amplification(
    monitored_client: MongoClient, db_name: str, recorder: Any
) -> None:
    saver = MongoDBSaver(monitored_client, db_name)
    chans = [f"c{i}" for i in range(5)]
    values = {c: "v" * MB for c in chans}
    cfg, ck = _step(saver, _cfg(), None, values, chans)
    recorder.clear()
    for step in range(10):
        ch = chans[step % 5]
        values = {**values, ch: chr(65 + step) * MB}
        cfg, ck = _step(saver, cfg, ck, values, [ch])
    written = sum(
        len(bson.encode(e.command))
        for e in recorder.events
        if e.command_name in ("update", "insert", "bulkWrite")
    )
    assert 10 * MB <= written < 11 * MB, written  # vs ~50 MB for the old format
    tup = MongoDBSaver(monitored_client, db_name).get_tuple(cfg)
    assert tup is not None and tup.checkpoint["channel_values"] == values


def test_metadata_filter_on_layout2(make_saver: Callable[..., MongoDBSaver]) -> None:
    """INV-005: list(filter=...) keeps its semantics on layout 2."""
    saver = make_saver()
    v = saver.get_next_version(None, None)
    for i, src in enumerate(["input", "loop", "loop"]):
        ck = empty_checkpoint()
        ck["channel_values"] = {"m": [i]}
        ck["channel_versions"] = {"m": v}
        saver.put(_cfg(f"t{i}"), ck, {"source": src, "step": i}, {"m": v})
    got = list(saver.list(None, filter={"source": "loop"}))
    assert sorted(t.checkpoint["channel_values"]["m"][0] for t in got) == [1, 2]


def test_async_wrappers_on_layout2(make_saver: Callable[..., MongoDBSaver]) -> None:
    """TC-ASYNC-001: the run_in_executor wrappers drive the same core."""
    import asyncio

    saver = make_saver()

    async def go() -> None:
        v = saver.get_next_version(None, None)
        ck = empty_checkpoint()
        ck["channel_values"] = {"m": [HumanMessage("x")], "n": 1}
        ck["channel_versions"] = {"m": v, "n": v}
        saved = await saver.aput(_cfg(), ck, {}, {"m": v, "n": v})
        await saver.aput_writes(saved, [("m", [AIMessage("y")])], "task", "~p")
        tup = await saver.aget_tuple(saved)
        assert tup is not None and tup.checkpoint == ck
        assert tup.pending_writes == [("task", "m", [AIMessage("y")])]
        listed = [t async for t in saver.alist(_cfg())]
        assert listed[0].checkpoint == ck

    asyncio.run(go())


def test_layout_constant() -> None:
    assert _core.LAYOUT_V2 == 2
