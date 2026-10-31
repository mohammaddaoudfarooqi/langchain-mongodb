"""PR1: size guard (REQ-E-001..005, INV-003/004)."""

import logging
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import (
    CheckpointSizeError,
    DocumentSizeInfo,
    MongoDBSaver,
)

MiB = 1024 * 1024


def _old_format_checkpoint(values: dict[str, Any]) -> Checkpoint:
    """A checkpoint with integer versions: written in the single-document format."""
    ck = empty_checkpoint()
    ck["channel_values"] = dict(values)
    ck["channel_versions"] = {k: 1 for k in values}
    return ck


def _new_format_checkpoint(values: dict[str, Any]) -> Checkpoint:
    ck = empty_checkpoint()
    ck["channel_values"] = dict(values)
    ck["channel_versions"] = {k: f"{1:032}.{0.5:016}" for k in values}
    return ck


def _cfg(thread: str = "t") -> Any:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}


def test_tc001_put_over_limit_raises_and_writes_nothing(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    ck = _old_format_checkpoint(
        {"big": b"x" * (17 * MiB), "small": "s", "medium": b"m" * 1000}
    )
    with pytest.raises(CheckpointSizeError) as ei:
        saver.put(_cfg(), ck, {}, {k: 1 for k in ck["channel_values"]})
    err = ei.value
    assert err.code == "CheckpointSizeError"
    assert err.collection == "checkpoints"
    assert err.bytes > err.limit
    assert err.limit == 16 * MiB - 16 * 1024
    assert err.largest_channels[0]["channel"] == "big"
    assert err.largest_channels[0]["bytes"] >= 17 * MiB
    assert len(err.largest_channels) <= 5
    assert client[db_name]["checkpoints"].count_documents({"thread_id": "t"}) == 0
    assert client[db_name]["checkpoint_blobs"].count_documents({}) == 0


def test_tc001_layout2_blob_over_limit_raises_and_writes_nothing(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    ck = _new_format_checkpoint({"big": b"x" * (17 * MiB), "other": [1, 2, 3]})
    with pytest.raises(CheckpointSizeError) as ei:
        saver.put(_cfg(), ck, {}, dict(ck["channel_versions"]))
    err = ei.value
    assert err.collection == "checkpoint_blobs"
    assert err.largest_channels[0]["channel"] == "big"
    for coll in ("checkpoints", "checkpoint_blobs"):
        assert client[db_name][coll].count_documents({}) == 0


def test_tc002_put_writes_over_limit_raises_and_writes_nothing(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    cfg = {
        "configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}
    }
    with pytest.raises(CheckpointSizeError) as ei:
        saver.put_writes(cfg, [("ok", 1), ("big", b"x" * (17 * MiB))], "task")  # type: ignore[arg-type]
    assert ei.value.collection == "checkpoint_writes"
    assert ei.value.largest_channels[0]["channel"] == "big"
    assert client[db_name]["checkpoint_writes"].count_documents({}) == 0


def test_tc003_warning_and_callback_near_limit(
    make_saver: Callable[..., MongoDBSaver], caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[DocumentSizeInfo] = []
    saver = make_saver(on_document_size=seen.append)
    ck = _old_format_checkpoint({"big": b"x" * int(13.5 * MiB), "small": 1})
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.mongodb"):
        saver.put(_cfg(), ck, {}, {"big": 1, "small": 1})
    assert any(
        "13" in r.getMessage() and "checkpoints" in r.getMessage()
        for r in caplog.records
    )
    assert len(seen) == 1
    info = seen[0]
    assert info.collection == "checkpoints"
    assert info.bytes > 0.8 * info.limit
    assert info.limit == 16 * MiB - 16 * 1024
    assert info.largest_channels[0]["channel"] == "big"
    assert saver.get_tuple(_cfg()) is not None


def test_tc004_callback_called_per_document(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    seen: list[DocumentSizeInfo] = []
    saver = make_saver(on_document_size=seen.append)
    saved = saver.put(_cfg(), _old_format_checkpoint({"a": 1}), {}, {"a": 1})
    saver.put_writes(saved, [("a", 2), ("b", b"y" * 100)], "task")
    assert [i.collection for i in seen] == [
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_writes",
    ]
    assert all(i.bytes > 0 for i in seen)
    assert seen[2].bytes > seen[1].bytes  # sizes are per document
    # below the warning ratio largest_channels is not computed
    assert seen[0].largest_channels == []


def test_custom_limit(make_saver: Callable[..., MongoDBSaver]) -> None:
    saver = make_saver(max_document_bytes=10_000)
    with pytest.raises(CheckpointSizeError) as ei:
        saver.put(_cfg(), _old_format_checkpoint({"a": b"z" * 20_000}), {}, {"a": 1})
    assert ei.value.limit == 10_000


def test_empty_writes_is_noop(
    monitored_client: MongoClient, db_name: str, recorder: Any
) -> None:
    saver = MongoDBSaver(monitored_client, db_name)
    recorder.clear()
    saver.put_writes(
        {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}},
        [],
        "task",
    )
    assert recorder.names() == []
