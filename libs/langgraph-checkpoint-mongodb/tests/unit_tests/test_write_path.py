"""PR5f: write path per server version (TC-041a/b, REQ-NF-001)."""

from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

WRITE_COMMANDS = {"insert", "update", "delete", "bulkWrite", "findAndModify"}


def _put(saver: MongoDBSaver) -> Any:
    v = saver.get_next_version(None, None)
    ck = empty_checkpoint()
    ck["channel_values"] = {"a": [1], "b": {"x": 2}, "c": 3}
    ck["channel_versions"] = {"a": v, "b": v, "c": v}
    return saver.put(
        {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
        ck,
        {},
        dict(ck["channel_versions"]),
    ), ck


def _write_names(recorder: Any) -> list[str]:
    return [n for n in recorder.names() if n in WRITE_COMMANDS]


def test_tc041a_pre_8_uses_collection_bulk_then_update(
    monitored_client: MongoClient, db_name: str, recorder: Any
) -> None:
    saver = MongoDBSaver(monitored_client, db_name)
    saver._use_client_bulk_write = False  # also exercised on 8.0+ servers
    recorder.clear()
    saved, ck = _put(saver)
    events = [e for e in recorder.events if e.command_name in WRITE_COMMANDS]
    assert [e.command_name for e in events] == ["update", "update"]
    assert events[0].command["update"] == "checkpoint_blobs"
    assert len(events[0].command["updates"]) == 2
    assert events[0].command.get("ordered", True) is True
    assert events[1].command["update"] == "checkpoints"
    recorder.clear()
    tup = saver.get_tuple(saved)
    assert tup is not None and tup.checkpoint == ck
    # manifest, then blobs + writes in parallel
    data_cmds = [n for n in recorder.names() if n not in ("saslStart", "saslContinue")]
    assert sorted(data_cmds) == ["find", "find", "find"]


def test_tc041b_8_0_uses_one_client_bulk_write(
    monitored_client: MongoClient, db_name: str, recorder: Any, max_wire_version: int
) -> None:
    if max_wire_version < 25:
        pytest.skip("client-level bulkWrite needs MongoDB 8.0+")
    saver = MongoDBSaver(monitored_client, db_name)
    assert saver._use_client_bulk_write is True
    recorder.clear()
    saved, ck = _put(saver)
    events = [e for e in recorder.events if e.command_name in WRITE_COMMANDS]
    assert [e.command_name for e in events] == ["bulkWrite"]
    cmd = events[0].command
    assert cmd["ordered"] is True
    assert [n["ns"] for n in cmd["nsInfo"]] == [
        f"{db_name}.checkpoint_blobs",
        f"{db_name}.checkpoints",
    ]
    assert len(cmd["ops"]) == 3
    tup = MongoDBSaver(monitored_client, db_name).get_tuple(saved)
    assert tup is not None and tup.checkpoint == ck


def test_wire_version_detection(
    client: MongoClient, db_name: str, max_wire_version: int
) -> None:
    saver = MongoDBSaver(client, db_name)
    assert saver._use_client_bulk_write is (max_wire_version >= 25)
