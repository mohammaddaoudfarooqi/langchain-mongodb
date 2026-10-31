"""PR3: unique indexes (REQ-E-020/021, INV-010)."""

import logging
import threading
from collections.abc import Callable
from typing import Any

import pytest
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import IndexMigrationRequired, MongoDBSaver

CHECKPOINT_KEYS = [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)]
WRITES_KEYS = [
    ("thread_id", 1),
    ("checkpoint_ns", 1),
    ("checkpoint_id", -1),
    ("task_id", 1),
    ("idx", 1),
]
BLOB_KEYS = [("thread_id", 1), ("checkpoint_ns", 1), ("channel", 1), ("version", 1)]


def _indexes(client: MongoClient, db: str, coll: str) -> dict[tuple, dict[str, Any]]:
    return {
        tuple(ix["key"].items()): dict(ix) for ix in client[db][coll].list_indexes()
    }


def test_tc020_fresh_database_gets_unique_indexes(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    assert saver.setup_errors == []
    for coll, keys in (
        ("checkpoints", CHECKPOINT_KEYS),
        ("checkpoint_writes", WRITES_KEYS),
        ("checkpoint_blobs", BLOB_KEYS),
    ):
        ix = _indexes(client, db_name, coll)
        assert tuple(keys) in ix, coll
        assert ix[tuple(keys)].get("unique") is True, coll
    # retention is off: the threads collection is never touched
    assert "threads" not in client[db_name].list_collection_names()


def test_tc021_non_unique_index_is_reported_not_replaced(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client[db_name]["checkpoints"].create_index(CHECKPOINT_KEYS, name="legacy_idx")
    before = _indexes(client, db_name, "checkpoints")
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.mongodb"):
        saver = make_saver()
    assert len(saver.setup_errors) == 1
    err = saver.setup_errors[0]
    assert isinstance(err, IndexMigrationRequired)
    assert err.code == "IndexMigrationRequired"
    assert err.collection == "checkpoints"
    assert err.index_name == "legacy_idx"
    assert err.keys == CHECKPOINT_KEYS
    assert any(
        "IndexMigrationRequired" in r.getMessage() or "legacy_idx" in r.getMessage()
        for r in caplog.records
    )
    assert _indexes(client, db_name, "checkpoints") == before
    # the saver still works
    assert saver.get_tuple({"configurable": {"thread_id": "x"}}) is None


def test_setup_twice_concurrently_has_no_errors(
    db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    errors: list[BaseException] = []
    savers: list[MongoDBSaver] = []

    def run() -> None:
        try:
            savers.append(make_saver())
        except BaseException as e:  # pragma: no cover - asserted below
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(s.setup_errors == [] for s in savers)
    make_saver()  # and again, sequentially
