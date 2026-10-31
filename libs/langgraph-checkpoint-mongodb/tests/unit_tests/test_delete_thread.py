import os

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

# Setup:
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
CHKPT_COLLECTION_NAME = "delete_thread_chkpts"
WRITES_COLLECTION_NAME = "delete_thread_writes"


def test_delete_thread() -> None:
    # Clear collections if they exist
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db[CHKPT_COLLECTION_NAME].delete_many({})
    db[WRITES_COLLECTION_NAME].delete_many({})
    client.close()

    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, CHKPT_COLLECTION_NAME, WRITES_COLLECTION_NAME
    ) as saver:
        # Thread 1 data
        chkpnt_1 = empty_checkpoint()
        thread_1_id = "thread-1"
        config_1 = RunnableConfig(
            configurable=dict(
                thread_id=thread_1_id, checkpoint_ns="", checkpoint_id=chkpnt_1["id"]
            )
        )
        metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"foo": "bar"},
        }

        # Thread 2 data
        chkpnt_2 = empty_checkpoint()
        thread_2_id = "thread-2"
        config_2 = RunnableConfig(
            configurable=dict(
                thread_id=thread_2_id, checkpoint_ns="", checkpoint_id=chkpnt_2["id"]
            )
        )
        metadata_2: CheckpointMetadata = {
            "source": "output",
            "step": 1,
            "writes": {"baz": "qux"},
        }

        # Save checkpoints for both threads
        saver.put(config_1, chkpnt_1, metadata_1, {})
        saver.put(config_2, chkpnt_2, metadata_2, {})

        # Add some writes
        saver.put_writes(config_1, [("channel1", "value1")], "task1")
        saver.put_writes(config_2, [("channel2", "value2")], "task2")

        # Verify we have data for both threads
        assert saver.get_tuple(config_1) is not None
        assert saver.get_tuple(config_2) is not None

        # Verify we have write data
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) > 0
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0

        # Delete thread 1
        saver.delete_thread(thread_1_id)

        # Verify thread 1 data is gone
        assert saver.get_tuple(config_1) is None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) == 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) == 0

        # Verify thread 2 data still exists
        assert saver.get_tuple(config_2) is not None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0


async def test_adelete_thread() -> None:
    # Clear collections if they exist
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db[CHKPT_COLLECTION_NAME].delete_many({})
    db[WRITES_COLLECTION_NAME].delete_many({})
    client.close()

    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, CHKPT_COLLECTION_NAME, WRITES_COLLECTION_NAME
    ) as saver:
        # Thread 1 data
        chkpnt_1 = empty_checkpoint()
        thread_1_id = "thread-1"
        config_1 = RunnableConfig(
            configurable=dict(
                thread_id=thread_1_id, checkpoint_ns="", checkpoint_id=chkpnt_1["id"]
            )
        )
        metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 1,
            "writes": {"foo": "bar"},
        }

        # Thread 2 data
        chkpnt_2 = empty_checkpoint()
        thread_2_id = "thread-2"
        config_2 = RunnableConfig(
            configurable=dict(
                thread_id=thread_2_id, checkpoint_ns="", checkpoint_id=chkpnt_2["id"]
            )
        )
        metadata_2: CheckpointMetadata = {
            "source": "output",
            "step": 1,
            "writes": {"baz": "qux"},
        }

        assert saver.checkpoint_collection.count_documents({}) == 0

        # Save checkpoints for both threads
        await saver.aput(config_1, chkpnt_1, metadata_1, {})
        await saver.aput(config_2, chkpnt_2, metadata_2, {})

        # Add some writes
        await saver.aput_writes(config_1, [("channel1", "value1")], "task1")
        await saver.aput_writes(config_2, [("channel2", "value2")], "task2")

        # Verify we have data for both threads
        assert await saver.aget_tuple(config_1) is not None
        assert await saver.aget_tuple(config_2) is not None

        # Verify we have write data
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) > 0
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0

        # Delete thread 1
        await saver.adelete_thread(thread_1_id)

        # Verify thread 1 data is gone
        assert await saver.aget_tuple(config_1) is None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_1_id}) == 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_1_id}) == 0

        # Verify thread 2 data still exists
        assert await saver.aget_tuple(config_2) is not None
        assert (
            saver.checkpoint_collection.count_documents({"thread_id": thread_2_id}) > 0
        )
        assert saver.writes_collection.count_documents({"thread_id": thread_2_id}) > 0


# ---------------------------------------------------------------- PR8

from collections.abc import Callable  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from langgraph.checkpoint.mongodb import RetentionConfig  # noqa: E402

COLLS = ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "threads")


def _populate(saver: MongoDBSaver, thread: str) -> None:
    for ns in ("", "child:1", "child:1|grand:2"):
        v = saver.get_next_version(None, None)
        ck = empty_checkpoint()
        ck["channel_values"] = {"data": [thread, ns], "n": 1}
        ck["channel_versions"] = {"data": v, "n": v}
        cfg = saver.put(
            {"configurable": {"thread_id": thread, "checkpoint_ns": ns}},
            ck,
            {},
            dict(ck["channel_versions"]),
        )
        saver.put_writes(cfg, [("data", [1]), ("n", 2)], "task")


def _counts(client: MongoClient, db: str, thread: str) -> dict[str, int]:
    return {c: client[db][c].count_documents({"thread_id": thread}) for c in COLLS}


class _FailingDelete:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def delete_many(self, *a: Any, **k: Any) -> Any:
        raise ConnectionError("injected failure after the first delete")


def test_tc070_delete_thread_all_collections_all_namespaces(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver(retention=RetentionConfig())
    _populate(saver, "gone")
    _populate(saver, "kept")
    assert all(n > 0 for n in _counts(client, db_name, "gone").values())
    saver.delete_thread("gone")
    assert _counts(client, db_name, "gone") == dict.fromkeys(COLLS, 0)
    assert all(n > 0 for n in _counts(client, db_name, "kept").values())


def test_tc070_delete_order(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = make_saver(retention=RetentionConfig())
    saver._supports_transactions = False
    _populate(saver, "t")
    order: list[str] = []
    for attr in (
        "checkpoint_collection",
        "writes_collection",
        "blobs_collection",
        "threads_collection",
    ):
        coll = getattr(saver, attr)

        class Spy:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def delete_many(self, *a: Any, **k: Any) -> Any:
                order.append(self._inner.name)
                return self._inner.delete_many(*a, **k)

        monkeypatch.setattr(saver, attr, Spy(coll))
    saver.delete_thread("t")
    assert order == list(COLLS)


def test_tc071_transaction_rolls_back_on_failure(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = make_saver(retention=RetentionConfig())
    if not saver._supports_transactions:
        pytest.skip("needs a replica set or sharded cluster")
    _populate(saver, "t")
    before = _counts(client, db_name, "t")
    monkeypatch.setattr(
        saver, "writes_collection", _FailingDelete(saver.writes_collection)
    )
    with pytest.raises(ConnectionError):
        saver.delete_thread("t")
    assert _counts(client, db_name, "t") == before  # checkpoints delete rolled back
    monkeypatch.undo()
    saver.delete_thread("t")
    assert _counts(client, db_name, "t") == dict.fromkeys(COLLS, 0)


def test_tc072_without_transactions_rerun_finishes(
    client: MongoClient,
    db_name: str,
    make_saver: Callable[..., MongoDBSaver],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = make_saver(retention=RetentionConfig())
    saver._supports_transactions = (
        False  # standalone behaviour (also native on standalone servers)
    )
    _populate(saver, "t")
    monkeypatch.setattr(
        saver, "writes_collection", _FailingDelete(saver.writes_collection)
    )
    with pytest.raises(ConnectionError):
        saver.delete_thread("t")
    partial = _counts(client, db_name, "t")
    assert partial["checkpoints"] == 0 and partial["checkpoint_writes"] > 0
    monkeypatch.undo()
    saver.delete_thread("t")
    saver.delete_thread("t")  # and again: nothing left, no error
    assert _counts(client, db_name, "t") == dict.fromkeys(COLLS, 0)


def test_delete_thread_without_retention_leaves_threads_alone(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    _populate(saver, "t")
    saver.delete_thread("t")
    assert "threads" not in client[db_name].list_collection_names()
    assert sum(_counts(client, db_name, "t").values()) == 0
