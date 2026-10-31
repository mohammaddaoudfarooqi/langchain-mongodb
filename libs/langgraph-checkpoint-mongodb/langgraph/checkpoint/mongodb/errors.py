"""Errors raised by the MongoDB checkpointer.

Each class carries a ``code`` string equal to its class name, matching the JS
saver (``@langchain/langgraph-checkpoint-mongodb``).
"""

from __future__ import annotations

from typing import Any


class MongoDBCheckpointError(Exception):
    """Base class of all MongoDB checkpointer errors."""

    code: str = "MongoDBCheckpointError"


class CheckpointSizeError(MongoDBCheckpointError):
    """A document would exceed ``max_document_bytes`` (R1). Nothing was written."""

    code = "CheckpointSizeError"

    def __init__(
        self,
        collection: str,
        bytes: int,
        limit: int,
        largest_channels: list[dict[str, Any]],
    ) -> None:
        self.collection = collection
        self.bytes = bytes
        self.limit = limit
        self.largest_channels = largest_channels
        top = ", ".join(f"{c['channel']}={c['bytes']}" for c in largest_channels)
        super().__init__(
            f"Document for collection '{collection}' is {bytes} bytes, over the "
            f"limit of {limit} bytes. Largest channels: [{top}]"
        )


class CheckpointIntegrityError(MongoDBCheckpointError):
    """A stored checkpoint references data that is missing (R4, R8)."""

    code = "CheckpointIntegrityError"

    def __init__(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
        missing: list[Any],
    ) -> None:
        self.thread_id = thread_id
        self.checkpoint_ns = checkpoint_ns
        self.checkpoint_id = checkpoint_id
        self.missing = missing
        super().__init__(
            f"Checkpoint {checkpoint_id!r} of thread {thread_id!r} "
            f"(namespace {checkpoint_ns!r}) is incomplete; missing: {missing!r}"
        )


class ThreadExpiredError(MongoDBCheckpointError):
    """A write targeted a thread that retention has tombstoned (R6)."""

    code = "ThreadExpiredError"

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(
            f"Thread {thread_id!r} has expired and been swept; it can no longer "
            "be written to."
        )


class CheckpointSerializationError(MongoDBCheckpointError, ValueError):
    """A stored value cannot be decoded: its compression codec is not
    available in this process, or it decompresses past
    ``max_decompressed_bytes`` (R5). Also a ``ValueError``."""

    code = "CheckpointSerializationError"


class IndexMigrationRequired(MongoDBCheckpointError):
    """A required unique index exists only as a non-unique index (R3)."""

    code = "IndexMigrationRequired"

    def __init__(self, collection: str, index_name: str, keys: list[Any]) -> None:
        self.collection = collection
        self.index_name = index_name
        self.keys = keys
        super().__init__(
            f"Index {index_name!r} on collection {collection!r} has keys {keys!r} "
            "but is not unique. Run scripts/check_unique_indexes.py to rebuild it "
            "as unique."
        )


__all__ = [
    "MongoDBCheckpointError",
    "CheckpointSizeError",
    "CheckpointIntegrityError",
    "CheckpointSerializationError",
    "ThreadExpiredError",
    "IndexMigrationRequired",
]
