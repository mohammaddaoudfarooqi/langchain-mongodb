from __future__ import annotations

import asyncio
import builtins  # ``MongoDBSaver.list`` shadows ``list`` in annotations
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from langchain_core.runnables import RunnableConfig, run_in_executor
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ClientBulkWriteException, DuplicateKeyError

from . import _core
from ._core import RetentionConfig
from .compression import DEFAULT_MAX_DECOMPRESSED_BYTES, CompressionConfig
from .errors import (
    CheckpointIntegrityError,
    IndexMigrationRequired,
    ThreadExpiredError,
)
from .utils import (
    DRIVER_METADATA,
    _validate_filter,
    _validate_identifier,
    dumps_metadata,
    loads_metadata,
)

if TYPE_CHECKING:  # not exported by older langgraph-checkpoint releases
    from langgraph.checkpoint.base import DeltaChannelHistory

logger = logging.getLogger("langgraph.checkpoint.mongodb")

_TTL_DEPRECATION_LOGGED = False


def _log_ttl_deprecation() -> None:
    """REQ-E-064: one warning per process; ``ttl`` keeps working (INV-009)."""
    global _TTL_DEPRECATION_LOGGED
    if not _TTL_DEPRECATION_LOGGED:
        _TTL_DEPRECATION_LOGGED = True
        logger.warning(
            "MongoDBSaver(ttl=...) is deprecated: per-document TTL can expire part "
            "of a thread while the rest survives. Use "
            "retention=RetentionConfig() with saver.sweep_expired(...) instead."
        )


def _create_saver_indexes(
    collection: Collection,
    compound_index: list[tuple[str, int]],
    ttl: Optional[int] = None,
    ttl_field: str = "created_at",
) -> list[IndexMigrationRequired]:
    """Ensure the unique compound index (and optional TTL index) exist.

    If an index with the same keys exists but is not unique, it is left
    untouched and an ``IndexMigrationRequired`` is logged and returned
    (REQ-E-021). Use ``scripts/check_unique_indexes.py`` to rebuild it.

    Args:
        collection (Collection): The MongoDB collection to create indexes on.
        compound_index (list[tuple[str, int]]): The unique compound index.
        ttl (int, optional): Time to live in seconds for the TTL index.
        ttl_field (str): Field the TTL index is built on.

    Returns:
        The index problems found (empty when everything is in order).
    """
    errors: list[IndexMigrationRequired] = []
    indexes = list(collection.list_indexes())
    state, existing = _core.find_index(indexes, compound_index)
    if state == "missing":
        collection.create_index(compound_index, unique=True)
    elif state == "non_unique":
        assert existing is not None
        err = IndexMigrationRequired(
            collection.name, existing["name"], list(compound_index)
        )
        logger.warning("%s: %s", err.code, err)
        errors.append(err)
    if ttl is not None:
        ttl_keys = [(ttl_field, ASCENDING)]
        if not _core.find_ttl_index(indexes, ttl_keys, ttl):
            collection.create_index(ttl_keys, expireAfterSeconds=ttl)
    return errors


class MongoDBSaver(BaseCheckpointSaver):
    """A checkpointer that stores StateGraph checkpoints in a MongoDB database.

    A compound index as shown below will be added to each of the collections
    backing the saver (checkpoints, pending writes). If the collections pre-exist,
    and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True,

    Args:
        client (MongoClient): The MongoDB connection.
        db_name (Optional[str]): Database name
        checkpoint_collection_name (Optional[str]): Name of Collection of Checkpoints
        writes_collection_name (Optional[str]): Name of Collection of intermediate writes.
        ttl (Optional[int]): Time to live in seconds. See https://www.mongodb.com/docs/manual/core/index-ttl/.
            Deprecated in favour of ``retention`` + ``sweep_expired``.
        serde (Optional[SerializerProtocol]): Serializer (default ``JsonPlusSerializer``).
        blobs_collection_name (Optional[str]): Collection of per-channel values.
            Defaults to ``checkpoint_blobs`` (``<checkpoint_collection_name>_blobs``
            for a custom checkpoint collection).
        max_document_bytes (int): Largest document the saver will write;
            larger ones raise ``CheckpointSizeError`` and nothing is written.
            Defaults to 16 MiB - 16 KiB.
        size_warning_ratio (float): Log a warning above this fraction of
            ``max_document_bytes`` (default 0.8).
        on_document_size (Optional[Callable[[DocumentSizeInfo], Any]]): Called
            once per written document with its collection and BSON size.
        inline_max_chars (int): Strings up to this length (and None, booleans,
            numbers) stay inline in the checkpoint manifest; other values are
            stored as blobs (default 1024). Do not lower it for existing threads.
        compression (Optional[CompressionConfig]): Compress blob and
            pending-write values of at least ``min_bytes`` (gzip, or zstd with
            the ``[zstd]`` extra). Compressed values are always readable.
        retention (Optional[RetentionConfig]): Track thread activity so that
            ``sweep_expired`` can delete idle threads.
        history_batch_size (int): Ancestors per query in
            ``get_delta_channel_history`` (default 100).

    Threads created by this version use string channel versions and the
    per-channel layout (``layout: 2`` manifests plus ``checkpoint_blobs``).
    Threads written by earlier releases keep integer versions and the
    single-document format; ``scripts/migrate_thread_layout.py`` converts them.

    Examples:

        >>> from langgraph.checkpoint.mongodb import MongoDBSaver
        >>> from langgraph.graph import StateGraph
        >>> from pymongo import MongoClient
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> client = MongoClient("mongodb://localhost:27017")
        >>> memory = MongoDBSaver(client)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef8b22d-df71-6ddc-8001-7c821b5c45fd'}}, metadata={'source': 'loop', 'writes': {'add_one': 4}, 'step': 1, 'parents': {}}, created_at='2024-10-15T18:25:34.088329+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef8b22d-df6f-6eec-8000-20f621dcf3b7'}}, tasks=())

        Adding sharding support:

        >>> from langgraph.checkpoint.mongodb import MongoDBSaver
        >>> from pymongo import MongoClient
        >>> memory = MongoDBSaver(client)
        >>> client = MongoClient("mongodb://localhost:27017")
        >>> client.admin.command('enableSharding', memory.db_name)
        >>> shard_key = {'your_shard_key': 1}  # Specify your shard key
        >>> client.admin.command('shardCollection', f'{memory.db_name}.{memory.checkpoint_collection_name}', key=shard_key)
        >>> client.admin.command('shardCollection', f'{memory.db_name}.{memory.writes_collection_name}', key=shard_key)
    """

    client: MongoClient
    db: MongoDatabase

    def __init__(
        self,
        client: MongoClient,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        serde: SerializerProtocol | None = None,
        *,
        blobs_collection_name: Optional[str] = None,
        max_document_bytes: int = _core.DEFAULT_MAX_DOCUMENT_BYTES,
        size_warning_ratio: float = _core.DEFAULT_SIZE_WARNING_RATIO,
        on_document_size: Optional[Callable[[_core.DocumentSizeInfo], Any]] = None,
        inline_max_chars: int = _core.DEFAULT_INLINE_MAX_CHARS,
        compression: Optional[CompressionConfig] = None,
        retention: Optional[RetentionConfig] = None,
        history_batch_size: int = _core.DEFAULT_HISTORY_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[db_name]
        self.checkpoint_collection = self.db[checkpoint_collection_name]
        self.writes_collection = self.db[writes_collection_name]
        self.blobs_collection = self.db[
            blobs_collection_name
            or _core.default_companion_name(checkpoint_collection_name, "blobs")
        ]
        self.ttl = ttl
        if ttl is not None:
            _log_ttl_deprecation()
        self.retention = (
            retention if retention is not None and retention.enabled else None
        )
        self.threads_collection = self.db[
            (retention.threads_collection_name if retention else None)
            or _core.default_companion_name(checkpoint_collection_name, "threads")
        ]
        self.inline_max_chars = inline_max_chars
        self.history_batch_size = history_batch_size
        self.compression = compression
        self._max_decompressed_bytes = (
            compression.max_decompressed_bytes
            if compression is not None
            else DEFAULT_MAX_DECOMPRESSED_BYTES
        )
        self.size_policy = _core.SizePolicy(
            max_document_bytes=max_document_bytes,
            size_warning_ratio=size_warning_ratio,
            on_document_size=on_document_size,
        )
        if serde is not None:
            self.serde = serde
        else:
            self.serde = JsonPlusSerializer()
        # worker threads are only started on first use
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="mongodb-saver"
        )

        # One handshake, cached: picks the write path (REQ-E-041) and whether
        # delete_thread can use a transaction (REQ-E-071).
        hello = self.client.admin.command("hello")
        self._max_wire_version = int(hello.get("maxWireVersion", 0))
        self._use_client_bulk_write = (
            self._max_wire_version >= _core.CLIENT_BULK_WRITE_MIN_WIRE_VERSION
        )
        self._supports_transactions = bool(hello.get("setName")) or (
            hello.get("msg") == "isdbgrid"
        )

        self.setup_errors: builtins.list[IndexMigrationRequired] = []
        for coll, keys in (
            (self.checkpoint_collection, _core.CHECKPOINTS_INDEX_KEYS),
            (self.writes_collection, _core.WRITES_INDEX_KEYS),
            (self.blobs_collection, _core.BLOBS_INDEX_KEYS),
        ):
            self.setup_errors.extend(_create_saver_indexes(coll, keys, self.ttl))
        if self.retention is not None:
            # unique {thread_id} plus a TTL index removing tombstones
            self.setup_errors.extend(
                _create_saver_indexes(
                    self.threads_collection,
                    _core.THREADS_INDEX_KEYS,
                    self.retention.tombstone_seconds,
                    ttl_field="tombstoned_at",
                )
            )
            # sweep_expired claims by last_active
            self.threads_collection.create_index([("last_active", ASCENDING)])

    @property
    def _names(self) -> _core.CollectionNames:
        return _core.CollectionNames(
            self.checkpoint_collection.name,
            self.writes_collection.name,
            self.blobs_collection.name,
            self.threads_collection.name,
        )

    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("MongoDBSaver is closed")
        return self._executor

    def _gather(self, *calls: Callable[[], Any]) -> builtins.list[Any]:
        """Run independent queries concurrently (one round trip of latency)."""
        if len(calls) == 1:
            return [calls[0]()]
        futures = [self._pool().submit(c) for c in calls[1:]]
        try:
            first = calls[0]()
        finally:
            wait(futures)
        return [first, *(f.result() for f in futures)]

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: Optional[str] = None,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[MongoDBSaver]:
        """Context manager to create a MongoDB checkpoint saver.

        A compound index as shown below will be added to each of the collections
        backing the saver (checkpoints, pending writes). If the collections pre-exist,
        and have indexes already, nothing will be done during initialization::

        keys=[("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
        unique=True

        Args:
            conn_string: MongoDB connection string. See [class:~pymongo.MongoClient].
            db_name: Database name. It will be created if it doesn't exist.
            checkpoint_collection_name: Checkpoint Collection name. Created if it doesn't exist.
            writes_collection_name: Collection name of intermediate writes. Created if it doesn't exist.
            ttl (Optional[int]): Time to live in seconds.
            **kwargs: Further keyword arguments of ``MongoDBSaver``.
        Yields: A new MongoDBSaver.
        """
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                conn_string,
                driver=DRIVER_METADATA,
            )
            yield MongoDBSaver(
                client,
                db_name,
                checkpoint_collection_name,
                writes_collection_name,
                ttl,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def close(self) -> None:
        """Close the resources used by the MongoDBSaver."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self.client.close()

    def get_next_version(self, current: Any, channel: None = None) -> Any:
        """Next channel version (REQ-E-037/038).

        Integer versions (threads written by earlier releases) keep counting
        as integers; new threads get fork-unique ``"{n:032}.{r}"`` strings.
        """
        return _core.next_version(current)

    # ------------------------------------------------------------------ reads

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

         This method retrieves a checkpoint tuple from the MongoDB database based on the
         provided config. If the config contains a "checkpoint_id" key, the checkpoint with
         the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
         for the given thread ID is retrieved.

         Args:
             config (RunnableConfig): The config to use for retrieving the checkpoint.

         Returns:
             Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

             Basic:
             >>> config = {"configurable": {"thread_id": "1"}}
             >>> checkpoint_tuple = memory.get_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)

             With checkpoint ID:
             >>> config = {
             ...    "configurable": {
             ...        "thread_id": "1",
             ...        "checkpoint_ns": "",
             ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
             ...    }
             ... }
             >>> checkpoint_tuple = memory.get_tuple(config)
             >>> print(checkpoint_tuple)
             CheckpointTuple(...)
        """
        thread_id = _validate_identifier(
            config["configurable"]["thread_id"], "thread_id"
        )
        checkpoint_ns = _validate_identifier(
            config["configurable"].get("checkpoint_ns", ""), "checkpoint_ns"
        )
        checkpoint_id = _validate_identifier(
            get_checkpoint_id(config), "checkpoint_id", optional=True
        )
        if checkpoint_id:
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.checkpoint_collection.find(
            query, sort=[("checkpoint_id", -1)], limit=1
        )
        for doc in result:
            return self._tuple_from_doc(doc)
        return None

    def _load_writes(
        self, query: dict[str, Any]
    ) -> builtins.list[tuple[str, str, Any]]:
        """Pending writes ordered by (task_path, task_id, idx) (REQ-E-040)."""
        return [
            (
                w["task_id"],
                w["channel"],
                _core.loads_value(
                    self.serde, w["type"], w["value"], self._max_decompressed_bytes
                ),
            )
            for w in _core.sort_write_docs(list(self.writes_collection.find(query)))
        ]

    def _load_pending_sends(
        self, thread_id: str, checkpoint_ns: str, parent_id: str
    ) -> builtins.list[Any]:
        query = _core.pending_sends_query(thread_id, checkpoint_ns, parent_id)
        return [w[2] for w in self._load_writes(query)]

    def _tuple_from_doc(self, doc: dict[str, Any]) -> CheckpointTuple:
        """Assemble a tuple from a ``checkpoints`` document (either layout).

        One query step after the manifest: blobs, pending writes and (for
        pre-v4 checkpoints) the parent's sends are fetched concurrently.
        """
        thread_id = doc["thread_id"]
        checkpoint_ns = doc["checkpoint_ns"]
        checkpoint_id = doc["checkpoint_id"]
        parent_id = doc.get("parent_checkpoint_id")
        config_values = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
        refs = doc.get("blob_refs") or [] if _core.is_layout2(doc) else []
        migrate = _core.needs_pending_sends_migration(checkpoint, parent_id)

        # The (potentially large) blob fetch runs on the calling thread: its
        # connection was just used for the manifest, so its TCP window is
        # warm. The small pending-writes query goes to the pool.
        blobs: Optional[builtins.list[Any]] = None
        calls: builtins.list[Callable[[], Any]] = []
        if refs:
            calls.append(
                lambda: list(
                    self.blobs_collection.find(
                        _core.blobs_query(thread_id, checkpoint_ns, refs)
                    )
                )
            )
        calls.append(lambda: self._load_writes(config_values))
        if migrate:
            calls.append(
                lambda: self._load_pending_sends(thread_id, checkpoint_ns, parent_id)  # type: ignore[arg-type]
            )
        results = self._gather(*calls)
        if refs:
            blobs = results.pop(0)
        pending_writes = results[0]
        if blobs is not None:
            checkpoint = _core.merge_blobs(
                checkpoint,
                _core.load_blob_values(
                    self.serde,
                    refs,
                    blobs,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    max_decompressed_bytes=self._max_decompressed_bytes,
                ),
            )
        if migrate:
            _core.migrate_pending_sends(
                checkpoint, results[-1], self.get_next_version(None, None)
            )
        return CheckpointTuple(
            {"configurable": config_values},
            checkpoint,
            loads_metadata(self.serde, doc["metadata"]),
            (
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the MongoDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

            Examples:
            >>> from langgraph.checkpoint.mongodb import MongoDBSaver
            >>> with MongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            ... # Run a graph, then list the checkpoints
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]
        """
        query = {}
        if config is not None:
            if "thread_id" in config["configurable"]:
                query["thread_id"] = _validate_identifier(
                    config["configurable"]["thread_id"], "thread_id"
                )
            if "checkpoint_ns" in config["configurable"]:
                query["checkpoint_ns"] = _validate_identifier(
                    config["configurable"]["checkpoint_ns"], "checkpoint_ns"
                )

        if filter:
            _validate_filter(filter)
            for key, value in filter.items():
                query[f"metadata.{key}"] = dumps_metadata(self.serde, value)

        if before is not None:
            before_id = _validate_identifier(
                before["configurable"]["checkpoint_id"], "before checkpoint_id"
            )
            query["checkpoint_id"] = {"$lt": before_id}

        result = self.checkpoint_collection.find(
            query, limit=0 if limit is None else limit, sort=[("checkpoint_id", -1)]
        )

        for doc in result:
            yield self._tuple_from_doc(doc)

    # ------------------------------------------------------- delta history

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Per-channel writes + seed along the parent chain (R8).

        Same result as the base implementation, but the ancestor walk reads
        only ids, parents, ``layout``, ``blob_refs`` and ``inline_channels``
        with range scans of ``history_batch_size`` checkpoints; writes are
        fetched with one ``$in`` query per batch of ancestors; only seed
        values are loaded. A missing ancestor raises
        ``CheckpointIntegrityError`` instead of silently truncating.
        """
        if not channels:
            return {}
        thread_id = _validate_identifier(
            config["configurable"]["thread_id"], "thread_id"
        )
        checkpoint_ns = _validate_identifier(
            config["configurable"].get("checkpoint_ns", ""), "checkpoint_ns"
        )
        scope = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        query = dict(scope)
        if checkpoint_id := _validate_identifier(
            get_checkpoint_id(config), "checkpoint_id", optional=True
        ):
            query["checkpoint_id"] = checkpoint_id
        coll = self.checkpoint_collection
        projection = _core.HISTORY_PROJECTION
        target = next(
            iter(coll.find(query, projection, sort=[("checkpoint_id", -1)], limit=1)),
            None,
        )
        if target is None:
            return _core.assemble_history(channels, [], {}, {}, {})

        remaining = set(channels)
        chain: builtins.list[dict[str, Any]] = []
        seed_index: dict[str, int] = {}
        payloads: dict[str, Any] = {}  # old-format ancestors: full checkpoint
        batch: dict[str, dict[str, Any]] = {}
        current = target.get("parent_checkpoint_id")
        while current is not None and remaining:
            doc = batch.get(current)
            if doc is None:
                batch = {
                    d["checkpoint_id"]: d
                    for d in coll.find(
                        {**scope, "checkpoint_id": {"$lte": current}},
                        projection,
                        sort=[("checkpoint_id", -1)],
                        limit=self.history_batch_size,
                    )
                }
                doc = batch.get(current) or coll.find_one(
                    {**scope, "checkpoint_id": current}, projection
                )  # ids that are not time-ordered
                if doc is None:
                    child = chain[-1] if chain else target
                    raise CheckpointIntegrityError(
                        thread_id,
                        checkpoint_ns,
                        child["checkpoint_id"],
                        [{"checkpoint_id": current}],
                    )
            present = _core.stored_channels(doc)
            if present is None:
                full = coll.find_one(
                    {**scope, "checkpoint_id": current}, {"type": 1, "checkpoint": 1}
                )
                assert full is not None
                ck = self.serde.loads_typed((full["type"], full["checkpoint"]))
                payloads[current] = ck
                present = set(ck.get("channel_values") or {})
            for ch in [c for c in remaining if c in present]:
                seed_index[ch] = len(chain)
                remaining.discard(ch)
            chain.append(doc)
            current = doc.get("parent_checkpoint_id")

        chain_ids = [d["checkpoint_id"] for d in chain]
        seed_values = self._history_seeds(scope, target, chain, seed_index, payloads)
        needed = (
            len(chain) if remaining or not seed_index else max(seed_index.values()) + 1
        )
        writes_by_checkpoint: dict[str, builtins.list[tuple[str, str, Any]]] = {}
        for start in range(0, needed, self.history_batch_size):
            ids = chain_ids[start : start + self.history_batch_size]
            docs = self.writes_collection.find(
                {
                    **scope,
                    "checkpoint_id": {"$in": ids},
                    "channel": {"$in": list(channels)},
                }
            )
            grouped: dict[str, builtins.list[dict[str, Any]]] = {}
            for w in docs:
                grouped.setdefault(w["checkpoint_id"], []).append(w)
            for cid, group in grouped.items():
                writes_by_checkpoint[cid] = [
                    (
                        w["task_id"],
                        w["channel"],
                        _core.loads_value(
                            self.serde,
                            w["type"],
                            w["value"],
                            self._max_decompressed_bytes,
                        ),
                    )
                    for w in _core.sort_write_docs(group)
                ]
        return _core.assemble_history(
            channels, chain_ids, seed_index, seed_values, writes_by_checkpoint
        )

    def _history_seeds(
        self,
        scope: dict[str, Any],
        target: dict[str, Any],
        chain: builtins.list[dict[str, Any]],
        seed_index: dict[str, int],
        payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Load only the seed values: one blob query + one manifest query."""
        seeds: dict[str, Any] = {}
        blob_refs: builtins.list[dict[str, Any]] = []
        inline_ids: dict[str, builtins.list[str]] = {}
        for ch, i in seed_index.items():
            doc = chain[i]
            cid = doc["checkpoint_id"]
            if cid in payloads:
                seeds[ch] = payloads[cid]["channel_values"][ch]
            elif (ref := _core.blob_ref(doc, ch)) is not None:
                blob_refs.append(ref)
            else:
                inline_ids.setdefault(cid, []).append(ch)
        if blob_refs:
            seeds.update(
                _core.load_blob_values(
                    self.serde,
                    blob_refs,
                    self.blobs_collection.find(
                        _core.blobs_query(
                            scope["thread_id"], scope["checkpoint_ns"], blob_refs
                        )
                    ),
                    thread_id=scope["thread_id"],
                    checkpoint_ns=scope["checkpoint_ns"],
                    checkpoint_id=target["checkpoint_id"],
                    max_decompressed_bytes=self._max_decompressed_bytes,
                )
            )
        if inline_ids:
            for full in self.checkpoint_collection.find(
                {**scope, "checkpoint_id": {"$in": list(inline_ids)}},
                {"checkpoint_id": 1, "type": 1, "checkpoint": 1},
            ):
                ck = self.serde.loads_typed((full["type"], full["checkpoint"]))
                for ch in inline_ids[full["checkpoint_id"]]:
                    seeds[ch] = ck["channel_values"][ch]
        return seeds

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Asynchronously wraps ``get_delta_channel_history``."""
        return await run_in_executor(
            None,
            lambda: self.get_delta_channel_history(config=config, channels=channels),
        )

    # ----------------------------------------------------------------- writes

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        Threads with string (or no) channel versions use the per-channel
        layout: a ``layout: 2`` manifest in the checkpoints collection plus one
        document per changed, non-trivial channel value in the blobs
        collection. Threads with integer versions (written by earlier
        releases) keep the single-document format.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            CheckpointSizeError: a document would exceed ``max_document_bytes``;
                nothing was written.

        Examples:

            >>> from langgraph.checkpoint.mongodb import MongoDBSaver
            >>> with MongoDBSaver.from_conn_string("mongodb://localhost:27017") as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "data": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        thread_id = _validate_identifier(
            config["configurable"]["thread_id"], "thread_id"
        )
        checkpoint_ns = _validate_identifier(
            config["configurable"]["checkpoint_ns"], "checkpoint_ns"
        )
        _validate_identifier(checkpoint["id"], "checkpoint id")
        parent_checkpoint_id = _validate_identifier(
            config["configurable"].get("checkpoint_id"),
            "checkpoint_id",
            optional=True,
        )
        now = datetime.now(tz=UTC)
        plan = _core.plan_put(
            serde=self.serde,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint=checkpoint,
            metadata=get_checkpoint_metadata(config, metadata),
            parent_checkpoint_id=parent_checkpoint_id,
            new_versions=new_versions,
            names=self._names,
            inline_max_chars=self.inline_max_chars,
            size_policy=self.size_policy,
            created_at=now if self.ttl else None,
            compression=self.compression,
        )
        self._write(
            thread_id,
            now,
            blob_ops=self._blob_ops(plan, now) if plan.layout == 2 else [],
            final_collection=self.checkpoint_collection,
            final_ops=[
                {
                    "filter": plan.manifest_filter,
                    "update": {"$set": plan.manifest_set},
                    "upsert": True,
                }
            ],
        )
        _core.report_sizes(self.size_policy, plan.infos)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def _blob_ops(
        self, plan: _core.PutPlan, now: datetime
    ) -> builtins.list[dict[str, Any]]:
        """Blob updates as plain specs: insert-if-absent (REQ-E-034)."""
        ops: builtins.list[dict[str, Any]] = []
        for b in plan.blob_writes:
            update: dict[str, Any] = {"$setOnInsert": b.set_on_insert}
            if self.ttl:
                update["$max"] = {"created_at": now}
            ops.append({"filter": b.filter, "update": update, "upsert": True})
        if self.ttl:
            # keep referenced-but-unchanged blobs alive as long as their newest
            # manifest (no payload is resent)
            for key in plan.blob_touches:
                ops.append(
                    {
                        "filter": key,
                        "update": {"$max": {"created_at": now}},
                        "upsert": False,
                    }
                )
        return ops

    def _write(
        self,
        thread_id: str,
        now: datetime,
        *,
        blob_ops: builtins.list[dict[str, Any]],
        final_collection: Collection,
        final_ops: builtins.list[dict[str, Any]],
    ) -> None:
        """Write blobs (if any) before the final documents (REQ-E-034).

        - MongoDB 8.0+: one ordered client-level ``bulkWrite``: thread touch
          (retention), blobs, then the manifest / writes (1 round trip).
        - Older servers: the thread touch, the blob ``bulkWrite``, then the
          manifest / writes (up to 3 round trips). The touch goes first so an
          expired thread gets no blobs that no sweep would ever remove.
        """
        touch = (
            _core.thread_touch(thread_id, now) if self.retention is not None else None
        )
        if self._use_client_bulk_write and (touch or blob_ops):
            self._client_bulk_write(
                thread_id, touch, blob_ops, final_collection, final_ops
            )
            return
        if touch is not None:
            self._touch_thread(thread_id, touch)
        if blob_ops:
            self.blobs_collection.bulk_write(
                [UpdateOne(**op) for op in blob_ops], ordered=True
            )
        if len(final_ops) == 1:
            op = final_ops[0]
            final_collection.update_one(op["filter"], op["update"], upsert=op["upsert"])
        else:
            final_collection.bulk_write([UpdateOne(**op) for op in final_ops])

    def _client_bulk_write(
        self,
        thread_id: str,
        touch: Optional[tuple[dict[str, Any], dict[str, Any]]],
        blob_ops: builtins.list[dict[str, Any]],
        final_collection: Collection,
        final_ops: builtins.list[dict[str, Any]],
    ) -> None:
        models: builtins.list[UpdateOne] = []
        if touch is not None:
            models.append(
                UpdateOne(
                    touch[0],
                    touch[1],
                    upsert=True,
                    namespace=self.threads_collection.full_name,
                )
            )
        blobs_ns = self.blobs_collection.full_name
        models.extend(UpdateOne(**op, namespace=blobs_ns) for op in blob_ops)
        final_ns = final_collection.full_name
        models.extend(UpdateOne(**op, namespace=final_ns) for op in final_ops)
        for attempt in range(2):
            try:
                self.client.bulk_write(models, ordered=True)
                return
            except ClientBulkWriteException as e:
                errors: builtins.list[Any] = list(e.write_errors or [])
                first: Mapping[str, Any] = errors[0] if errors else {}
                if touch is None or first.get("idx") != 0 or first.get("code") != 11000:
                    raise
                # ordered: nothing after the touch was written
                if attempt or self._is_tombstoned(thread_id):
                    raise ThreadExpiredError(thread_id) from None

    def _is_tombstoned(self, thread_id: str) -> bool:
        rec = self.threads_collection.find_one(
            {"thread_id": thread_id}, {"tombstone": 1}
        )
        return bool(rec and rec.get("tombstone"))

    def _touch_thread(
        self, thread_id: str, touch: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Raise ``last_active``; a tombstone makes the upsert fail (REQ-E-062)."""
        try:
            self.threads_collection.update_one(touch[0], touch[1], upsert=True)
        except DuplicateKeyError:
            # Either the thread is tombstoned, or we lost a race inserting a
            # brand-new thread record (then the record exists: retry once).
            if self._is_tombstoned(thread_id):
                raise ThreadExpiredError(thread_id) from None
            self.threads_collection.update_one(touch[0], touch[1], upsert=True)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the MongoDB database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.

        Raises:
            CheckpointSizeError: a write document would exceed
                ``max_document_bytes``; nothing was written.
        """
        if not writes:
            return  # INV-004: no database operation
        thread_id = _validate_identifier(
            config["configurable"]["thread_id"], "thread_id"
        )
        checkpoint_ns = _validate_identifier(
            config["configurable"]["checkpoint_ns"], "checkpoint_ns"
        )
        checkpoint_id = _validate_identifier(
            config["configurable"]["checkpoint_id"], "checkpoint_id"
        )
        _validate_identifier(task_id, "task_id")
        _validate_identifier(task_path, "task_path")
        now = datetime.now(tz=UTC)
        ops = _core.plan_put_writes(
            serde=self.serde,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            writes=writes,
            task_id=task_id,
            task_path=task_path,
            names=self._names,
            special_idx=WRITES_IDX_MAP,
            size_policy=self.size_policy,
            created_at=now if self.ttl else None,
            compression=self.compression,
        )
        self._write(
            thread_id,
            now,
            blob_ops=[],
            final_collection=self.writes_collection,
            final_ops=[
                {"filter": op.filter, "update": op.update, "upsert": True} for op in ops
            ],
        )
        _core.report_sizes(self.size_policy, [op.info for op in ops])

    # -------------------------------------------------------------- retention

    def _purge_thread_data(self, thread_id: str) -> int:
        """Delete a thread's manifests, writes and blobs (idempotent)."""
        query = {"thread_id": thread_id}
        return sum(
            coll.delete_many(query).deleted_count
            for coll in (
                self.checkpoint_collection,
                self.writes_collection,
                self.blobs_collection,
            )
        )

    def sweep_expired(
        self,
        older_than: timedelta | datetime,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Delete threads inactive for longer than ``older_than`` (R6).

        Each expired thread is claimed with a conditional update that sets a
        tombstone, then its manifests, writes and blobs are deleted. The
        tombstone is kept (removed by a TTL index after
        ``retention.tombstone_seconds``) so late writers get
        ``ThreadExpiredError`` instead of resurrecting a half-deleted thread.
        Threads claimed by an interrupted earlier sweep are finished first.

        ``older_than`` must comfortably exceed the duration of a single
        ``put``; it is meant to be hours or days. The application schedules
        this call; the saver runs no background work.

        Args:
            older_than: Age (``timedelta``) or absolute cutoff (``datetime``).
            batch_size: Maximum number of threads processed by this call.

        Returns:
            ``{"threads": <threads purged>, "documents": <documents deleted>}``
        """
        if self.retention is None:
            raise ValueError("sweep_expired() needs retention=RetentionConfig(...)")
        now = datetime.now(tz=UTC)
        cutoff = _core.resolve_cutoff(older_than, now)
        threads = documents = 0

        def purge(thread_id: str) -> None:
            nonlocal threads, documents
            documents += self._purge_thread_data(thread_id)
            # dropping last_active marks the purge as complete
            self.threads_collection.update_one(
                {"thread_id": thread_id, "tombstone": True},
                {"$unset": {"last_active": ""}},
            )
            threads += 1

        for rec in self.threads_collection.find(
            _core.UNPURGED_TOMBSTONES, {"thread_id": 1}, limit=batch_size
        ):
            purge(rec["thread_id"])
        claim_filter, claim_update = _core.claim_expired(cutoff, now)
        while threads < batch_size:
            rec = self.threads_collection.find_one_and_update(
                claim_filter, claim_update, projection={"thread_id": 1}
            )
            if rec is None:
                break
            purge(rec["thread_id"])
        return {"threads": threads, "documents": documents}

    async def asweep_expired(
        self, older_than: timedelta | datetime, batch_size: int = 100
    ) -> dict[str, int]:
        """Asynchronously wraps ``sweep_expired``."""
        return await run_in_executor(None, self.sweep_expired, older_than, batch_size)

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete everything stored for a thread, in every namespace (R7).

        Order: checkpoints, pending writes, blobs, then the retention record
        (if retention is enabled). On a replica set or sharded cluster this
        runs in one transaction; otherwise every delete is idempotent, so
        re-running after a failure finishes the job.

        Args:
            thread_id (str): The thread ID whose checkpoints should be deleted.
        """
        _validate_identifier(thread_id, "thread_id")
        query = {"thread_id": thread_id}
        collections = [
            self.checkpoint_collection,
            self.writes_collection,
            self.blobs_collection,
        ]
        if self.retention is not None:
            collections.append(self.threads_collection)

        def delete_all(session: Any = None) -> None:
            for coll in collections:
                coll.delete_many(query, session=session)

        if self._supports_transactions:
            with self.client.start_session() as session:
                session.with_transaction(delete_all)
        else:
            delete_all()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Asynchronously wraps the blocking `self.get_tuple` method.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.

        """
        return await run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Asynchronously wraps the blocking `self.list` generator.

        Runs `self.list(...)` in a background thread and yields its items
        asynchronously from an asyncio.Queue. This allows integration of
        synchronous iterators into async code.

        Args:
            config: Configuration object passed to `self.list`.
            filter: Optional filter dictionary.
            before: Optional parameter to limit results before a given checkpoint.
            limit: Optional maximum number of results to yield.

        Yields:
           AsyncIterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[CheckpointTuple] = asyncio.Queue()
        sentinel = object()

        def run() -> None:
            try:
                for item in self.list(
                    config, filter=filter, before=before, limit=limit
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)  # type: ignore

        await run_in_executor(None, run)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Asynchronously wraps the blocking `self.put` method.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return await run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Asynchronously wraps the blocking `self.put_writes` method.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return await run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Asynchronously wraps the blocking `self.delete_thread` method.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        return await run_in_executor(None, self.delete_thread, thread_id)
