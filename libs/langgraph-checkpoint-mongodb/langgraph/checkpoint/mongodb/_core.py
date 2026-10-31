"""Driver-free core of the MongoDB checkpointer.

Everything that builds, splits, measures, orders or reassembles documents lives
here so that the sync ``MongoDBSaver`` and any future ``AsyncMongoDBSaver``
share one implementation. Saver classes only perform I/O.

This module MUST NOT import ``pymongo`` (enforced by a test). It uses the
``bson`` codec (shipped alongside pymongo, but not the driver) only to measure
exact document sizes.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Union

import bson
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.base import SerializerProtocol

from . import compression as _compression
from .compression import DEFAULT_MAX_DECOMPRESSED_BYTES, CompressionConfig
from .errors import CheckpointIntegrityError, CheckpointSizeError

logger = logging.getLogger("langgraph.checkpoint.mongodb")

TASKS = "__pregel_tasks"
KiB = 1024
MiB = 1024 * KiB

DEFAULT_MAX_DOCUMENT_BYTES = 16 * MiB - 16 * KiB
DEFAULT_SIZE_WARNING_RATIO = 0.8
MAX_LARGEST_CHANNELS = 5

# --------------------------------------------------------------------------
# Size guard (R1)
# --------------------------------------------------------------------------


@dataclass
class DocumentSizeInfo:
    """Size telemetry for one document (REQ-E-003/004/005).

    ``largest_channels`` is only computed when the document is over the
    warning threshold; otherwise it is empty.
    """

    collection: str
    bytes: int
    limit: int
    largest_channels: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SizePolicy:
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    size_warning_ratio: float = DEFAULT_SIZE_WARNING_RATIO
    on_document_size: Callable[[DocumentSizeInfo], Any] | None = None


def measure(doc: dict[str, Any]) -> int:
    """Exact BSON size of ``doc``."""
    return len(bson.encode(doc))


def largest(
    channel_sizes: Iterable[tuple[str, int]], n: int = MAX_LARGEST_CHANNELS
) -> list[dict[str, Any]]:
    ordered = sorted(channel_sizes, key=lambda cs: cs[1], reverse=True)[:n]
    return [{"channel": c, "bytes": b} for c, b in ordered]


def check_document_size(
    collection: str,
    doc: dict[str, Any],
    policy: SizePolicy,
    channel_sizes: Callable[[], Iterable[tuple[str, int]]],
) -> DocumentSizeInfo:
    """Measure ``doc``; raise ``CheckpointSizeError`` if it is over the limit.

    ``channel_sizes`` is only called (lazily) above the warning threshold.
    """
    size = measure(doc)
    limit = policy.max_document_bytes
    info = DocumentSizeInfo(collection, size, limit)
    if size > policy.size_warning_ratio * limit:
        info.largest_channels = largest(channel_sizes())
        if size > limit:
            raise CheckpointSizeError(collection, size, limit, info.largest_channels)
        logger.warning(
            "MongoDB checkpointer: document for '%s' is %d bytes (%.1f MiB), "
            "%.0f%% of the %d byte limit. Largest channels: %s",
            collection,
            size,
            size / MiB,
            100 * size / limit,
            limit,
            info.largest_channels,
        )
    return info


def report_sizes(policy: SizePolicy, infos: Iterable[DocumentSizeInfo]) -> None:
    """Call ``on_document_size`` once per written document (REQ-E-005)."""
    if policy.on_document_size is None:
        return
    for info in infos:
        policy.on_document_size(info)


# --------------------------------------------------------------------------
# Metadata (unchanged semantics: dict with serialized leaves, INV-005)
# --------------------------------------------------------------------------


def loads_metadata(
    serde: SerializerProtocol, metadata: dict[str, Any]
) -> CheckpointMetadata:
    """Deserialize metadata document.

    Metadata is stored with string keys and serde-serialized leaves so that it
    can be filtered on efficiently.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = loads_metadata(serde, value)
        return output
    else:
        return serde.loads_typed(metadata)


def dumps_metadata(
    serde: SerializerProtocol,
    metadata: Union[CheckpointMetadata, Any],
) -> Union[tuple[str, bytes], dict[str, Any]]:
    """Serialize all values in a metadata dictionary, keeping keys as strings."""
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = dumps_metadata(serde, value)
        return output
    else:
        return serde.dumps_typed(metadata)


# --------------------------------------------------------------------------
# Pre-v4 pending sends (V1 / INV-006)
# --------------------------------------------------------------------------


def needs_pending_sends_migration(
    checkpoint: dict[str, Any], parent_checkpoint_id: str | None
) -> bool:
    """Pre-v4 checkpoints kept ``Send`` packets as TASKS writes on the parent."""
    return bool(parent_checkpoint_id) and int(checkpoint.get("v", 4)) < 4


def pending_sends_query(
    thread_id: str, checkpoint_ns: str, parent_checkpoint_id: str
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": parent_checkpoint_id,
        "channel": TASKS,
    }


def migrate_pending_sends(
    checkpoint: dict[str, Any], sends: list[Any], first_version: Any
) -> None:
    """Mirror of the Postgres saver's ``_migrate_pending_sends``.

    Adds the parent's TASKS writes to ``channel_values`` and gives the TASKS
    channel the checkpoint's maximum version so it is seen as available.
    """
    if not sends:
        return
    checkpoint["channel_values"][TASKS] = list(sends)
    versions = checkpoint["channel_versions"]
    versions[TASKS] = max(versions.values()) if versions else first_version


# --------------------------------------------------------------------------
# Pending-write ordering (REQ-E-040)
# --------------------------------------------------------------------------


def write_sort_key(doc: dict[str, Any]) -> tuple[str, str, int]:
    """Order pending writes by ``(task_path, task_id, idx)``.

    Documents written before ``task_path`` was stored sort as ``""``.
    """
    return (doc.get("task_path", "") or "", doc["task_id"], int(doc["idx"]))


def sort_write_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(docs, key=write_sort_key)


# --------------------------------------------------------------------------
# Indexes (R3)
# --------------------------------------------------------------------------

IndexKeys = list[tuple[str, int]]

CHECKPOINTS_INDEX_KEYS: IndexKeys = [
    ("thread_id", 1),
    ("checkpoint_ns", 1),
    ("checkpoint_id", -1),
]
WRITES_INDEX_KEYS: IndexKeys = [
    ("thread_id", 1),
    ("checkpoint_ns", 1),
    ("checkpoint_id", -1),
    ("task_id", 1),
    ("idx", 1),
]
BLOBS_INDEX_KEYS: IndexKeys = [
    ("thread_id", 1),
    ("checkpoint_ns", 1),
    ("channel", 1),
    ("version", 1),
]
THREADS_INDEX_KEYS: IndexKeys = [("thread_id", 1)]
TTL_INDEX_KEYS: IndexKeys = [("created_at", 1)]
TOMBSTONE_TTL_INDEX_KEYS: IndexKeys = [("tombstoned_at", 1)]


def index_keys(index: Mapping[str, Any]) -> IndexKeys:
    return [(k, v) for k, v in index["key"].items()]


def find_index(
    existing: Iterable[Mapping[str, Any]], keys: IndexKeys
) -> tuple[str, Mapping[str, Any] | None]:
    """Return ``("missing" | "unique" | "non_unique", index_doc)``.

    Only plain (non-partial) indexes with exactly ``keys`` count.
    """
    found: Mapping[str, Any] | None = None
    for ix in existing:
        if index_keys(ix) == keys and "partialFilterExpression" not in ix:
            if ix.get("unique"):
                return "unique", ix
            found = ix
    return ("non_unique", found) if found is not None else ("missing", None)


def find_ttl_index(
    existing: Iterable[Mapping[str, Any]], keys: IndexKeys, seconds: int
) -> bool:
    return any(
        index_keys(ix) == keys and ix.get("expireAfterSeconds") == seconds
        for ix in existing
    )


def default_companion_name(checkpoint_collection_name: str, kind: str) -> str:
    """Default name of the blobs / threads collection.

    ``checkpoint_blobs`` / ``threads`` for the default ``checkpoints``
    collection. Savers with a custom checkpoint collection get their own
    companion collection (``<name>_blobs``, ``<name>_threads``) so that
    several savers in one database never share or delete each other's data.
    """
    if checkpoint_collection_name == "checkpoints":
        return {"blobs": "checkpoint_blobs", "threads": "threads"}[kind]
    return f"{checkpoint_collection_name}_{kind}"


# --------------------------------------------------------------------------
# Versions (REQ-E-037/038)
# --------------------------------------------------------------------------

VersionKind = Literal["string", "number", "empty"]


def next_version(current: Any) -> Any:
    """Next channel version.

    - numbers stay numbers (``5 -> 6``) so existing integer threads keep
      integer versions (REQ-E-038, premortem 3);
    - ``None`` or a string returns ``"{n:032}.{r}"`` where ``n`` is the integer
      prefix plus one and ``r`` is a random fraction rendered with exactly 16
      characters (``"0." + 14 digits``). Strings sort by their zero-padded
      prefix; the random suffix makes fork siblings distinct (REQ-E-037).
    """
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        return current + 1
    n = 0 if current is None else int(str(current).split(".")[0])
    return f"{n + 1:032}.{random.random():.14f}"


def version_kind(channel_versions: Mapping[str, Any]) -> VersionKind:
    if not channel_versions:
        return "empty"
    if any(isinstance(v, str) for v in channel_versions.values()):
        return "string"
    return "number"


def uses_layout2(channel_versions: Mapping[str, Any]) -> bool:
    """String or empty versions use layout 2; numbers use the old format."""
    return version_kind(channel_versions) != "number"


# --------------------------------------------------------------------------
# Layout 2: per-channel storage (R4)
# --------------------------------------------------------------------------

LAYOUT_V2 = 2
DEFAULT_INLINE_MAX_CHARS = 1024
CLIENT_BULK_WRITE_MIN_WIRE_VERSION = 25  # MongoDB 8.0


@dataclass(frozen=True)
class CollectionNames:
    checkpoints: str
    writes: str
    blobs: str
    threads: str = "threads"


def is_layout2(doc: Mapping[str, Any]) -> bool:
    return doc.get("layout") == LAYOUT_V2


def classify_channel(value: Any, inline_max_chars: int) -> Literal["inline", "blob"]:
    """Inline: None, booleans, numbers and short strings. Everything else is a blob."""
    if value is None or isinstance(value, (bool, int, float)):
        return "inline"
    if isinstance(value, str) and len(value) <= inline_max_chars:
        return "inline"
    return "blob"


@dataclass
class BlobWrite:
    """Insert-if-absent of one channel value into the blobs collection."""

    filter: dict[str, Any]
    set_on_insert: dict[str, Any]
    info: DocumentSizeInfo


@dataclass
class PutPlan:
    """Everything ``put`` needs to write, computed (and size-checked) up front."""

    layout: int  # 1 = old single-document format, 2 = per-channel
    manifest_filter: dict[str, Any]
    manifest_set: dict[str, Any]
    blob_writes: list[BlobWrite]
    # blob keys referenced but not rewritten; only refreshed when a TTL is set
    blob_touches: list[dict[str, Any]]
    infos: list[DocumentSizeInfo]


def dumps_value(
    serde: SerializerProtocol,
    value: Any,
    compression: CompressionConfig | None = None,
) -> tuple[str, bytes]:
    """Serialize a blob / pending-write value, compressing it if configured."""
    type_, data = serde.dumps_typed(value)
    return _compression.compress(compression, type_, data)


def loads_value(
    serde: SerializerProtocol,
    type_: str,
    data: bytes,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> Any:
    """Inverse of ``dumps_value``; understands ``+gzip`` / ``+zstd`` always."""
    base, codec = _compression.split_type(type_)
    if codec is not None:
        data = _compression.decompress(codec, data, max_decompressed_bytes)
    return serde.loads_typed((base, data))


def plan_put(
    *,
    serde: SerializerProtocol,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint: Mapping[str, Any],
    metadata: Any,
    parent_checkpoint_id: str | None,
    new_versions: Mapping[str, Any] | None,
    names: CollectionNames,
    inline_max_chars: int = DEFAULT_INLINE_MAX_CHARS,
    size_policy: SizePolicy = SizePolicy(),
    created_at: datetime | None = None,
    compression: CompressionConfig | None = None,
) -> PutPlan:
    """Build (and size-check) every document ``put`` will write.

    Raises ``CheckpointSizeError`` before anything is written (REQ-E-002).
    """
    manifest_filter = {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint["id"],
    }
    values: Mapping[str, Any] = checkpoint.get("channel_values", {})
    versions: Mapping[str, Any] = checkpoint.get("channel_versions", {})

    def value_sizes(channels: Iterable[str]) -> Callable[[], list[tuple[str, int]]]:
        return lambda: [(c, len(dumps_value(serde, values[c])[1])) for c in channels]

    if not uses_layout2(versions):
        # Old single-document format, byte-for-byte as before (INV-002).
        type_, data = serde.dumps_typed(checkpoint)
        doc: dict[str, Any] = {
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": type_,
            "checkpoint": data,
            "metadata": dumps_metadata(serde, metadata),
        }
        if created_at is not None:
            doc["created_at"] = created_at
        info = check_document_size(
            names.checkpoints,
            {**manifest_filter, **doc},
            size_policy,
            value_sizes(list(values)),
        )
        return PutPlan(1, manifest_filter, doc, [], [], [info])

    inline: dict[str, Any] = {}
    refs: list[dict[str, Any]] = []
    if not isinstance(values, Mapping):
        # Not a channel mapping (malformed but historically accepted): keep the
        # checkpoint as-is in the manifest.
        values = {}
    for ch, value in values.items():
        if ch not in versions or classify_channel(value, inline_max_chars) == "inline":
            inline[ch] = value
        else:
            refs.append({"channel": ch, "version": versions[ch]})

    blob_writes: list[BlobWrite] = []
    touches: list[dict[str, Any]] = []
    for ref in refs:
        key = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "channel": ref["channel"],
            "version": ref["version"],
        }
        if new_versions is not None and ref["channel"] not in new_versions:
            touches.append(key)
            continue
        type_, data = dumps_value(serde, values[ref["channel"]], compression)
        body: dict[str, Any] = {"type": type_, "value": data}
        info = check_document_size(
            names.blobs,
            {**key, **body, **({"created_at": created_at} if created_at else {})},
            size_policy,
            lambda c=ref["channel"], n=len(data): [(c, n)],  # type: ignore[misc]
        )
        blob_writes.append(BlobWrite(key, body, info))

    manifest_checkpoint = (
        {**checkpoint, "channel_values": inline}
        if isinstance(checkpoint.get("channel_values", {}), Mapping)
        else dict(checkpoint)
    )
    type_, data = serde.dumps_typed(manifest_checkpoint)
    manifest: dict[str, Any] = {
        "parent_checkpoint_id": parent_checkpoint_id,
        "layout": LAYOUT_V2,
        "type": type_,
        "checkpoint": data,
        "metadata": dumps_metadata(serde, metadata),
        "blob_refs": refs,
        "inline_channels": list(inline),
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    manifest_info = check_document_size(
        names.checkpoints,
        {**manifest_filter, **manifest},
        size_policy,
        value_sizes(list(inline)),
    )
    infos = [b.info for b in blob_writes] + [manifest_info]
    return PutPlan(2, manifest_filter, manifest, blob_writes, touches, infos)


def blobs_query(
    thread_id: str, checkpoint_ns: str, refs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One query fetching exactly the referenced ``(channel, version)`` pairs."""
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "$or": [{"channel": r["channel"], "version": r["version"]} for r in refs],
    }


def load_blob_values(
    serde: SerializerProtocol,
    refs: Sequence[Mapping[str, Any]],
    blob_docs: Iterable[Mapping[str, Any]],
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str | None,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Deserialize the blobs for ``refs``; raise if any is missing (REQ-E-036)."""
    by_key = {(d["channel"], d["version"]): d for d in blob_docs}
    missing = [
        {"channel": r["channel"], "version": r["version"]}
        for r in refs
        if (r["channel"], r["version"]) not in by_key
    ]
    if missing:
        raise CheckpointIntegrityError(thread_id, checkpoint_ns, checkpoint_id, missing)
    out: dict[str, Any] = {}
    for r in refs:
        d = by_key[(r["channel"], r["version"])]
        out[r["channel"]] = loads_value(
            serde, d["type"], d["value"], max_decompressed_bytes
        )
    return out


def merge_blobs(checkpoint: dict[str, Any], blob_values: Mapping[str, Any]) -> Any:
    """Return ``checkpoint`` with blob values merged into ``channel_values``."""
    if not blob_values:
        return checkpoint
    return {
        **checkpoint,
        "channel_values": {**checkpoint.get("channel_values", {}), **blob_values},
    }


# --------------------------------------------------------------------------
# Old format -> layout 2 conversion (REQ-E-042, used by the migration script)
# --------------------------------------------------------------------------

ZERO_SUFFIX = f"{0.0:.14f}"  # "0.00000000000000"


def content_suffix(type_: str, data: bytes) -> str:
    """Deterministic 16-character fraction derived from a serialized value.

    Old threads reuse integer versions across fork siblings; deriving the
    suffix from the content keeps converted versions distinct exactly when
    the values differ, and makes conversion repeatable.
    """
    digest = hashlib.sha256(type_.encode() + b"\x00" + bytes(data)).hexdigest()
    return f"{int(digest[:12], 16) / 16**12:.14f}"[:16]


def convert_versions(
    serde: SerializerProtocol, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of an integer-version checkpoint with string versions.

    Every comparison LangGraph makes inside one checkpoint is preserved:
    ``versions_seen[node][ch]`` maps to the *same* string as
    ``channel_versions[ch]`` when they were equal, and otherwise to a
    zero-suffix string with the same integer prefix.
    """
    values: Mapping[str, Any] = checkpoint.get("channel_values", {}) or {}
    versions: Mapping[str, Any] = checkpoint.get("channel_versions", {}) or {}
    mapped: dict[str, Any] = {}
    for ch, v in versions.items():
        if isinstance(v, str):
            mapped[ch] = v
            continue
        if isinstance(v, bool) or not float(v).is_integer():
            raise ValueError(f"cannot convert non-integer version {v!r} of {ch!r}")
        suffix = (
            content_suffix(*dumps_value(serde, values[ch]))
            if ch in values
            else ZERO_SUFFIX
        )
        mapped[ch] = f"{int(v):032}.{suffix}"

    def seen_version(ch: str, sv: Any) -> Any:
        if isinstance(sv, str):
            return sv
        if ch in versions and not isinstance(versions[ch], str) and sv == versions[ch]:
            return mapped[ch]
        return f"{int(sv):032}.{ZERO_SUFFIX}"

    versions_seen = {
        node: {ch: seen_version(ch, sv) for ch, sv in seen.items()}
        for node, seen in (checkpoint.get("versions_seen", {}) or {}).items()
    }
    return {**checkpoint, "channel_versions": mapped, "versions_seen": versions_seen}


# --------------------------------------------------------------------------
# Pending writes (INV-003/004)
# --------------------------------------------------------------------------


@dataclass
class WriteOp:
    filter: dict[str, Any]
    update: dict[str, Any]
    info: DocumentSizeInfo


def plan_put_writes(
    *,
    serde: SerializerProtocol,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str,
    names: CollectionNames,
    special_idx: Mapping[str, int],
    size_policy: SizePolicy = SizePolicy(),
    created_at: datetime | None = None,
    compression: CompressionConfig | None = None,
) -> list[WriteOp]:
    """Upserts for ``put_writes``; size-checked before anything is written.

    The upsert filter is exactly the unique key
    ``(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)``. Writes are
    replaced only if every channel is a special one (errors, interrupts...);
    otherwise the first write wins (``$setOnInsert``), as before.
    """
    set_method = "$set" if all(w[0] in special_idx for w in writes) else "$setOnInsert"
    ops: list[WriteOp] = []
    for idx, (channel, value) in enumerate(writes):
        key = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "idx": special_idx.get(channel, idx),
        }
        type_, data = dumps_value(serde, value, compression)
        body: dict[str, Any] = {
            "task_path": task_path,
            "channel": channel,
            "type": type_,
            "value": data,
        }
        if created_at is not None:
            body["created_at"] = created_at
        info = check_document_size(
            names.writes,
            {**key, **body},
            size_policy,
            lambda c=channel, n=len(data): [(c, n)],  # type: ignore[misc]
        )
        ops.append(WriteOp(key, {set_method: body}, info))
    return ops


# --------------------------------------------------------------------------
# Thread-scoped retention (R6)
# --------------------------------------------------------------------------

DEFAULT_TOMBSTONE_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class RetentionConfig:
    """Thread-scoped retention; off unless passed to the saver.

    ``threads_collection_name`` defaults to ``threads`` (or
    ``<checkpoint_collection>_threads`` for a custom checkpoint collection).
    """

    enabled: bool = True
    tombstone_seconds: int = DEFAULT_TOMBSTONE_SECONDS
    threads_collection_name: str | None = None


def thread_touch(
    thread_id: str, now: datetime
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upsert raising ``last_active``; fails with a duplicate key on a tombstone.

    The filter excludes tombstoned records, so for a swept thread the upsert
    tries to insert a second ``{thread_id}`` and the unique index rejects it
    (REQ-E-062). No separate read is needed on the happy path.
    """
    return (
        {"thread_id": thread_id, "tombstone": {"$exists": False}},
        {"$max": {"last_active": now}, "$setOnInsert": {"generation": 0}},
    )


def claim_expired(
    cutoff: datetime, now: datetime
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conditional claim of one expired thread (REQ-E-061)."""
    return (
        {"last_active": {"$lt": cutoff}, "tombstone": {"$exists": False}},
        {"$set": {"tombstone": True, "tombstoned_at": now}, "$inc": {"generation": 1}},
    )


#: Claimed threads whose data deletion has not completed (interrupted sweeps).
#: A finished purge unsets ``last_active``.
UNPURGED_TOMBSTONES = {"tombstone": True, "last_active": {"$exists": True}}


def resolve_cutoff(older_than: datetime | timedelta, now: datetime) -> datetime:
    return older_than if isinstance(older_than, datetime) else now - older_than


# --------------------------------------------------------------------------
# DeltaChannel history (R8)
# --------------------------------------------------------------------------

DEFAULT_HISTORY_BATCH_SIZE = 100

#: Fields read while walking ancestors; never the checkpoint payload (REQ-E-081).
HISTORY_PROJECTION = {
    "_id": 0,
    "checkpoint_id": 1,
    "parent_checkpoint_id": 1,
    "layout": 1,
    "blob_refs": 1,
    "inline_channels": 1,
}


def stored_channels(doc: Mapping[str, Any]) -> set[str] | None:
    """Channels with a stored value in a manifest, or ``None`` (old format:
    only the payload knows)."""
    if not is_layout2(doc):
        return None
    return set(doc.get("inline_channels") or []) | {
        r["channel"] for r in doc.get("blob_refs") or []
    }


def blob_ref(doc: Mapping[str, Any], channel: str) -> dict[str, Any] | None:
    for r in doc.get("blob_refs") or []:
        if r["channel"] == channel:
            return {"channel": channel, "version": r["version"]}
    return None


def assemble_history(
    channels: Sequence[str],
    chain_ids: Sequence[str],
    seed_index: Mapping[str, int],
    seed_values: Mapping[str, Any],
    writes_by_checkpoint: Mapping[str, Sequence[tuple[str, str, Any]]],
) -> dict[str, Any]:
    """Shape the result exactly like ``BaseCheckpointSaver.get_delta_channel_history``.

    ``chain_ids`` lists ancestors nearest-first. A channel collects the writes
    of every ancestor up to and including its seed ancestor (or the whole
    chain without a seed), oldest ancestor first, each ancestor's writes in
    ``(task_path, task_id, idx)`` order.
    """
    result: dict[str, Any] = {}
    for ch in channels:
        last = seed_index.get(ch, len(chain_ids) - 1)
        writes = [
            w
            for i in range(last, -1, -1)
            for w in writes_by_checkpoint.get(chain_ids[i], ())
            if w[1] == ch
        ]
        entry: dict[str, Any] = {"writes": writes}
        if ch in seed_values:
            entry["seed"] = seed_values[ch]
        result[ch] = entry
    return result
