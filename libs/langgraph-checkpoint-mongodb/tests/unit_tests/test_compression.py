"""PR6: compression (TC-050..055)."""

import gzip
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import (
    CheckpointSerializationError,
    CompressionConfig,
    MongoDBSaver,
)
from langgraph.checkpoint.mongodb import compression as comp

KiB = 1024
MB = 1_000_000


def _text(n: int) -> str:
    return ("the quick brown fox jumps over the lazy dog " * (n // 44 + 1))[:n]


def _put(saver: MongoDBSaver, values: dict[str, Any], metadata: Any = None) -> Any:
    meta: Any = metadata or {}
    v = saver.get_next_version(None, None)
    ck = empty_checkpoint()
    ck["channel_values"] = values
    ck["channel_versions"] = {k: v for k in values}
    cfg = saver.put(
        {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
        ck,
        meta,
        dict(ck["channel_versions"]),
    )
    return cfg, ck


def test_tc050_large_blob_compressed_small_blob_not(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver(compression=CompressionConfig(codec="gzip", min_bytes=64 * KiB))
    cfg, ck = _put(saver, {"big": _text(MB), "small": _text(10_000)})
    blobs = {b["channel"]: b for b in client[db_name]["checkpoint_blobs"].find()}
    assert blobs["big"]["type"] == "msgpack+gzip"
    assert len(blobs["big"]["value"]) < MB / 10
    assert blobs["small"]["type"] == "msgpack"
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.checkpoint == ck

    # pending writes are compressed too
    saver.put_writes(cfg, [("big", _text(MB)), ("small", "s")], "task")
    w = {d["channel"]: d for d in client[db_name]["checkpoint_writes"].find()}
    assert w["big"]["type"] == "msgpack+gzip" and w["small"]["type"] == "msgpack"
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.pending_writes == [
        ("task", "big", _text(MB)),
        ("task", "small", "s"),
    ]


def test_tc051_reader_without_compression_reads_gzip(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    cfg, ck = _put(
        make_saver(compression=CompressionConfig()), {"big": _text(200 * KiB)}
    )
    plain = make_saver()
    assert plain.compression is None
    tup = plain.get_tuple(cfg)
    assert tup is not None and tup.checkpoint == ck


def test_tc052_unavailable_codec_raises_naming_it(
    make_saver: Callable[..., MongoDBSaver], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("zstandard")
    cfg, _ = _put(
        make_saver(compression=CompressionConfig(codec="zstd")),
        {"big": _text(200 * KiB)},
    )
    monkeypatch.delitem(comp.CODECS, "zstd")
    with pytest.raises(CheckpointSerializationError, match="zstd"):
        make_saver().get_tuple(cfg)
    with pytest.raises(CheckpointSerializationError, match="zstd"):
        CompressionConfig(codec="zstd")


def test_tc053_decompression_bomb_raises(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    bomb = gzip.compress(b"\0" * (8 * MB))
    with pytest.raises(CheckpointSerializationError, match="max_decompressed_bytes"):
        comp.decompress("gzip", bomb, MB)
    assert comp.decompress("gzip", bomb, 8 * MB) == b"\0" * (8 * MB)
    # through the saver, with a small cap
    cfg, _ = _put(make_saver(compression=CompressionConfig()), {"big": _text(2 * MB)})
    capped = make_saver(compression=CompressionConfig(max_decompressed_bytes=MB))
    with pytest.raises(CheckpointSerializationError, match="max_decompressed_bytes"):
        capped.get_tuple(cfg)


def test_tc053_zstd_bomb_raises() -> None:
    pytest.importorskip("zstandard")
    bomb = comp.CODECS["zstd"].compress(b"\0" * (8 * MB), None)
    with pytest.raises(CheckpointSerializationError, match="max_decompressed_bytes"):
        comp.decompress("zstd", bomb, MB)


def test_tc054_manifest_metadata_never_compressed(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver(compression=CompressionConfig(min_bytes=1))
    big_meta = {"notes": _text(200 * KiB)}
    cfg, ck = _put(saver, {"big": _text(200 * KiB), "n": 1}, metadata=big_meta)
    m = client[db_name]["checkpoints"].find_one({})
    assert m is not None
    assert "+" not in m["type"]
    assert m["metadata"]["notes"][0] == "msgpack"
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.metadata["notes"] == big_meta["notes"]
    # still filterable by value (INV-005)
    assert len(list(saver.list(None, filter={"notes": big_meta["notes"]}))) == 1


def test_tc055_zstd_round_trip(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    pytest.importorskip("zstandard")
    saver = make_saver(compression=CompressionConfig(codec="zstd", min_bytes=1024))
    cfg, ck = _put(saver, {"big": _text(MB)})
    b = client[db_name]["checkpoint_blobs"].find_one({"channel": "big"})
    assert b is not None and b["type"] == "msgpack+zstd"
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.checkpoint == ck


def test_compression_lets_big_compressible_channel_fit(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    """A 20 MB (compressible) channel fails without compression, fits with it."""
    from langgraph.checkpoint.mongodb import CheckpointSizeError

    with pytest.raises(CheckpointSizeError):
        _put(make_saver(), {"huge": _text(20 * MB)})
    cfg, ck = _put(
        make_saver(compression=CompressionConfig()), {"huge": _text(20 * MB)}
    )
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.checkpoint == ck


def test_split_type_keeps_other_suffixes() -> None:
    assert comp.split_type("msgpack+gzip") == ("msgpack", "gzip")
    assert comp.split_type("msgpack+aes") == ("msgpack+aes", None)
    assert comp.split_type("msgpack+aes+zstd") == ("msgpack+aes", "zstd")
    assert comp.split_type("json") == ("json", None)


def test_tc050_incompressible_value_is_stored_uncompressed(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    """Compress only if >= min_bytes AND the compressed form is smaller."""
    import os

    noise = os.urandom(200 * KiB)
    saver = make_saver(compression=CompressionConfig(min_bytes=64 * KiB))
    cfg, ck = _put(saver, {"noise": noise})
    b = client[db_name]["checkpoint_blobs"].find_one({"channel": "noise"})
    assert b is not None and "+" not in b["type"]  # stored as the serde wrote it
    tup = make_saver().get_tuple(cfg)
    assert tup is not None and tup.checkpoint == ck


def test_serialization_error_is_a_checkpoint_error() -> None:
    from langgraph.checkpoint.mongodb import MongoDBCheckpointError

    err = CheckpointSerializationError("boom")
    assert isinstance(err, MongoDBCheckpointError) and isinstance(err, ValueError)
    assert err.code == "CheckpointSerializationError"
