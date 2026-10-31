"""Optional compression of blob and pending-write values (R5).

A compressed value is stored with the serializer's type plus a codec suffix,
e.g. ``"msgpack+gzip"`` or ``"msgpack+zstd"``. Readers always understand the
suffixes (REQ-E-051), whether or not compression is configured.

- ``gzip``: standard library ``zlib`` (gzip container), always available.
- ``zstd``: needs the optional ``zstandard`` package
  (``pip install "langgraph-checkpoint-mongodb[zstd]"``).

Driver-free (no ``pymongo`` import).
"""

from __future__ import annotations

import io
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from .errors import CheckpointSerializationError

KiB = 1024
MiB = 1024 * KiB

GZIP = "gzip"
ZSTD = "zstd"
#: Every codec name a stored type may carry, available or not.
KNOWN_CODECS = (GZIP, ZSTD)

DEFAULT_MIN_BYTES = 64 * KiB
DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 16 * MiB
_READ_CHUNK = 1 * MiB


@dataclass(frozen=True)
class Codec:
    name: str
    compress: Callable[[bytes, int | None], bytes]
    # (data, max_output_bytes) -> bytes; raises ValueError above the cap
    decompress: Callable[[bytes, int], bytes]


def _too_big(codec: str, cap: int) -> CheckpointSerializationError:
    return CheckpointSerializationError(
        f"Decompressed {codec} value exceeds max_decompressed_bytes ({cap} bytes)"
    )


def _gzip_compress(data: bytes, level: int | None) -> bytes:
    c = zlib.compressobj(6 if level is None else level, zlib.DEFLATED, 31)
    return c.compress(data) + c.flush()


def _gzip_decompress(data: bytes, cap: int) -> bytes:
    d = zlib.decompressobj(31)
    out = d.decompress(data, cap + 1)
    if len(out) > cap or d.unconsumed_tail:
        raise _too_big(GZIP, cap)
    if not d.eof:
        raise CheckpointSerializationError("Truncated gzip value")
    return out


CODECS: dict[str, Codec] = {GZIP: Codec(GZIP, _gzip_compress, _gzip_decompress)}

try:  # optional zstd support
    import zstandard as _zstd

    def _zstd_compress(data: bytes, level: int | None) -> bytes:
        return _zstd.ZstdCompressor(level=3 if level is None else level).compress(data)

    def _zstd_decompress(data: bytes, cap: int) -> bytes:
        reader = _zstd.ZstdDecompressor().stream_reader(io.BytesIO(data))
        buf = bytearray()
        while True:
            chunk = reader.read(_READ_CHUNK)
            if not chunk:
                return bytes(buf)
            buf += chunk
            if len(buf) > cap:
                raise _too_big(ZSTD, cap)

    CODECS[ZSTD] = Codec(ZSTD, _zstd_compress, _zstd_decompress)
except ImportError:  # pragma: no cover - depends on the environment
    pass


def _require(codec: str) -> Codec:
    try:
        return CODECS[codec]
    except KeyError:
        hint = (
            ' Install it with `pip install "langgraph-checkpoint-mongodb[zstd]"`.'
            if codec == ZSTD
            else ""
        )
        raise CheckpointSerializationError(
            f"Compression codec '{codec}' is not available in this process.{hint}"
        ) from None


@dataclass(frozen=True)
class CompressionConfig:
    """Compress blob and pending-write values of at least ``min_bytes``."""

    codec: str = GZIP
    min_bytes: int = DEFAULT_MIN_BYTES
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES
    level: int | None = None

    def __post_init__(self) -> None:
        if self.codec not in KNOWN_CODECS:
            raise ValueError(
                f"Unknown compression codec {self.codec!r}; use one of {KNOWN_CODECS}"
            )
        _require(self.codec)


def split_type(type_: str) -> tuple[str, str | None]:
    """``"msgpack+gzip" -> ("msgpack", "gzip")``; other suffixes are kept."""
    base, sep, suffix = type_.rpartition("+")
    if sep and suffix in KNOWN_CODECS:
        return base, suffix
    return type_, None


def compress(
    config: CompressionConfig | None, type_: str, data: bytes
) -> tuple[str, bytes]:
    """Compress if ``data`` is at least ``min_bytes`` and gets smaller."""
    if config is None or len(data) < config.min_bytes:
        return type_, data
    packed = _require(config.codec).compress(bytes(data), config.level)
    if len(packed) >= len(data):
        return type_, data
    return f"{type_}+{config.codec}", packed


def decompress(codec: str, data: bytes, max_decompressed_bytes: int) -> bytes:
    return _require(codec).decompress(bytes(data), max_decompressed_bytes)
