"""Benchmark: old single-document format vs per-channel layout 2 (REQ-NF-003).

Not a test. For each total state size, a thread of 4 large channels (plus two
small inline ones) is advanced ``--iterations`` steps; every step changes ONE
large channel. Per step it times ``put`` (new checkpoint), ``put_writes`` (one
~10 KB write) and ``get_tuple`` (latest), and records the bytes sent to the
server by the ``put``.

"old" uses integer versions, so the saver writes the single-document format
(what threads created by <= 0.5.0 keep using); "layout2" uses string versions.

    MONGODB_URI=... uv run python bench/layout.py [--sizes 1,6,15] [--iterations 50]

Databases are named ``ckpt_test_py_bench_*`` and dropped afterwards.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import string
import time
import uuid
from typing import Any

import bson
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from pymongo import MongoClient, monitoring

from langgraph.checkpoint.mongodb import MongoDBSaver

MB = 1_000_000
WRITE_COMMANDS = {"insert", "update", "bulkWrite"}


class BytesListener(monitoring.CommandListener):
    def __init__(self) -> None:
        self.bytes = 0
        self.active = False

    def started(self, event: monitoring.CommandStartedEvent) -> None:
        if self.active and event.command_name in WRITE_COMMANDS:
            self.bytes += len(bson.encode(event.command))

    def succeeded(self, event: monitoring.CommandSucceededEvent) -> None:
        pass

    def failed(self, event: monitoring.CommandFailedEvent) -> None:
        pass


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def payload(rng: random.Random, n: int) -> str:
    chunk = "".join(rng.choices(string.ascii_letters, k=4096))
    return (chunk * (n // 4096 + 1))[:n]


def run_case(uri: str, total_mb: int, layout: str, iterations: int) -> dict[str, Any]:
    listener = BytesListener()
    client: MongoClient = MongoClient(uri, event_listeners=[listener])
    db = f"ckpt_test_py_bench_{uuid.uuid4().hex[:8]}"
    rng = random.Random(total_mb)
    try:
        saver = MongoDBSaver(client, db)
        per = total_mb * MB // 4
        big = [f"big{i}" for i in range(4)]
        values: dict[str, Any] = {c: payload(rng, per) for c in big}
        values.update({"status": "running", "step": 0})

        def version(prev: Any) -> Any:
            if layout == "old":
                return (prev or 0) + 1
            return saver.get_next_version(prev, None)

        cfg: Any = {"configurable": {"thread_id": "bench", "checkpoint_ns": ""}}
        ck = empty_checkpoint()
        v = version(None)
        ck["channel_values"] = dict(values)
        ck["channel_versions"] = dict.fromkeys(values, v)
        cfg = saver.put(cfg, ck, {"step": 0}, dict(ck["channel_versions"]))

        put_t, writes_t, get_t, put_bytes = [], [], [], []
        write_value = payload(rng, 10_000)
        for i in range(1, iterations + 1):
            ch = big[i % 4]
            values = {**values, ch: payload(rng, per), "step": i, "status": f"s{i}"}
            nxt = create_checkpoint(ck, None, i)
            cur_max = max(ck["channel_versions"].values())
            nv = version(cur_max)
            versions = dict(ck["channel_versions"])
            new_versions = {c: nv for c in (ch, "step", "status")}
            versions.update(new_versions)
            nxt["channel_values"] = values
            nxt["channel_versions"] = versions

            listener.bytes, listener.active = 0, True
            t0 = time.perf_counter()
            cfg = saver.put(cfg, nxt, {"step": i}, new_versions)
            put_t.append(time.perf_counter() - t0)
            listener.active = False
            put_bytes.append(listener.bytes)
            ck = nxt

            t0 = time.perf_counter()
            saver.put_writes(cfg, [("big0", write_value)], f"task-{i}", "~bench")
            writes_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            tup = saver.get_tuple(
                {"configurable": {"thread_id": "bench", "checkpoint_ns": ""}}
            )
            get_t.append(time.perf_counter() - t0)
            assert (
                tup is not None and tup.checkpoint["channel_values"][ch] == values[ch]
            )
        ms = 1000.0
        return {
            "size": total_mb,
            "layout": layout,
            "put_p50": pct(put_t, 50) * ms,
            "put_p99": pct(put_t, 99) * ms,
            "writes_p50": pct(writes_t, 50) * ms,
            "writes_p99": pct(writes_t, 99) * ms,
            "get_p50": pct(get_t, 50) * ms,
            "get_p99": pct(get_t, 99) * ms,
            "bytes_per_step": statistics.mean(put_bytes),
            "server": client.server_info()["version"],
            "path": "client bulkWrite"
            if saver._use_client_bulk_write
            else "collection bulkWrite + update",
        }
    finally:
        client.drop_database(db)
        client.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", default="1,6,15")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument(
        "--uri",
        default=os.environ.get(
            "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
        ),
    )
    args = p.parse_args()
    rows = [
        run_case(args.uri, int(s), layout, args.iterations)
        for s in args.sizes.split(",")
        for layout in ("old", "layout2")
    ]
    print(
        f"server {rows[0]['server']}, layout-2 write path: {rows[0]['path']}, {args.iterations} steps, 1 of 4 large channels changing per step\n"
    )
    print(
        "| state | layout | put p50 | put p99 | put_writes p50 | put_writes p99 | get_tuple p50 | get_tuple p99 | bytes written / step |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['size']} MB | {r['layout']} | {r['put_p50']:.1f} ms | {r['put_p99']:.1f} ms | "
            f"{r['writes_p50']:.1f} ms | {r['writes_p99']:.1f} ms | {r['get_p50']:.1f} ms | "
            f"{r['get_p99']:.1f} ms | {r['bytes_per_step'] / MB:.2f} MB |"
        )


if __name__ == "__main__":
    main()
