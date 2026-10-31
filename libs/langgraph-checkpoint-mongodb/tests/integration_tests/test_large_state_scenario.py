"""End-to-end large-state acceptance scenario: real StateGraph, real server, and a
FRESH saver instance for every read and resume.

1. ~6 MB history + ~12 MB tool result (> 16 MB total state): run, interrupt,
   resume, complete with values intact.
2. One channel > 16 MiB: CheckpointSizeError, nothing written, the thread
   resumes from the previous checkpoint.
3. One channel changing per step: only that channel's blob is rewritten.
4. Two update_state forks from one checkpoint read back their own values.
5. A thread written by the published 0.5.0 package resumes on the new code
   (old format, integer versions, no TypeError); after the migration script
   it resumes identically as layout 2.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bson
import pytest
from langgraph.types import Command
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import CheckpointSizeError, MongoDBSaver

from .graphs import LARGE_STATE_INPUT, large_state_graph, large_state_graph_small
from .legacy import legacy_write

MiB = 1024 * 1024
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _fresh(make_saver: Callable[..., MongoDBSaver], **graph_kw: Any) -> Any:
    return large_state_graph(**graph_kw).compile(checkpointer=make_saver())


def _expected_history(n: int = 30, total: int = 6_000_000) -> list[str]:
    per = total // n
    return [f"{i}:" + chr(97 + i % 26) * per for i in range(n)]


# ------------------------------------------------------------------ part 1


def test_large_state_1_over_16mb_state_interrupt_and_resume(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    cfg: Any = {"configurable": {"thread_id": "large-1"}}
    first = _fresh(make_saver).invoke(LARGE_STATE_INPUT, cfg)
    assert first["__interrupt__"][0].value == {
        "question": "approve?",
        "tool_bytes": 12_000_000,
    }

    state = _fresh(make_saver).get_state(cfg)
    assert state.next == ("approval",)
    assert len(state.values["variables"]["tool_result"]) == 12_000_000

    out = _fresh(make_saver).invoke(Command(resume="yes"), cfg)
    assert out["status"] == "done"
    assert out["variables"] == {
        "customer": "acme",
        "tool_result": "T" * 12_000_000,
        "rows": 42,
    }
    assert [m.content for m in out["conversation_history"][:-1]] == _expected_history()
    assert out["conversation_history"][-1].content == "final answer"
    assert [m.content for m in out["agent_a_conversation_history"]] == [
        "agent A summary"
    ]

    final = _fresh(make_saver).get_state(cfg)
    assert final.next == () and final.values == out
    # total state is > 16 MiB, yet every stored document is under the limit
    total = sum(
        len(bson.encode(d))
        for d in client[db_name]["checkpoint_blobs"].find({"thread_id": "large-1"})
    )
    assert total > 16 * MiB
    for coll in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        for d in client[db_name][coll].find({"thread_id": "large-1"}):
            assert len(bson.encode(d)) < 16 * MiB - 16 * 1024


# ------------------------------------------------------------------ part 2


def test_large_state_2_oversized_channel_raises_and_thread_resumes(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    cfg: Any = {"configurable": {"thread_id": "large-2b"}}
    small = {"history_bytes": 100_000, "tool_bytes": 100_000}
    graph = large_state_graph(huge_bytes=17 * MiB, **small).compile(
        checkpointer=make_saver(), interrupt_before=["huge"]
    )
    graph.invoke(LARGE_STATE_INPUT, cfg)

    def snapshot() -> dict[str, list[Any]]:
        return {
            c: sorted(
                str(d["_id"])
                for d in client[db_name][c].find({"thread_id": "large-2b"}, {"_id": 1})
            )
            for c in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
        }

    before = snapshot()
    previous = _fresh(make_saver, huge_bytes=17 * MiB, **small).get_state(cfg)
    assert previous.next == ("huge",)

    # durability="sync": the failing step writes nothing at all
    with pytest.raises(CheckpointSizeError) as ei:
        large_state_graph(huge_bytes=17 * MiB, **small).compile(
            checkpointer=make_saver()
        ).invoke(None, cfg, durability="sync")
    err = ei.value
    assert err.bytes > err.limit
    assert err.collection in ("checkpoint_writes", "checkpoint_blobs")
    assert err.largest_channels[0]["channel"] == "variables"
    assert snapshot() == before

    # default (async) durability: core may run the next step while the failed
    # write is in flight; its put is rejected too. At most an orphan pending
    # write keyed to a checkpoint that was never persisted can appear.
    with pytest.raises(CheckpointSizeError):
        large_state_graph(huge_bytes=17 * MiB, **small).compile(
            checkpointer=make_saver()
        ).invoke(None, cfg)
    after = snapshot()
    assert after["checkpoints"] == before["checkpoints"]
    assert after["checkpoint_blobs"] == before["checkpoint_blobs"]
    persisted = set(
        client[db_name]["checkpoints"].distinct(
            "checkpoint_id", {"thread_id": "large-2b"}
        )
    )
    for w in client[db_name]["checkpoint_writes"].find({"thread_id": "large-2b"}):
        if str(w["_id"]) not in before["checkpoint_writes"]:
            assert w["checkpoint_id"] not in persisted, w["channel"]
            assert len(w["value"]) < MiB
    # the thread is still at the previous checkpoint and resumes from it
    again = _fresh(make_saver, huge_bytes=17 * MiB, **small).get_state(cfg)
    assert again.config == previous.config and again.next == ("huge",)
    fixed = large_state_graph(huge_bytes=1 * MiB, **small).compile(checkpointer=make_saver())
    out = fixed.invoke(None, cfg)
    assert "__interrupt__" in out
    done = (
        large_state_graph(huge_bytes=1 * MiB, **small)
        .compile(checkpointer=make_saver())
        .invoke(Command(resume="ok"), cfg)
    )
    assert done["status"] == "done" and len(done["variables"]["huge"]) == 1 * MiB


# ------------------------------------------------------------------ part 3


def _blob_bytes_per_write(recorder: Any, blobs_coll: str) -> list[int]:
    """Bytes sent to the blobs collection, per write command."""
    out: list[int] = []
    for e in recorder.events:
        cmd = e.command
        if e.command_name == "update" and cmd.get("update") == blobs_coll:
            out.append(sum(len(bson.encode(u)) for u in cmd["updates"]))
        elif e.command_name == "bulkWrite":
            ns_idx = [
                i
                for i, n in enumerate(cmd["nsInfo"])
                if n["ns"].endswith("." + blobs_coll)
            ]
            if ns_idx:
                n = sum(
                    len(bson.encode(op))
                    for op in cmd["ops"]
                    if op.get("update") in ns_idx
                )
                if n:
                    out.append(n)
    return out


def test_large_state_3_only_changed_channel_is_rewritten(
    monitored_client: MongoClient, db_name: str, recorder: Any
) -> None:
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph

    from .graphs import LargeState

    def seed(state: LargeState) -> dict:
        return {
            "conversation_history": [AIMessage("h" * 1_000_000, id="h")],
            "agent_a_conversation_history": [AIMessage("a" * 1_000_000, id="a")],
            "variables": {"v": "0" * 1_000_000, "step": 0},
            "status": "seeded",
        }

    def tick(state: LargeState) -> dict:
        n = state["variables"]["step"] + 1
        return {
            "variables": {"v": str(n % 10) * 1_000_000, "step": n},
            "status": f"tick{n}",
        }

    g = StateGraph(LargeState)
    g.add_node("seed", seed)
    g.add_node("tick", tick)
    g.add_edge(START, "seed")
    g.add_edge("seed", "tick")
    g.add_conditional_edges(
        "tick", lambda s: "tick" if s["variables"]["step"] < 6 else END
    )
    cfg: Any = {"configurable": {"thread_id": "large-3"}}
    g.compile(checkpointer=MongoDBSaver(monitored_client, db_name)).invoke(
        {"status": "new"}, cfg
    )

    all_writes = _blob_bytes_per_write(recorder, "checkpoint_blobs")
    print("\nGEP-3 bytes written to checkpoint_blobs per write:", all_writes)
    # the first (tiny) write is the __start__ input; then one ~3 MB write when
    # the three big channels appear, then ~1 MB per step: only `variables`
    # changed although the state is ~3 MB at every step
    per_write = [n for n in all_writes if n > 10_000]
    assert per_write[0] > 2_900_000
    tick_writes = per_write[1:]
    assert len(tick_writes) == 6
    assert all(950_000 < n < 1_100_000 for n in tick_writes), tick_writes
    db = monitored_client[db_name]
    counts = {
        ch: db["checkpoint_blobs"].count_documents(
            {"thread_id": "large-3", "channel": ch}
        )
        for ch in ("conversation_history", "agent_a_conversation_history", "variables")
    }
    assert counts == {
        "conversation_history": 1,
        "agent_a_conversation_history": 1,
        "variables": 7,
    }
    out = large_state_graph().compile(checkpointer=MongoDBSaver(monitored_client, db_name))
    state = g.compile(checkpointer=MongoDBSaver(monitored_client, db_name)).get_state(
        cfg
    )
    assert state.values["variables"] == {"v": "6" * 1_000_000, "step": 6}
    assert state.values["conversation_history"][0].content == "h" * 1_000_000
    del out


# ------------------------------------------------------------------ part 4


def test_large_state_4_forks_read_their_own_values(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    small = {"history_bytes": 50_000, "tool_bytes": 50_000}
    cfg: Any = {"configurable": {"thread_id": "large-4"}}
    _fresh(make_saver, **small).invoke(LARGE_STATE_INPUT, cfg)
    _fresh(make_saver, **small).invoke(Command(resume="yes"), cfg)
    history = list(_fresh(make_saver, **small).get_state_history(cfg))
    ancestor = next(s for s in history if s.next == ("agent_a",))
    left = _fresh(make_saver, **small).update_state(
        ancestor.config,
        {"variables": {**ancestor.values["variables"], "branch": "LEFT"}},
    )
    right = _fresh(make_saver, **small).update_state(
        ancestor.config,
        {"variables": {**ancestor.values["variables"], "branch": "RIGHT"}},
    )
    assert (
        _fresh(make_saver, **small).get_state(left).values["variables"]["branch"]
        == "LEFT"
    )
    assert (
        _fresh(make_saver, **small).get_state(right).values["variables"]["branch"]
        == "RIGHT"
    )
    # each branch runs on to the interrupt with its own value
    for branch_cfg, side in ((left, "LEFT"), (right, "RIGHT")):
        out = _fresh(make_saver, **small).invoke(None, branch_cfg)
        assert out["variables"]["branch"] == side
    # the original line is untouched
    assert (
        "branch"
        not in _fresh(make_saver, **small)
        .get_state(history[0].config)
        .values["variables"]
    )


# ------------------------------------------------------------------ part 5


def test_large_state_5_published_050_thread_resumes_then_migrates(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    for t in ("old-a", "old-b"):
        rep = legacy_write("large_state_graph_small", db_name, t, input_name="LARGE_STATE_INPUT")
        assert rep["next"] == ["approval"]
        assert all(isinstance(v, int) for v in rep["versions"].values())
    coll = client[db_name]["checkpoints"]
    small = large_state_graph_small

    # (a) resume on the new code without migrating: no TypeError, stays old
    out_a = (
        small()
        .compile(checkpointer=make_saver())
        .invoke(Command(resume="yes"), {"configurable": {"thread_id": "old-a"}})
    )
    assert out_a["status"] == "done"
    assert all("layout" not in d for d in coll.find({"thread_id": "old-a"}))
    tup = make_saver().get_tuple({"configurable": {"thread_id": "old-a"}})
    assert tup and all(
        isinstance(v, int) for v in tup.checkpoint["channel_versions"].values()
    )
    assert (
        client[db_name]["checkpoint_blobs"].count_documents({"thread_id": "old-a"}) == 0
    )

    # (b) migrate the twin with the script, then resume: identical result
    state_before = (
        small()
        .compile(checkpointer=make_saver())
        .get_state({"configurable": {"thread_id": "old-b"}})
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "migrate_thread_layout.py"),
            "--db",
            db_name,
            "--thread-id",
            "old-b",
            "--apply",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"status": "converted"' in proc.stdout
    assert all(d.get("layout") == 2 for d in coll.find({"thread_id": "old-b"}))
    state_after = (
        small()
        .compile(checkpointer=make_saver())
        .get_state({"configurable": {"thread_id": "old-b"}})
    )
    assert (
        state_after.values == state_before.values
        and state_after.next == state_before.next
    )
    out_b = (
        small()
        .compile(checkpointer=make_saver())
        .invoke(Command(resume="yes"), {"configurable": {"thread_id": "old-b"}})
    )
    assert out_b == out_a
    assert all(d.get("layout") == 2 for d in coll.find({"thread_id": "old-b"}))
    tup_b = make_saver().get_tuple({"configurable": {"thread_id": "old-b"}})
    assert tup_b and all(
        isinstance(v, str) for v in tup_b.checkpoint["channel_versions"].values()
    )
