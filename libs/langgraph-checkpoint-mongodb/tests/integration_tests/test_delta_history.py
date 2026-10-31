"""PR9: get_delta_channel_history override (TC-080..083)."""

import math
import random
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import CheckpointIntegrityError, MongoDBSaver


class CachedReads:
    """Reference reader for the base implementation: ``get_tuple`` memoized
    per explicit checkpoint id (keeps the base walk affordable on
    high-latency clusters; results are identical)."""

    def __init__(self, saver: MongoDBSaver) -> None:
        self.saver = saver
        self.cache: dict[tuple[str, str, str], Any] = {}

    def get_tuple(self, config: Any) -> Any:
        c = config["configurable"]
        if not c.get("checkpoint_id"):
            return self.saver.get_tuple(config)
        key = (c["thread_id"], c.get("checkpoint_ns", ""), c["checkpoint_id"])
        if key not in self.cache:
            self.cache[key] = self.saver.get_tuple(config)
        return self.cache[key]


def base_history(
    saver: MongoDBSaver,
    config: Any,
    channels: Sequence[str],
    reads: CachedReads | None = None,
) -> Any:
    """``BaseCheckpointSaver.get_delta_channel_history`` (the reference)."""
    return BaseCheckpointSaver.get_delta_channel_history(
        reads or CachedReads(saver),  # type: ignore[arg-type]
        config=config,
        channels=channels,
    )


def _random_history(
    saver: MongoDBSaver, rng: random.Random, thread: str, int_versions: bool
) -> list[Any]:
    """A random checkpoint tree (forks), two namespaces, random channel presence
    (absent / inline / blob), overwrites and random writes per checkpoint."""
    configs: list[Any] = []
    for ns in ("", "sub:1"):
        nodes: list[tuple[Any, Any]] = []  # (config, checkpoint)
        for step in range(rng.randrange(6, 14)):
            parent = rng.choice(nodes) if nodes and rng.random() < 0.9 else None
            if parent is not None and rng.random() < 0.6:
                parent = nodes[-1]  # mostly linear, sometimes forks
            ck = empty_checkpoint()
            prev_versions = dict(parent[1]["channel_versions"]) if parent else {}
            cur_max = max(prev_versions.values()) if prev_versions else None
            versions, values, new_versions = dict(prev_versions), {}, {}
            for ch in ("d1", "d2", "plain"):
                kind = rng.choice(["absent", "absent", "inline", "blob", "keep"])
                if kind == "keep" and parent and ch in parent[1]["channel_values"]:
                    values[ch] = parent[1]["channel_values"][ch]
                    continue
                if kind == "absent":
                    continue
                values[ch] = (
                    rng.randrange(100) if kind == "inline" else [ch, step, rng.random()]
                )
                if int_versions:
                    versions[ch] = (cur_max or 0) + 1
                else:
                    versions[ch] = saver.get_next_version(cur_max, None)
                new_versions[ch] = versions[ch]
            ck["channel_values"], ck["channel_versions"] = values, versions
            cfg = saver.put(
                parent[0]
                if parent
                else {"configurable": {"thread_id": thread, "checkpoint_ns": ns}},
                ck,
                {"step": step},
                new_versions,
            )
            for t in range(rng.randrange(0, 4)):
                writes = [
                    (rng.choice(["d1", "d2", "plain", "other"]), [step, t, i])
                    for i in range(rng.randrange(1, 3))
                ]
                saver.put_writes(
                    cfg,
                    writes,
                    f"task-{rng.randrange(3)}",
                    rng.choice(["", "~a", "~b"]),
                )
            nodes.append((cfg, ck))
            configs.append(cfg)
    return configs


@pytest.mark.parametrize("int_versions", [False, True])
def test_tc080_override_equals_base_on_generated_histories(
    make_saver: Callable[..., MongoDBSaver], int_versions: bool
) -> None:
    saver = make_saver()
    rng = random.Random(80 + int_versions)
    checked = 0
    for h in range(10 if not int_versions else 3):
        configs = _random_history(saver, rng, f"h{h}", int_versions)
        reads = CachedReads(saver)
        for cfg in configs:
            for chans in (["d1"], ["d1", "d2", "plain", "other"], []):
                assert saver.get_delta_channel_history(
                    config=cfg, channels=chans
                ) == base_history(saver, cfg, chans, reads), (h, cfg, chans)
                checked += 1
        # latest checkpoint (no checkpoint_id) and an unknown thread
        latest: Any = {"configurable": {"thread_id": f"h{h}", "checkpoint_ns": ""}}
        assert saver.get_delta_channel_history(
            config=latest, channels=["d1"]
        ) == base_history(saver, latest, ["d1"])
    nothing: Any = {"configurable": {"thread_id": "missing", "checkpoint_ns": ""}}
    assert saver.get_delta_channel_history(config=nothing, channels=["d1"]) == {
        "d1": {"writes": []}
    }
    assert checked > 100


def concat(state: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    out = list(state or [])
    for w in writes:
        out.extend(w)
    return out


class DS(TypedDict, total=False):
    log: Annotated[list[str], DeltaChannel(concat, snapshot_frequency=3)]
    n: int


def test_tc080_real_delta_graph_with_snapshots_and_forks(
    make_saver: Callable[..., MongoDBSaver],
) -> None:
    def step(state: DS) -> dict:
        n = state.get("n", 0) + 1
        return {"log": [f"s{n}"], "n": n}

    g = StateGraph(DS)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", lambda s: "step" if s["n"] < 8 else END)
    cfg: Any = {"configurable": {"thread_id": "real"}}
    graph = g.compile(checkpointer=make_saver())
    graph.invoke({"log": ["start"], "n": 0}, cfg)
    hist = list(graph.get_state_history(cfg))
    fork = graph.update_state(hist[4].config, {"log": ["forked"]})
    saver = make_saver()
    reads = CachedReads(saver)
    for s in [*hist, graph.get_state(fork)]:
        assert saver.get_delta_channel_history(
            config=s.config, channels=["log", "n"]
        ) == base_history(saver, s.config, ["log", "n"], reads)
    fresh = g.compile(checkpointer=make_saver())
    assert fresh.get_state(hist[0].config).values["log"] == ["start"] + [
        f"s{i}" for i in range(1, 9)
    ]
    assert fresh.get_state(fork).values["log"][-1] == "forked"


def _chain(saver: MongoDBSaver, steps: int, thread: str = "long") -> Any:
    """``steps`` checkpoints; channel d seeded (blob) only at the root, one
    write to d per step.

    Documents are produced by the saver's own core planners (identical shapes
    to ``put`` / ``put_writes``) and bulk-inserted, so building a long chain
    does not cost two round trips per step.
    """
    from langgraph.checkpoint.base import WRITES_IDX_MAP

    from langgraph.checkpoint.mongodb import _core

    manifests: list[dict[str, Any]] = []
    blobs: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    parent: str | None = None
    version: Any = None
    for i in range(steps):
        ck = empty_checkpoint()
        version = saver.get_next_version(version, None)
        ck["channel_versions"] = {"d": version}
        ck["channel_values"] = {"d": ["seed"]} if i == 0 else {}
        plan = _core.plan_put(
            serde=saver.serde,
            thread_id=thread,
            checkpoint_ns="",
            checkpoint=ck,
            metadata={"step": i},
            parent_checkpoint_id=parent,
            new_versions={"d": version} if i == 0 else {},
            names=saver._names,
        )
        manifests.append({**plan.manifest_filter, **plan.manifest_set})
        blobs.extend({**b.filter, **b.set_on_insert} for b in plan.blob_writes)
        for op in _core.plan_put_writes(
            serde=saver.serde,
            thread_id=thread,
            checkpoint_ns="",
            checkpoint_id=ck["id"],
            writes=[("d", [i])],
            task_id=f"t{i}",
            task_path="~p",
            names=saver._names,
            special_idx=WRITES_IDX_MAP,
        ):
            writes.append({**op.filter, **op.update["$setOnInsert"]})
        parent = ck["id"]
    saver.checkpoint_collection.insert_many(manifests)
    if blobs:
        saver.blobs_collection.insert_many(blobs)
    saver.writes_collection.insert_many(writes)
    return {
        "configurable": {
            "thread_id": thread,
            "checkpoint_ns": "",
            "checkpoint_id": parent,
        }
    }


def test_tc081_query_budget_and_projection(
    monitored_client: MongoClient, db_name: str, recorder: Any
) -> None:
    saver = MongoDBSaver(monitored_client, db_name)
    target = _chain(saver, 1000)
    recorder.clear()
    hist = saver.get_delta_channel_history(config=target, channels=["d"])
    finds = [
        e for e in recorder.events if e.command_name in ("find", "aggregate", "getMore")
    ]
    budget = math.ceil(1000 / 100) + math.ceil(1000 / 100) + 2
    assert len(finds) <= budget, [e.command_name for e in finds]
    # no full checkpoint payloads are read while walking
    for e in finds:
        if e.command.get("find") == "checkpoints":
            proj = e.command.get("projection") or {}
            assert proj and "checkpoint" not in proj, e.command
    assert hist["d"]["seed"] == ["seed"]
    assert [w[2] for w in hist["d"]["writes"]] == [[i] for i in range(999)]


def test_tc082_history_writes_ordered(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    root = _chain(saver, 2, "ord")
    parent_id = client[db_name]["checkpoints"].find_one(
        {"checkpoint_id": root["configurable"]["checkpoint_id"]}
    )["parent_checkpoint_id"]  # type: ignore[index]
    client[db_name]["checkpoint_writes"].delete_many({"thread_id": "ord"})
    expected = [
        ("", "b", 1),
        ("~a", "a", 0),
        ("~a", "a", 1),
        ("~a", "b", 0),
        ("~b", "c", 0),
    ]
    for path, task, idx in reversed(expected):
        t, v = saver.serde.dumps_typed(f"{path}|{task}|{idx}")
        client[db_name]["checkpoint_writes"].insert_one(
            {
                "thread_id": "ord",
                "checkpoint_ns": "",
                "checkpoint_id": parent_id,
                "task_id": task,
                "task_path": path,
                "idx": idx,
                "channel": "d",
                "type": t,
                "value": v,
            }
        )
    hist = saver.get_delta_channel_history(config=root, channels=["d"])
    assert [w[2] for w in hist["d"]["writes"]] == [
        f"{p}|{t}|{i}" for p, t, i in expected
    ]


def test_tc083_missing_ancestor_raises(
    client: MongoClient, db_name: str, make_saver: Callable[..., MongoDBSaver]
) -> None:
    saver = make_saver()
    target = _chain(saver, 10, "gap")
    ids = [
        d["checkpoint_id"]
        for d in client[db_name]["checkpoints"]
        .find({"thread_id": "gap"})
        .sort("checkpoint_id", 1)
    ]
    client[db_name]["checkpoints"].delete_one({"checkpoint_id": ids[4]})
    # the base implementation silently truncates the walk...
    truncated = base_history(saver, target, ["d"])
    assert "seed" not in truncated["d"]
    # ...the override refuses to return a partial history
    with pytest.raises(CheckpointIntegrityError) as ei:
        saver.get_delta_channel_history(config=target, channels=["d"])
    assert ei.value.missing == [{"checkpoint_id": ids[4]}]
    assert ei.value.checkpoint_id == ids[5]  # the child whose parent is gone
    assert ei.value.thread_id == "gap"


async def test_async_delta_history(make_saver: Callable[..., MongoDBSaver]) -> None:
    saver = make_saver()
    target = _chain(saver, 5, "async")
    got = await saver.aget_delta_channel_history(config=target, channels=["d"])
    assert got == base_history(saver, target, ["d"])
