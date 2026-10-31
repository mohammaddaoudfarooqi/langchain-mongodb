"""Operational scripts: check_unique_indexes (TC-022) and migrate_thread_layout (TC-042)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
CHECKPOINT_KEYS = [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)]


def run_script(name: str, *args: str) -> tuple[int, Any, str]:
    """Run a script as a subprocess; the URI is passed through the environment."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args, "--json"],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=300,
    )
    out = json.loads(proc.stdout) if proc.stdout.strip() else None
    return proc.returncode, out, proc.stderr


def _by_collection(report: Any) -> dict[str, dict[str, Any]]:
    return {r["collection"]: r for r in report["collections"]}


def test_tc022_reports_duplicates_and_changes_nothing(
    client: MongoClient, db_name: str
) -> None:
    coll = client[db_name]["checkpoints"]
    coll.create_index(CHECKPOINT_KEYS, name="legacy_idx")
    row = {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}
    coll.insert_many([dict(row), dict(row), {**row, "checkpoint_id": "2"}])
    before = list(coll.list_indexes())

    code, report, err = run_script("check_unique_indexes.py", "--db", db_name)
    assert code == 0, err
    r = _by_collection(report)["checkpoints"]
    assert r["state"] == "non_unique"
    assert r["index"] == "legacy_idx"
    assert r["duplicates"] == 1
    assert r["action"] == "none"
    assert list(coll.list_indexes()) == before
    assert coll.count_documents({}) == 3

    # --apply refuses while duplicates exist
    code, report, err = run_script(
        "check_unique_indexes.py", "--db", db_name, "--apply"
    )
    assert code == 2, err
    assert list(coll.list_indexes()) == before


def test_tc022_apply_rebuilds_index_as_unique_and_is_idempotent(
    client: MongoClient, db_name: str
) -> None:
    coll = client[db_name]["checkpoints"]
    coll.create_index(CHECKPOINT_KEYS, name="legacy_idx")
    coll.insert_one({"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"})

    code, report, err = run_script(
        "check_unique_indexes.py", "--db", db_name, "--apply"
    )
    assert code == 0, err
    assert _by_collection(report)["checkpoints"]["action"] == "converted_to_unique"
    ix = {tuple(i["key"].items()): i for i in coll.list_indexes()}
    assert ix[tuple(CHECKPOINT_KEYS)].get("unique") is True
    # missing indexes on the other collections are created
    for name in ("checkpoint_writes", "checkpoint_blobs"):
        assert any(i.get("unique") for i in client[db_name][name].list_indexes())

    code, report, err = run_script(
        "check_unique_indexes.py", "--db", db_name, "--apply"
    )
    assert code == 0, err
    assert {r["action"] for r in report["collections"]} == {"none"}
    assert {r["state"] for r in report["collections"]} == {"unique"}


# ------------------------------------------------------------------ TC-042

import importlib.util  # noqa: E402
import operator  # noqa: E402
from typing import Annotated, TypedDict  # noqa: E402

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from langgraph.checkpoint.mongodb import MongoDBSaver  # noqa: E402


def _load_migrate_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "migrate_thread_layout", SCRIPTS / "migrate_thread_layout.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IntVersionSaver(MongoDBSaver):
    """Writes integer versions, i.e. exactly what releases <= 0.5.0 stored."""

    def get_next_version(self, current: Any, channel: None = None) -> Any:
        return BaseCheckpointSaver.get_next_version(self, current, channel)


class S(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    n: int
    blob: dict
    log: Annotated[list[str], operator.add]


def _graph() -> StateGraph:
    def step(state: S) -> dict:
        n = state.get("n", 0) + 1
        return {
            "messages": [AIMessage(f"m{n}", id=f"m{n}")],
            "n": n,
            "blob": {"n": n, "pad": "p" * 2000},
            "log": [f"s{n}"],
        }

    g = StateGraph(S)
    g.add_node("step", step)
    g.add_node("end", lambda s: {"log": ["end"]})
    g.add_edge(START, "step")
    g.add_conditional_edges("step", lambda s: "step" if s["n"] < 4 else "end")
    g.add_edge("end", END)
    return g


def _write_old_thread(client: MongoClient, db: str, thread: str) -> None:
    g = _graph().compile(
        checkpointer=IntVersionSaver(client, db), interrupt_before=["end"]
    )
    cfg: Any = {"configurable": {"thread_id": thread}}
    g.invoke({"messages": [HumanMessage("hi", id="h")]}, cfg)
    # a fork: two update_state branches from the same ancestor, same int version
    saver = IntVersionSaver(client, db)
    hist = list(_graph().compile(checkpointer=saver).get_state_history(cfg))
    ancestor = hist[2].config
    _graph().compile(checkpointer=saver).update_state(ancestor, {"blob": {"side": "L"}})


def _history(client: MongoClient, db: str, thread: str) -> list[Any]:
    g = _graph().compile(checkpointer=MongoDBSaver(client, db))
    cfg: Any = {"configurable": {"thread_id": thread}}
    return [
        (s.config["configurable"]["checkpoint_id"], s.values, s.next)
        for s in g.get_state_history(cfg)
    ]


def test_tc042_migration_converts_and_resumes_identically(
    client: MongoClient, db_name: str
) -> None:
    mig = _load_migrate_module()
    for t in ("a", "b"):
        _write_old_thread(client, db_name, t)
    coll = client[db_name]["checkpoints"]
    before = _history(client, db_name, "a")

    # dry run changes nothing
    code, report, err = run_script(
        "migrate_thread_layout.py", "--db", db_name, "--thread-id", "a"
    )
    assert code == 0, err
    assert report["threads"][0]["status"] == "would_convert"
    assert all("layout" not in d for d in coll.find({"thread_id": "a"}))

    code, report, err = run_script(
        "migrate_thread_layout.py",
        "--db",
        db_name,
        "--thread-id",
        "a",
        "--apply",
        "--keep-backup",
    )
    assert code == 0, err
    r = report["threads"][0]
    assert r["status"] == "converted" and r["checkpoints"] == coll.count_documents(
        {"thread_id": "a"}
    )
    docs = list(coll.find({"thread_id": "a"}))
    assert docs and all(d["layout"] == 2 for d in docs)
    assert client[db_name]["checkpoint_blobs"].count_documents({"thread_id": "a"}) > 0
    # originals backed up; staging cleaned up after success
    backup = client[db_name]["checkpoints_layout1_backup"]
    assert backup.count_documents({"thread_id": "a"}) == len(docs)
    assert all("layout" not in d for d in backup.find({"thread_id": "a"}))
    staging = client[db_name]["checkpoints_layout2_staging"]
    assert staging.count_documents({"thread_id": "a"}) == 0
    # the "b" thread was not selected and is untouched
    assert all("layout" not in d for d in coll.find({"thread_id": "b"}))

    # reads are identical, versions became strings
    assert _history(client, db_name, "a") == before
    tup = MongoDBSaver(client, db_name).get_tuple({"configurable": {"thread_id": "a"}})
    assert tup and all(
        isinstance(v, str) for v in tup.checkpoint["channel_versions"].values()
    )

    # resume after conversion == resume of the unconverted twin
    cfg_a: Any = {"configurable": {"thread_id": "a"}}
    cfg_b: Any = {"configurable": {"thread_id": "b"}}
    out_a = (
        _graph().compile(checkpointer=MongoDBSaver(client, db_name)).invoke(None, cfg_a)
    )
    out_b = (
        _graph().compile(checkpointer=MongoDBSaver(client, db_name)).invoke(None, cfg_b)
    )
    assert out_a == out_b and out_a["log"][-1] == "end"
    assert all(d["layout"] == 2 for d in coll.find({"thread_id": "a"}))

    # re-running is a no-op
    n_blobs = client[db_name]["checkpoint_blobs"].count_documents({})
    code, report, err = run_script(
        "migrate_thread_layout.py", "--db", db_name, "--thread-id", "a", "--apply"
    )
    assert code == 0, err
    assert report["threads"][0]["status"] == "already_layout2"
    assert client[db_name]["checkpoint_blobs"].count_documents({}) == n_blobs
    del mig


def test_tc042_sources_untouched_until_verified(
    client: MongoClient, db_name: str
) -> None:
    mig = _load_migrate_module()
    _write_old_thread(client, db_name, "a")
    coll = client[db_name]["checkpoints"]
    originals = {d["_id"]: d for d in coll.find({"thread_id": "a"})}

    def sabotage() -> None:  # lose a blob between writing and verification
        client[db_name]["checkpoint_blobs"].delete_one(
            {"thread_id": "a", "channel": "blob"}
        )

    saver = MongoDBSaver(client, db_name)
    result = mig.migrate_thread(saver, "a", apply=True, before_verify=sabotage)
    assert result["status"] == "verification_failed"
    assert {d["_id"]: d for d in coll.find({"thread_id": "a"})} == originals
    # the converted manifests were staged (and are kept for inspection)
    staged = client[db_name]["checkpoints_layout2_staging"]
    assert staged.count_documents({"thread_id": "a"}) == len(originals)
    assert all(d["layout"] == 2 for d in staged.find({"thread_id": "a"}))
    # a clean re-run then succeeds
    assert mig.migrate_thread(saver, "a", apply=True)["status"] == "converted"
    assert all(d["layout"] == 2 for d in coll.find({"thread_id": "a"}))


def test_tc042_fork_versions_do_not_collide(client: MongoClient, db_name: str) -> None:
    """Old fork siblings share integer versions; converted versions must not."""
    mig = _load_migrate_module()
    _write_old_thread(client, db_name, "a")
    before = _history(client, db_name, "a")
    assert (
        mig.migrate_thread(MongoDBSaver(client, db_name), "a", apply=True)["status"]
        == "converted"
    )
    assert _history(client, db_name, "a") == before
    blobs = list(
        client[db_name]["checkpoint_blobs"].find({"thread_id": "a", "channel": "blob"})
    )
    assert len({b["version"] for b in blobs}) == len(blobs)


def test_tc042_all_threads_flag(client: MongoClient, db_name: str) -> None:
    for t in ("a", "b"):
        _write_old_thread(client, db_name, t)
    code, report, err = run_script(
        "migrate_thread_layout.py", "--db", db_name, "--all-threads", "--apply"
    )
    assert code == 0, err
    assert sorted((r["thread_id"], r["status"]) for r in report["threads"]) == [
        ("a", "converted"),
        ("b", "converted"),
    ]


@pytest.mark.parametrize("bad", [["--db", "x"], ["--thread-id", "a"]])
def test_migration_cli_requires_selection(bad: list[str]) -> None:
    code, _, _ = run_script("migrate_thread_layout.py", *bad)
    assert code != 0


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scripts_in_process_text_output(
    client: MongoClient, db_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    check = _load_script("check_unique_indexes")
    mig = _load_script("migrate_thread_layout")
    coll = client[db_name]["checkpoints"]
    coll.create_index(CHECKPOINT_KEYS, name="legacy_idx")
    row = {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}
    coll.insert_many([dict(row), dict(row)])
    assert check.main(["--db", db_name]) == 0
    assert check.main(["--db", db_name, "--apply"]) == 2
    out = capsys.readouterr().out
    assert "state=non_unique duplicates=1" in out and "Duplicates found" in out
    coll.delete_many({})
    assert check.main(["--db", db_name, "--apply"]) == 0
    assert "action=converted_to_unique" in capsys.readouterr().out

    coll.drop()
    _write_old_thread(client, db_name, "a")
    assert mig.main(["--db", db_name, "--thread-id", "a", "--apply"]) == 0
    assert "a: converted" in capsys.readouterr().out
    assert client[db_name]["checkpoints_layout1_backup"].count_documents({}) == 0
    assert mig.main(["--db", db_name, "--thread-id", "nope"]) == 0
    assert "nope: empty" in capsys.readouterr().out


def test_tc042_post_swap_verification_failure_restores_sources(
    client: MongoClient, db_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mig = _load_script("migrate_thread_layout")
    _write_old_thread(client, db_name, "a")
    coll = client[db_name]["checkpoints"]
    originals = {d["_id"]: d for d in coll.find({"thread_id": "a"})}
    monkeypatch.setattr(mig, "_verify_through_saver", lambda *a: False)
    result = mig.migrate_thread(MongoDBSaver(client, db_name), "a", apply=True)
    assert result["status"] == "restored"
    assert {d["_id"]: d for d in coll.find({"thread_id": "a"})} == originals
    backup = client[db_name]["checkpoints_layout1_backup"]
    assert backup.count_documents({"thread_id": "a"}) == len(originals)
