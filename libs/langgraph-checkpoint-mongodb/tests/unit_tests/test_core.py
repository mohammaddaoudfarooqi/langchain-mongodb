"""Pure tests of the driver-free core (no server needed)."""

import ast
import random
from pathlib import Path
from typing import Any

import bson
from bson.binary import Binary

from langgraph.checkpoint.mongodb import _core

CORE_PATH = Path(_core.__file__)


def test_core_has_no_pymongo_import() -> None:
    """REQ-NF-005: the core must stay driver-free."""
    tree = ast.parse(CORE_PATH.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "pymongo" not in imported
    assert "motor" not in imported


def _random_value(rng: random.Random, depth: int = 0) -> Any:
    kind = rng.randrange(7 if depth < 4 else 4)
    if kind == 0:
        return rng.randint(-(2**40), 2**40)
    if kind == 1:
        return "".join(rng.choice("abcé☃\u0001") for _ in range(rng.randrange(50)))
    if kind == 2:
        return Binary(rng.randbytes(rng.randrange(2000)))
    if kind == 3:
        return rng.random()
    if kind == 4:
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    return {f"k{i}": _random_value(rng, depth + 1) for i in range(rng.randrange(5))}


def test_tc005_measure_matches_bson_encode() -> None:
    rng = random.Random(1234)
    for _ in range(20):
        doc = {f"f{i}": _random_value(rng) for i in range(rng.randrange(1, 8))}
        assert _core.measure(doc) == len(bson.encode(doc))


# ---------------------------------------------------------------- 5a versions

VERSION_RE = r"^\d{32}\.[0-9.]{16}$"


def test_tc037_next_version_from_none_and_string() -> None:
    import re

    first = _core.next_version(None)
    assert isinstance(first, str)
    assert re.match(VERSION_RE, first)
    assert first.startswith("0" * 31 + "1.")
    current = "00000000000000000000000000000005.0.83751409126543"
    nxt = _core.next_version(current)
    assert re.match(VERSION_RE, nxt)
    assert nxt.split(".")[0] == f"{6:032}"
    assert nxt > current
    # fork-unique: two calls from the same parent version differ
    assert len({_core.next_version(current) for _ in range(50)}) == 50
    # in-memory saver strings (variable-length suffix) are accepted as input
    assert _core.next_version(f"{9:032}.{0.123456789012345678:016}") > f"{9:032}.9"


def test_tc038_next_version_from_number_stays_numeric() -> None:
    assert _core.next_version(5) == 6
    assert _core.next_version(0) == 1
    assert _core.next_version(2.0) == 3.0


def test_golden_version_format_shared_with_js() -> None:
    import re

    v = "00000000000000000000000000000007.0.83751409126543"  # design-delta example
    assert re.match(VERSION_RE, v)
    for _ in range(1000):
        assert re.match(VERSION_RE, _core.next_version(v))


def test_version_kind() -> None:
    assert _core.version_kind({}) == "empty"
    assert _core.version_kind({"a": 1, "b": 2}) == "number"
    assert _core.version_kind({"a": 1, "b": _core.next_version(None)}) == "string"
    assert _core.uses_layout2({}) and _core.uses_layout2({"a": "x"})
    assert not _core.uses_layout2({"a": 3})


# ---------------------------------------------------------------- 5b layout


def test_tc031_classification_table() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    table: list[tuple[Any, str]] = [
        (None, "inline"),
        (True, "inline"),
        (False, "inline"),
        (0, "inline"),
        (-12345678901234567890, "inline"),
        (3.5, "inline"),
        ("", "inline"),
        ("x" * 1024, "inline"),
        ("x" * 1025, "blob"),
        ({}, "blob"),
        ({"a": 1}, "blob"),
        ([], "blob"),
        ([1, 2], "blob"),
        (HumanMessage("hi"), "blob"),
        ([AIMessage("yo")], "blob"),
        (b"bytes", "blob"),
    ]
    for value, expected in table:
        assert _core.classify_channel(value, 1024) == expected, value
    assert _core.classify_channel("x" * 11, 10) == "blob"


def _serde() -> Any:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _checkpoint(values: dict[str, Any], versions: dict[str, Any]) -> Any:
    from langgraph.checkpoint.base import empty_checkpoint

    ck = empty_checkpoint()
    ck["channel_values"] = values
    ck["channel_versions"] = versions
    return ck


def _plan(ck: Any, new_versions: Any) -> Any:
    return _core.plan_put(
        serde=_serde(),
        thread_id="t",
        checkpoint_ns="ns",
        checkpoint=ck,
        metadata={"step": 1},
        parent_checkpoint_id="p",
        new_versions=new_versions,
        names=_core.CollectionNames(
            "checkpoints", "checkpoint_writes", "checkpoint_blobs"
        ),
    )


def test_tc032_blob_docs_only_for_new_versions() -> None:
    v1, v2 = _core.next_version(None), _core.next_version(None)
    ck = _checkpoint(
        {"a": [1], "b": {"x": 1}, "c": "small", "d": "y" * 2000},
        {"a": v1, "b": v2, "c": v1, "d": v2},
    )
    plan = _plan(ck, {"a": v1, "c": v1})
    assert plan.layout == 2
    assert [b.filter["channel"] for b in plan.blob_writes] == ["a"]
    assert plan.blob_writes[0].filter == {
        "thread_id": "t",
        "checkpoint_ns": "ns",
        "channel": "a",
        "version": v1,
    }
    refs = plan.manifest_set["blob_refs"]
    assert refs == [
        {"channel": "a", "version": v1},
        {"channel": "b", "version": v2},
        {"channel": "d", "version": v2},
    ]
    assert plan.manifest_set["inline_channels"] == ["c"]
    assert plan.manifest_set["layout"] == 2


def test_tc033_no_new_versions_means_all_changed() -> None:
    v = _core.next_version(None)
    ck = _checkpoint({"a": [1], "b": {"x": 1}, "c": 1}, {"a": v, "b": v, "c": v})
    plan = _plan(ck, None)
    assert sorted(b.filter["channel"] for b in plan.blob_writes) == ["a", "b"]


def test_empty_channel_and_unversioned_values() -> None:
    v = _core.next_version(None)
    # "gone" has a version but no value -> neither inline nor ref
    ck = _checkpoint({"a": [1], "loose": [2]}, {"a": v, "gone": v})
    plan = _plan(ck, None)
    assert plan.manifest_set["blob_refs"] == [{"channel": "a", "version": v}]
    # a value without a version cannot be referenced, so it stays inline
    assert plan.manifest_set["inline_channels"] == ["loose"]


def test_old_format_plan_is_unchanged_single_document() -> None:
    ck = _checkpoint({"a": [1], "b": "x"}, {"a": 1, "b": 2})
    plan = _plan(ck, {"a": 1})
    assert plan.layout == 1
    assert plan.blob_writes == []
    assert set(plan.manifest_set) == {
        "parent_checkpoint_id",
        "type",
        "checkpoint",
        "metadata",
    }
    assert (
        _serde().loads_typed(
            (plan.manifest_set["type"], plan.manifest_set["checkpoint"])
        )
        == ck
    )


def test_merge_reproduces_channel_values_property() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    rng = random.Random(7)
    serde = _serde()
    pool: list[Any] = [
        None,
        True,
        1,
        2.5,
        "s",
        "z" * 3000,
        [1, 2],
        {"k": [1, {"n": None}]},
        HumanMessage("hi"),
        [AIMessage("a"), HumanMessage("b")],
        b"raw",
    ]
    for _ in range(30):
        n = rng.randrange(0, 8)
        values = {f"c{i}": rng.choice(pool) for i in range(n)}
        versions = {k: _core.next_version(None) for k in values}
        ck = _checkpoint(values, versions)
        plan = _plan(ck, None)
        manifest_ck = serde.loads_typed(
            (plan.manifest_set["type"], plan.manifest_set["checkpoint"])
        )
        blob_docs = [{**b.filter, **b.set_on_insert} for b in plan.blob_writes]
        merged = _core.merge_blobs(
            manifest_ck,
            _core.load_blob_values(
                serde,
                plan.manifest_set["blob_refs"],
                blob_docs,
                thread_id="t",
                checkpoint_ns="ns",
                checkpoint_id=ck["id"],
            ),
        )
        assert merged["channel_values"] == values
        assert merged == ck


def test_missing_blob_raises_integrity_error() -> None:
    from langgraph.checkpoint.mongodb import CheckpointIntegrityError

    v = _core.next_version(None)
    refs = [{"channel": "a", "version": v}, {"channel": "b", "version": v}]
    t, data = _serde().dumps_typed([1])
    docs = [{"channel": "a", "version": v, "type": t, "value": data}]
    try:
        _core.load_blob_values(
            _serde(), refs, docs, thread_id="t", checkpoint_ns="", checkpoint_id="c"
        )
    except CheckpointIntegrityError as e:
        assert e.missing == [{"channel": "b", "version": v}]
        assert (e.thread_id, e.checkpoint_ns, e.checkpoint_id) == ("t", "", "c")
        assert e.code == "CheckpointIntegrityError"
    else:  # pragma: no cover
        raise AssertionError("expected CheckpointIntegrityError")


def test_non_mapping_channel_values_kept_verbatim() -> None:
    ck = _checkpoint("not-a-dict", {})  # type: ignore[arg-type]
    plan = _plan(ck, {})
    assert plan.layout == 2 and plan.blob_writes == []
    assert (
        plan.manifest_set["blob_refs"] == []
        and plan.manifest_set["inline_channels"] == []
    )
    stored = _serde().loads_typed(
        (plan.manifest_set["type"], plan.manifest_set["checkpoint"])
    )
    assert _core.merge_blobs(stored, {}) == ck
