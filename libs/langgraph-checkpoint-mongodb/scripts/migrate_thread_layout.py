# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymongo>=4.18.2,<5",
#   "langgraph-checkpoint-mongodb>=0.6.0",
# ]
# ///

"""Convert old-format checkpointer threads to the per-channel layout (``layout: 2``).

Optional. Threads written by langgraph-checkpoint-mongodb <= 0.5.1 keep working
unchanged in the single-document format (they keep integer channel versions).
Convert a thread when its state is approaching the 16 MiB document limit, or to
get per-channel storage for it.

For every selected thread (all namespaces) the script:

1. converts each old checkpoint's integer channel versions to the string format
   (``"{n:032}.{suffix}"``; the suffix is derived from the value's bytes, so fork
   siblings that shared an integer version get distinct versions and re-runs
   produce the same strings). Ordering inside each checkpoint is preserved;
2. writes the channel blobs (insert-if-absent);
3. writes the converted manifests to ``<checkpoints>_layout2_staging``;
4. verifies, before touching any source document, that every staged manifest
   reassembles from the stored blobs to exactly the original values (on
   failure the staging documents are kept for inspection);
5. copies the source documents to ``<checkpoints>_layout1_backup``;
6. replaces each source document with its staged ``layout: 2`` manifest (in a
   transaction when the deployment supports it);
7. re-reads every checkpoint through ``MongoDBSaver`` and compares it to the
   original. If that fails, the source documents are restored from the
   backup; otherwise the thread's staging documents and backup are removed
   (``--keep-backup`` keeps the backup; ``delete_thread`` and retention never
   touch it).

Pending writes are not changed. Without ``--apply`` nothing is written and the
script reports what it would convert. Re-running on a converted thread is a
no-op. Run it while the selected threads are idle.

Invoke using PEP 723 (Inline Script Metadata)::

    $ export MONGODB_URI=...   # or pass --mongodb-uri
    $ uv run scripts/migrate_thread_layout.py --db checkpointing_db --thread-id T1
    $ uv run scripts/migrate_thread_layout.py --db checkpointing_db --thread-id T1 --apply
    $ uv run scripts/migrate_thread_layout.py --db checkpointing_db --all-threads --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from pymongo import MongoClient, ReplaceOne, UpdateOne

from langgraph.checkpoint.mongodb import MongoDBSaver, _core
from langgraph.checkpoint.mongodb.errors import CheckpointIntegrityError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--mongodb-uri",
        default=os.environ.get(
            "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
        ),
        help="MongoDB connection URI (default: $MONGODB_URI)",
    )
    p.add_argument("--db", required=True)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--thread-id", action="append", help="Thread to convert (repeatable)"
    )
    sel.add_argument(
        "--all-threads", action="store_true", help="Convert every old-format thread"
    )
    p.add_argument("--checkpoint-collection", default="checkpoints")
    p.add_argument("--writes-collection", default="checkpoint_writes")
    p.add_argument("--blobs-collection", default=None)
    p.add_argument("--apply", action="store_true", help="Write the conversion")
    p.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep the thread's backup after success (delete_thread and "
        "retention never remove it)",
    )
    p.add_argument("--json", action="store_true", help="Print a JSON report")
    return p.parse_args(argv)


def _expected(saver: MongoDBSaver, doc: dict[str, Any]) -> dict[str, Any]:
    original = saver.serde.loads_typed((doc["type"], doc["checkpoint"]))
    return _core.convert_versions(saver.serde, original)


def _plan(
    saver: MongoDBSaver, doc: dict[str, Any], converted: dict[str, Any]
) -> _core.PutPlan:
    plan = _core.plan_put(
        serde=saver.serde,
        thread_id=doc["thread_id"],
        checkpoint_ns=doc["checkpoint_ns"],
        checkpoint=converted,
        metadata={},
        parent_checkpoint_id=doc.get("parent_checkpoint_id"),
        new_versions=None,
        names=saver._names,
        inline_max_chars=saver.inline_max_chars,
        size_policy=saver.size_policy,
        created_at=doc.get("created_at"),
    )
    plan.manifest_set["metadata"] = doc["metadata"]  # keep the stored bytes
    return plan


def _verify_staged(
    saver: MongoDBSaver, staged: dict[str, Any] | None, expected: Any
) -> bool:
    """Reassemble a staged manifest from the stored blobs; compare."""
    if staged is None:
        return False
    doc = staged
    refs = staged["blob_refs"]
    blobs = (
        list(
            saver.blobs_collection.find(
                _core.blobs_query(doc["thread_id"], doc["checkpoint_ns"], refs)
            )
        )
        if refs
        else []
    )
    manifest_ck = saver.serde.loads_typed((staged["type"], staged["checkpoint"]))
    try:
        values = _core.load_blob_values(
            saver.serde,
            refs,
            blobs,
            thread_id=doc["thread_id"],
            checkpoint_ns=doc["checkpoint_ns"],
            checkpoint_id=doc["checkpoint_id"],
        )
    except CheckpointIntegrityError:
        return False
    return bool(_core.merge_blobs(manifest_ck, values) == expected)


def _verify_through_saver(
    saver: MongoDBSaver, doc: dict[str, Any], expected: Any
) -> bool:
    try:
        tup = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "checkpoint_ns": doc["checkpoint_ns"],
                    "checkpoint_id": doc["checkpoint_id"],
                }
            }
        )
    except CheckpointIntegrityError:
        return False
    if tup is None:
        return False
    got: dict[str, Any] = dict(tup.checkpoint)
    if _core.needs_pending_sends_migration(expected, doc.get("parent_checkpoint_id")):
        # the read path adds pre-v4 sends; compare the stored part only
        got["channel_values"] = {
            k: v for k, v in got["channel_values"].items() if k != _core.TASKS
        }
        got["channel_versions"] = {
            k: v
            for k, v in got["channel_versions"].items()
            if k in expected["channel_versions"]
        }
    return bool(got == expected)


def migrate_thread(
    saver: MongoDBSaver,
    thread_id: str,
    *,
    apply: bool = False,
    keep_backup: bool = False,
    before_verify: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Convert one thread; returns a report with a ``status``.

    Statuses: ``empty``, ``already_layout2``, ``would_convert``, ``converted``,
    ``verification_failed`` (sources untouched), ``restored`` (post-swap check
    failed; sources restored from the backup).
    """
    coll = saver.checkpoint_collection
    old_docs = [
        d for d in coll.find({"thread_id": thread_id}) if not _core.is_layout2(d)
    ]
    report: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoints": len(old_docs),
        "blobs": 0,
    }
    if not old_docs:
        report["status"] = (
            "already_layout2"
            if coll.count_documents({"thread_id": thread_id})
            else "empty"
        )
        return report

    planned = []
    for doc in old_docs:
        expected = _expected(saver, doc)
        planned.append((doc, expected, _plan(saver, doc, expected)))
    blob_ops = {
        tuple(sorted(b.filter.items())): b for _, _, p in planned for b in p.blob_writes
    }
    report["blobs"] = len(blob_ops)
    if not apply:
        report["status"] = "would_convert"
        return report

    # 1. blobs (insert-if-absent; harmless if we stop later)
    if blob_ops:
        saver.blobs_collection.bulk_write(
            [
                UpdateOne(b.filter, {"$setOnInsert": b.set_on_insert}, upsert=True)
                for b in blob_ops.values()
            ],
            ordered=False,
        )
    # 2. stage the converted manifests
    staging = saver.db[f"{coll.name}_layout2_staging"]
    staging.bulk_write(
        [
            ReplaceOne(
                p.manifest_filter, {**p.manifest_filter, **p.manifest_set}, upsert=True
            )
            for _, _, p in planned
        ]
    )
    if before_verify is not None:
        before_verify()
    # 3. verify the staged manifests + blobs before touching any source document
    staged = {
        (s["checkpoint_ns"], s["checkpoint_id"]): s
        for s in staging.find({"thread_id": thread_id}, {"_id": 0})
    }

    def staged_doc(d: dict[str, Any]) -> dict[str, Any] | None:
        return staged.get((d["checkpoint_ns"], d["checkpoint_id"]))

    if not all(_verify_staged(saver, staged_doc(d), e) for d, e, _ in planned):
        report["status"] = "verification_failed"  # staging kept for inspection
        return report
    # 4. backup, 5. swap
    backup = saver.db[f"{coll.name}_layout1_backup"]
    backup.bulk_write(
        [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d, _, _ in planned]
    )
    swaps = [
        ReplaceOne(
            {"_id": d["_id"], "layout": {"$exists": False}},
            staged[(d["checkpoint_ns"], d["checkpoint_id"])],  # verified above
        )
        for d, _, _ in planned
    ]
    if saver._supports_transactions:
        with saver.client.start_session() as session:
            session.with_transaction(lambda s: coll.bulk_write(swaps, session=s))
    else:
        coll.bulk_write(swaps)
    # 6. verify through the saver; restore on failure
    if not all(_verify_through_saver(saver, d, e) for d, e, _ in planned):
        coll.bulk_write([ReplaceOne({"_id": d["_id"]}, d) for d, _, _ in planned])
        report["status"] = "restored"
        return report
    staging.delete_many({"thread_id": thread_id})
    if not keep_backup:
        backup.delete_many({"_id": {"$in": [d["_id"] for d, _, _ in planned]}})
    report["status"] = "converted"
    return report


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    client: MongoClient = MongoClient(args.mongodb_uri)
    try:
        saver = MongoDBSaver(
            client,
            args.db,
            args.checkpoint_collection,
            args.writes_collection,
            blobs_collection_name=args.blobs_collection,
        )
        if args.all_threads:
            threads = sorted(
                saver.checkpoint_collection.distinct(
                    "thread_id", {"layout": {"$exists": False}}
                )
            )
        else:
            threads = list(dict.fromkeys(args.thread_id))
        reports = [
            migrate_thread(saver, t, apply=args.apply, keep_backup=args.keep_backup)
            for t in threads
        ]
    finally:
        client.close()
    failed = any(r["status"] in ("verification_failed", "restored") for r in reports)
    return (1 if failed else 0), {
        "db": args.db,
        "apply": args.apply,
        "threads": reports,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(args)
    if args.json:
        print(json.dumps(report))
    else:
        for r in report["threads"]:
            print(
                f"{r['thread_id']}: {r['status']} (checkpoints={r['checkpoints']}, blobs={r['blobs']})"
            )
    return code


if __name__ == "__main__":
    sys.exit(main())
