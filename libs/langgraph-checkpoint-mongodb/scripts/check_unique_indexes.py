# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymongo>=4.18.2,<5",
#   "langgraph-checkpoint-mongodb>=0.6.0",
# ]
# ///

"""Check (and optionally fix) the unique indexes the MongoDB checkpointer needs.

The saver needs these indexes to be **unique** (REQ-E-020):

- ``checkpoints``        ``{thread_id, checkpoint_ns, checkpoint_id}``
- ``checkpoint_writes``  ``{thread_id, checkpoint_ns, checkpoint_id, task_id, idx}``
- ``checkpoint_blobs``   ``{thread_id, checkpoint_ns, channel, version}``
- ``threads``            ``{thread_id}`` (only checked if the collection exists)

Databases created by older versions may hold a *non-unique* index with the same
keys; the saver reports that as ``IndexMigrationRequired`` but never replaces
indexes itself. This script:

- without ``--apply``: reports, per collection, the index state and the number
  of duplicated logical keys. It changes nothing.
- with ``--apply``: if no duplicates exist, converts a non-unique index to
  unique **in place** (``collMod`` ``prepareUnique`` then ``unique``; MongoDB
  6.0+), or creates a missing one. If duplicates exist it changes nothing for
  that collection and exits with status 2. Remove the duplicates first.

The script is idempotent: re-running after a successful ``--apply`` reports
every index as unique and does nothing.

Invoke using PEP 723 (Inline Script Metadata)::

    $ export MONGODB_URI=...   # or pass --mongodb-uri
    $ uv run scripts/check_unique_indexes.py --db checkpointing_db
    $ uv run scripts/check_unique_indexes.py --db checkpointing_db --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from langgraph.checkpoint.mongodb import _core


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--mongodb-uri",
        default=os.environ.get(
            "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
        ),
        help="MongoDB connection URI (default: $MONGODB_URI)",
    )
    p.add_argument("--db", required=True, help="Database holding the collections")
    p.add_argument("--checkpoint-collection", default="checkpoints")
    p.add_argument("--writes-collection", default="checkpoint_writes")
    p.add_argument("--blobs-collection", default="checkpoint_blobs")
    p.add_argument("--threads-collection", default="threads")
    p.add_argument("--apply", action="store_true", help="Create/convert indexes")
    p.add_argument("--json", action="store_true", help="Print a JSON report")
    return p.parse_args(argv)


def count_duplicates(coll: Collection, keys: _core.IndexKeys) -> int:
    """Number of logical keys that occur more than once."""
    group_id = {k.replace(".", "_"): f"${k}" for k, _ in keys}
    res = list(
        coll.aggregate(
            [
                {"$group": {"_id": group_id, "n": {"$sum": 1}}},
                {"$match": {"n": {"$gt": 1}}},
                {"$count": "duplicates"},
            ],
            allowDiskUse=True,
        )
    )
    return int(res[0]["duplicates"]) if res else 0


def check_collection(
    db: Database, name: str, keys: _core.IndexKeys, apply: bool
) -> dict[str, Any]:
    coll = db[name]
    exists = name in db.list_collection_names()
    indexes = list(coll.list_indexes()) if exists else []
    state, ix = _core.find_index(indexes, keys)
    report: dict[str, Any] = {
        "collection": name,
        "keys": [[k, v] for k, v in keys],
        "index": ix["name"] if ix else None,
        "state": state,
        "duplicates": count_duplicates(coll, keys) if exists else 0,
        "action": "none",
    }
    if not apply or state == "unique":
        return report
    if report["duplicates"]:
        report["action"] = "blocked_by_duplicates"
        return report
    if state == "missing":
        report["index"] = coll.create_index(keys, unique=True)
        report["action"] = "created_unique"
    else:
        key_pattern = dict(keys)
        db.command(
            {
                "collMod": name,
                "index": {"keyPattern": key_pattern, "prepareUnique": True},
            }
        )
        db.command(
            {"collMod": name, "index": {"keyPattern": key_pattern, "unique": True}}
        )
        report["action"] = "converted_to_unique"
    report["state"] = "unique"
    return report


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    client: MongoClient = MongoClient(args.mongodb_uri)
    try:
        db = client[args.db]
        targets: list[tuple[str, _core.IndexKeys]] = [
            (args.checkpoint_collection, _core.CHECKPOINTS_INDEX_KEYS),
            (args.writes_collection, _core.WRITES_INDEX_KEYS),
            (args.blobs_collection, _core.BLOBS_INDEX_KEYS),
        ]
        if args.threads_collection in db.list_collection_names():
            targets.append((args.threads_collection, _core.THREADS_INDEX_KEYS))
        reports = [check_collection(db, n, k, args.apply) for n, k in targets]
    finally:
        client.close()
    blocked = any(r["action"] == "blocked_by_duplicates" for r in reports)
    return (2 if blocked else 0), {"db": args.db, "collections": reports}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report = run(args)
    if args.json:
        print(json.dumps(report))
    else:
        for r in report["collections"]:
            print(
                f"{r['collection']}: index={r['index']} state={r['state']} "
                f"duplicates={r['duplicates']} action={r['action']}"
            )
        if code == 2:
            print("Duplicates found: remove them, then re-run with --apply.")
    return code


if __name__ == "__main__":
    sys.exit(main())
