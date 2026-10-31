"""Tiny helpers shared by the demo_*.py scripts (not tests)."""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from pymongo import MongoClient


@contextmanager
def demo_db() -> Iterator[tuple[MongoClient, str]]:
    client: MongoClient = MongoClient(
        os.environ.get(
            "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
        )
    )
    name = f"ckpt_test_py_demo_{uuid.uuid4().hex[:8]}"
    try:
        yield client, name
    finally:
        client.drop_database(name)
        client.close()
