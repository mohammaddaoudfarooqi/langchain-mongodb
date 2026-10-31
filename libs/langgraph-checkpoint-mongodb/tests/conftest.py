"""Shared fixtures for tests added with the per-channel storage work.

Every test that uses ``db_name`` gets its own database whose name starts with
``ckpt_test_py_``; it is dropped on teardown. This keeps tests isolated and
safe to run against a shared cluster.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pymongo import MongoClient, monitoring

from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)
TEST_DB_PREFIX = "ckpt_test_py_"


@pytest.fixture
def client() -> Iterator[MongoClient]:
    c: MongoClient = MongoClient(MONGODB_URI)
    yield c
    c.close()


@pytest.fixture
def db_name(client: MongoClient) -> Iterator[str]:
    name = f"{TEST_DB_PREFIX}{uuid.uuid4().hex[:12]}"
    yield name
    client.drop_database(name)


@pytest.fixture
def make_saver(client: MongoClient, db_name: str) -> Callable[..., MongoDBSaver]:
    """Return a factory producing a *new* saver instance on every call."""

    def factory(**kwargs: Any) -> MongoDBSaver:
        return MongoDBSaver(client, db_name, **kwargs)

    return factory


class CommandRecorder(monitoring.CommandListener):
    """Records started commands (name, database, command document)."""

    def __init__(self) -> None:
        self.events: list[monitoring.CommandStartedEvent] = []

    def started(self, event: monitoring.CommandStartedEvent) -> None:
        self.events.append(event)

    def succeeded(self, event: monitoring.CommandSucceededEvent) -> None:
        pass

    def failed(self, event: monitoring.CommandFailedEvent) -> None:
        pass

    def names(self) -> list[str]:
        return [e.command_name for e in self.events]

    def clear(self) -> None:
        self.events.clear()


@pytest.fixture
def recorder() -> CommandRecorder:
    return CommandRecorder()


@pytest.fixture
def monitored_client(recorder: CommandRecorder) -> Iterator[MongoClient]:
    c: MongoClient = MongoClient(MONGODB_URI, event_listeners=[recorder])
    yield c
    c.close()


@pytest.fixture(scope="session")
def max_wire_version() -> int:
    c: MongoClient = MongoClient(MONGODB_URI)
    try:
        return int(c.admin.command("hello").get("maxWireVersion", 0))
    finally:
        c.close()
