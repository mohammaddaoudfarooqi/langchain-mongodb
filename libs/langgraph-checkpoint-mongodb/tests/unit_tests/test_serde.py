import os
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = os.environ.get(
    "MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"
)
DB_NAME = os.environ.get("DB_NAME", "langgraph-test")
COLLECTION_NAME = "serde_checkpoints"


class CustomSerializer(JsonPlusSerializer):
    def __init__(self) -> None:
        super().__init__()
        self.dumps_called = False
        self.loads_called = False

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        self.dumps_called = True
        return super().dumps_typed(obj)

    def loads_typed(self, obj: tuple[str, bytes]) -> Any:
        self.loads_called = True
        return super().loads_typed(obj)


def test_custom_serde(input_data: dict[str, Any]) -> None:
    client: MongoClient = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    db.drop_collection(COLLECTION_NAME)
    client.close()

    custom_serializer = CustomSerializer()

    with MongoDBSaver.from_conn_string(
        MONGODB_URI, DB_NAME, COLLECTION_NAME, serde=custom_serializer
    ) as saver:
        put_config = saver.put(
            input_data["config_1"],
            input_data["chkpnt_1"],
            input_data["metadata_1"],
            {},
        )

        assert custom_serializer.dumps_called

        retrieved_checkpoint_tuple = saver.get_tuple(put_config)

        assert custom_serializer.loads_called

        assert retrieved_checkpoint_tuple is not None
        assert retrieved_checkpoint_tuple.checkpoint == input_data["chkpnt_1"]
        assert retrieved_checkpoint_tuple.metadata == input_data["metadata_1"]


def test_tc012_unserializable_value_raises_and_writes_nothing(
    client: MongoClient, db_name: str
) -> None:
    """REQ-E-012: a serialization failure propagates and nothing is written."""
    import pytest
    from langgraph.checkpoint.base import empty_checkpoint

    loop: dict[str, Any] = {"a": 1}
    loop["self"] = loop
    saver = MongoDBSaver(client, db_name)
    for versions in ({"loop": 1}, {"loop": f"{1:032}.{0.5:016}"}):
        ck = empty_checkpoint()
        ck["channel_values"] = {"loop": loop}
        ck["channel_versions"] = dict(versions)
        with pytest.raises((TypeError, ValueError, RecursionError)):
            saver.put(
                {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                ck,
                {},
                dict(versions),
            )
    with pytest.raises((TypeError, ValueError, RecursionError)):
        saver.put_writes(
            {
                "configurable": {
                    "thread_id": "t",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1",
                }
            },
            [("ok", 1), ("loop", loop)],
            "task",
        )
    for coll in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        assert client[db_name][coll].count_documents({}) == 0
