# LangGraph Checkpoint MongoDB

Implementation of LangGraph CheckpointSaver that uses MongoDB.

# Installation

```bash
pip install -U langgraph-checkpoint-mongodb
```

## Usage

For more detailed usage examples and documentation, please refer to the [MongoDB LangGraph documentation](https://www.mongodb.com/docs/atlas/ai-integrations/langgraph/).

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "checkpoint_example"

with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Query scalar checkpoint values

In the per-channel layout, the default serializer stores versioned scalar values such as booleans, `None`, finite numbers, and short UTF-8 strings directly in `checkpoint_blobs.value`. The `type` field is BSON null for native values. Checkpoints share a channel document while its version stays unchanged. Pending writes can also store native scalars in `checkpoint_writes.value`.

```javascript
db.checkpoint_blobs.find(
  { thread_id: "my-thread", checkpoint_ns: "", channel: "system_prompt", type: { $type: "null" }, value: "You are a helpful assistant." },
  { _id: 0, version: 1 }
)
```

This query can return historical versions. To check current state, match the returned channel and version against the latest checkpoint's `blob_refs`. The example uses the default collection names. Strings use a UTF-8 byte budget of at most 1024, which a smaller `inline_max_chars` setting lowers. Custom serializers continue to encode values themselves.

Existing checkpoints and serialized pending writes remain readable. The new format does not rewrite historical records. Numeric-only legacy threads must be migrated to use per-channel checkpoint storage. Upgrade all readers before writing native scalars to a shared database. Older savers cannot decode the new scalar records. MongoDB's per-document BSON size limit still applies.

### Async

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "checkpoint_example"

with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```
