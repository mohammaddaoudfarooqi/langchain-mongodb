"""Write a thread with the *published* saver (run under a legacy venv).

Usage: python -I legacy_writer.py <graphs.py path> <factory> <db> <thread_id>
       [--interrupt-before NODE]   (MONGODB_URI from the environment)
Prints the installed saver version and the interrupt state as JSON.
"""

import argparse
import importlib.util
import json
import os
from importlib.metadata import version
from typing import Any

from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

p = argparse.ArgumentParser()
p.add_argument("graphs")
p.add_argument("factory")
p.add_argument("db")
p.add_argument("thread")
p.add_argument("--interrupt-before", action="append", default=[])
p.add_argument("--input", default="")
args = p.parse_args()

spec = importlib.util.spec_from_file_location("graphs", args.graphs)
assert spec and spec.loader
graphs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphs)

client: MongoClient = MongoClient(os.environ["MONGODB_URI"])
saver = MongoDBSaver(client, args.db)
graph = getattr(graphs, args.factory)().compile(
    checkpointer=saver, interrupt_before=args.interrupt_before or None
)
cfg: Any = {"configurable": {"thread_id": args.thread}}
inp = getattr(graphs, args.input) if args.input else None
graph.invoke(inp, cfg)
state = graph.get_state(cfg)
print(
    json.dumps(
        {
            "saver_version": version("langgraph-checkpoint-mongodb"),
            "next": list(state.next),
            "versions": saver.get_tuple(cfg).checkpoint["channel_versions"],  # type: ignore[union-attr]
        }
    )
)
client.close()
