"""Helpers to run the published langgraph-checkpoint-mongodb 0.5.0 in a temp venv."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
LEGACY_VENV = Path(os.environ.get("CKPT_LEGACY_VENV", "/tmp/ckpt_py_old_venv"))
LEGACY_PYTHON = LEGACY_VENV / "bin" / "python"
LEGACY_REQS = [
    "langgraph-checkpoint-mongodb==0.5.0",
    "langgraph==1.2.11",
    "langgraph-checkpoint==4.2.0",
]


def ensure_legacy_python() -> Path:
    """Create the legacy venv on first use (needs ``uv`` and network)."""
    if LEGACY_PYTHON.exists():
        return LEGACY_PYTHON
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available to build the legacy 0.5.0 venv")
    try:
        subprocess.run(
            [uv, "venv", str(LEGACY_VENV), "--python", "3.11", "-q"], check=True
        )
        subprocess.run(
            [uv, "pip", "install", "-q", "--python", str(LEGACY_PYTHON), *LEGACY_REQS],
            check=True,
        )
    except subprocess.CalledProcessError as e:  # pragma: no cover
        pytest.skip(f"could not build the legacy venv: {e}")
    return LEGACY_PYTHON


def legacy_write(
    factory: str,
    db: str,
    thread: str,
    *,
    input_name: str = "",
    interrupt_before: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run ``graphs.<factory>()`` with the published saver; return its report."""
    python = ensure_legacy_python()
    cmd = [
        str(python),
        "-I",
        str(HERE / "legacy_writer.py"),
        str(HERE / "graphs.py"),
        factory,
        db,
        thread,
    ]
    for node in interrupt_before:
        cmd += ["--interrupt-before", node]
    if input_name:
        cmd += ["--input", input_name]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=dict(os.environ),
        cwd="/tmp",
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["saver_version"] == "0.5.0"
    return report
