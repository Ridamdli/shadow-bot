"""Shared state/command channel between the bot process and the dashboard.

The bot writes data/state.json every couple of seconds.
The dashboard writes data/control.json; the bot polls it.
Files are written atomically so readers never see half-written JSON.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
STATE = DATA / "state.json"
CONTROL = DATA / "control.json"


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(state: dict) -> None:
    state["ts"] = time.time()
    _atomic_write(STATE, state)


def read_state() -> dict:
    return _read(STATE)


def send_command(cmd: str) -> None:
    """cmd: pause | resume | flatten"""
    _atomic_write(CONTROL, {"cmd": cmd, "id": time.time_ns()})


def read_command() -> tuple[str | None, int]:
    c = _read(CONTROL)
    return c.get("cmd"), int(c.get("id", 0))
