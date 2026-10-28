"""Control dashboard: start/stop the bot, pause/flatten, run research, see results.

    uvicorn dashboard.app:app --host 127.0.0.1 --port 8000

Bound to localhost on purpose: it can start a bot that trades real money.
If you run it on a server, put it behind an SSH tunnel or VPN, never open it publicly.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from shadow_bot import control

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
app = FastAPI(title="Shadow Bot Control")

procs: dict[str, subprocess.Popen] = {}


def _alive(name: str) -> bool:
    p = procs.get(name)
    return p is not None and p.poll() is None


def _spawn(name: str, args: list[str]) -> None:
    if _alive(name):
        raise HTTPException(409, f"{name} is already running")
    log = open(DATA / f"{name}.log", "w")
    procs[name] = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
    )


def _stop(name: str, timeout: float = 25.0) -> None:
    p = procs.get(name)
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGINT)  # graceful: strategy flattens on stop
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.terminate()


class BotParams(BaseModel):
    symbol: str = Field("BTCUSDT", pattern=r"^[A-Z0-9]{5,15}$")
    size: float = Field(0.002, gt=0, le=1)
    offset: float = Field(0.30, gt=0, le=5)
    trail: float = Field(0.40, gt=0, le=10)
    grid: float = Field(0.30, gt=0, le=5)
    levels: int = Field(3, ge=1, le=10)
    max_daily_loss: float = Field(20.0, gt=0, le=10_000)
    mainnet: bool = False


class Command(BaseModel):
    cmd: str = Field(pattern=r"^(pause|resume|flatten)$")


class FetchParams(BaseModel):
    symbol: str = Field("BTCUSDT", pattern=r"^[A-Z0-9]{5,15}$")
    days: int = Field(90, ge=1, le=730)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/state")
def state():
    s = control.read_state()
    age = time.time() - s["ts"] if s.get("ts") else None
    return {
        "state": s,
        "state_age_s": age,
        "bot_running": _alive("bot"),
        "sweep_running": _alive("sweep"),
        "fetch_running": _alive("fetch"),
    }


@app.post("/api/command")
def command(c: Command):
    if not _alive("bot"):
        raise HTTPException(409, "bot is not running")
    control.send_command(c.cmd)
    return {"ok": True}


@app.post("/api/bot/start")
def bot_start(p: BotParams):
    args = [
        "run_live.py", "--symbol", p.symbol, "--size", str(p.size),
        "--offset", str(p.offset), "--trail", str(p.trail), "--grid", str(p.grid),
        "--levels", str(p.levels), "--max-daily-loss", str(p.max_daily_loss),
    ]
    if p.mainnet:
        args.append("--mainnet")
    _spawn("bot", args)
    return {"ok": True}


@app.post("/api/bot/stop")
def bot_stop():
    _stop("bot")
    return {"ok": True}


@app.get("/api/logs/{name}")
def logs(name: str):
    if name not in ("bot", "sweep", "fetch"):
        raise HTTPException(404)
    f = DATA / f"{name}.log"
    if not f.exists():
        return {"lines": []}
    return {"lines": f.read_text(errors="replace").splitlines()[-80:]}


@app.post("/api/research/fetch")
def research_fetch(p: FetchParams):
    _spawn("fetch", ["research/fetch_data.py", "--symbol", p.symbol, "--days", str(p.days)])
    return {"ok": True}


@app.post("/api/research/sweep")
def research_sweep(p: FetchParams):
    csv = DATA / f"{p.symbol}_1m.csv"
    if not csv.exists():
        raise HTTPException(400, f"{csv.name} missing - download data first")
    _spawn("sweep", ["research/sweep.py", "--csv", str(csv)])
    return {"ok": True}


@app.get("/api/research/top")
def research_top():
    f = DATA / "sweep_top.json"
    if not f.exists():
        return JSONResponse([])
    return JSONResponse(content=__import__("json").loads(f.read_text()))
