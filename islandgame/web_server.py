"""Renderer server: serves the Three.js page, broadcasts game events over a websocket,
and offers a manual-play API so a human can play from the browser like in Swift Playgrounds.

The MCP server (a separate process) POSTs events to /event; this process fans them out to browsers.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .engine import run_program, describe_level
from .levels import CAMPAIGN, get_level

from ._env import wandb_project

WANDB_PROJECT = os.environ.get("ISLAND_WANDB_PROJECT") or wandb_project()   # entity/project, "" if no W&B
TRAIN_URL = "https://api.training.wandb.ai/v1/"


def _build_players() -> dict[str, dict]:
    """Players the browser can pick, from .env (see .env.example). Order = order in the dropdown.

    ISLAND_MODEL / ISLAND_BASE_URL   any OpenAI-compatible endpoint: Ollama, vLLM, OpenAI, W&B Inference ...
    ISLAND_CHECKPOINT[_STEP]         a model trained with train/train_rl.py, served by W&B Serverless Training
    WANDB_PROJECT set                adds gpt-oss-20b on W&B Inference as a frozen baseline
    """
    players: dict[str, dict] = {}
    model = os.environ.get("ISLAND_MODEL")
    if model:
        players["custom"] = {"label": os.environ.get("ISLAND_MODEL_LABEL", model), "model": model,
                             "base_url": os.environ.get("ISLAND_BASE_URL") or None, "checkpoint": -1}
    ckpt = os.environ.get("ISLAND_CHECKPOINT")
    if ckpt:
        step = int(os.environ.get("ISLAND_CHECKPOINT_STEP", "0"))
        players["qwen14b-base"] = {"label": "Qwen3-14B, untrained (checkpoint 0)", "model": f"{ckpt}:step0", "base_url": TRAIN_URL, "checkpoint": 0}
        players["qwen14b-rl"] = {"label": f"Qwen3-14B after RL (checkpoint {step})", "model": f"{ckpt}:step{step}", "base_url": TRAIN_URL, "checkpoint": step}
    if WANDB_PROJECT:
        players["gpt-oss-20b"] = {"label": "gpt-oss-20b, frozen (W&B Inference)", "model": "openai/gpt-oss-20b", "base_url": None, "checkpoint": -1}
    return players


AI_PLAYERS = _build_players()
DEFAULT_PLAYER = "qwen14b-rl" if "qwen14b-rl" in AI_PLAYERS else next(iter(AI_PLAYERS), "")
_weave_ready = False
_ai_busy = False


def _ensure_weave() -> bool:
    """Init Weave once, lazily, so the page works even with no W&B login."""
    global _weave_ready
    if _weave_ready:
        return True
    try:
        import weave
        from weave.trace.autopatch import AutopatchSettings, IntegrationSettings
        weave.init(WANDB_PROJECT, autopatch_settings=AutopatchSettings(openai=IntegrationSettings(enabled=False)))
        _weave_ready = True
    except Exception:
        _weave_ready = False
    return _weave_ready

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Island Hopper renderer")
_clients: set[WebSocket] = set()
_last_events: list[dict] = []  # replay the current level to late-joining browsers


async def broadcast(event: dict) -> None:
    if event.get("type") == "level":
        _last_events.clear()
    _last_events.append(event)
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/api/levels")
async def levels() -> JSONResponse:
    return JSONResponse([lv.to_dict() for lv in CAMPAIGN])


@app.get("/api/levels/{level_id}")
async def level(level_id: str) -> JSONResponse:
    lv = get_level(level_id)
    return JSONResponse({"level": lv.to_dict(), "description": describe_level(lv)})


class RunRequest(BaseModel):
    level_id: str
    commands: list[str]
    thought: str = ""
    attempt: int = 1
    source: str = "manual"


@app.post("/api/run")
async def api_run(req: RunRequest) -> JSONResponse:
    lv = get_level(req.level_id)
    result = run_program(lv, req.commands, attempt=req.attempt)
    await broadcast({"type": "level", "level": lv.to_dict(), "attempt": req.attempt, "source": req.source})
    await broadcast({"type": "program", "commands": req.commands, "thought": req.thought, "source": req.source})
    await broadcast({"type": "run", "result": result.to_dict(), "source": req.source})
    return JSONResponse(result.to_dict())


@app.get("/api/players")
async def players() -> JSONResponse:
    return JSONResponse({"default": DEFAULT_PLAYER,
                         "players": [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "base_url"}} for k, v in AI_PLAYERS.items()]})


class AgentPlayRequest(BaseModel):
    level_id: str
    player: str = DEFAULT_PLAYER
    attempts: int = 3


@app.post("/api/agent_play")
async def agent_play(req: AgentPlayRequest) -> JSONResponse:
    """Let a model play the level. Runs in a worker thread; the Player streams events to the renderer itself."""
    global _ai_busy
    if _ai_busy:
        return JSONResponse({"error": "an AI game is already running"}, status_code=409)
    if req.player not in AI_PLAYERS:
        return JSONResponse({"error": f"unknown player {req.player}"}, status_code=400)
    spec = AI_PLAYERS[req.player]
    lv = get_level(req.level_id)
    _ai_busy = True
    await broadcast({"type": "toast", "text": f"{spec['label']} is thinking…", "ms": 2500})

    def run() -> dict:
        from .agent import Player, PlayerConfig, WANDB_INFERENCE_URL
        traced = _ensure_weave()
        cfg = PlayerConfig(model=spec["model"], base_url=spec["base_url"] or WANDB_INFERENCE_URL,
                           wandb_project=WANDB_PROJECT, max_attempts=req.attempts, tag=f"ui-{req.player}",
                           extra={"checkpoint": spec["checkpoint"], "source": "ui"})
        return Player(cfg, trace=traced).play_level(lv)

    async def task() -> None:
        global _ai_busy
        try:
            summary = await asyncio.to_thread(run)
            await broadcast({"type": "toast", "text": ("✅ solved" if summary["solved"] else "❌ not solved") +
                             f" in {summary['attempts']} attempt(s), score {summary['final_score']}", "ms": 4000})
        except Exception as e:
            await broadcast({"type": "toast", "text": f"AI error: {type(e).__name__}: {str(e)[:80]}", "ms": 5000})
        finally:
            _ai_busy = False

    asyncio.create_task(task())
    return JSONResponse({"started": True, "player": spec["label"]})


@app.post("/event")
async def event(ev: dict) -> dict:
    await broadcast(ev)
    return {"ok": True, "clients": len(_clients)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        for ev in _last_events:
            await ws.send_text(json.dumps(ev))
        while True:
            await ws.receive_text()  # keepalive pings from the page; ignored
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


def main() -> None:
    host = os.environ.get("ISLAND_HOST", "127.0.0.1")
    port = int(os.environ.get("ISLAND_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
