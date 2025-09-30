from __future__ import annotations

import asyncio
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from .simulator import Simulator

app = FastAPI()

_sim: Simulator | None = None
_sim_task: asyncio.Task | None = None
_subscribers: Set[WebSocket] = set()


class Command(BaseModel):
    command: str
    params: Dict[str, Any] = {}


async def _broadcast(event: str, data: Dict[str, Any] | None = None) -> None:
    """Send an event to all connected WebSocket clients."""
    payload = {"event": event}
    if data is not None:
        payload["data"] = data
    for ws in list(_subscribers):
        try:
            await ws.send_json(payload)
        except WebSocketDisconnect:
            _subscribers.discard(ws)


async def _run_simulation() -> None:
    """Background task running the Simulator and broadcasting final metrics."""
    global _sim, _sim_task
    assert _sim is not None
    try:
        await asyncio.to_thread(_sim.run)
        await _broadcast("finished", _sim.get_metrics())
    except Exception as exc:  # pragma: no cover - runtime errors
        await _broadcast("error", {"message": str(exc)})
    finally:
        _sim_task = None


@app.post("/simulations/start")
async def start_simulation(cmd: Command) -> Dict[str, Any]:
    """Start a new simulation with the given parameters."""
    global _sim, _sim_task
    if cmd.command != "start_sim":
        raise HTTPException(status_code=400, detail="Invalid command")
    if _sim_task is not None:
        raise HTTPException(status_code=400, detail="Simulation already running")
    _sim = Simulator(**cmd.params)
    _sim_task = asyncio.create_task(_run_simulation())
    await _broadcast("started", cmd.params)
    return {"status": "started"}


@app.post("/simulations/stop")
async def stop_simulation() -> Dict[str, Any]:
    """Stop the running simulation."""
    global _sim, _sim_task
    if _sim is None or not _sim.running:
        raise HTTPException(status_code=400, detail="No simulation running")
    _sim.stop()
    if _sim_task is not None:
        await _sim_task
    return {"status": "stopped"}


@app.get("/simulations/status")
async def simulation_status() -> Dict[str, Any]:
    """Return the current simulation status and the latest metrics."""
    task = _sim_task
    if task is not None and not task.done():
        status = "running"
    else:
        status = "idle" if _sim is None else "stopped"
    metrics: Dict[str, Any]
    if _sim is not None:
        metrics = _sim.get_metrics()
    else:
        metrics = {}
    return {"status": status, "metrics": metrics}


@app.websocket("/ws")
async def metrics_stream(websocket: WebSocket) -> None:
    """Send real-time metrics over WebSocket."""
    await websocket.accept()
    _subscribers.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            if _sim is not None:
                metrics = _sim.get_metrics()
            else:
                metrics = {}
            await websocket.send_json({"event": "metrics", "data": metrics})
    except WebSocketDisconnect:
        pass
    finally:
        _subscribers.discard(websocket)
