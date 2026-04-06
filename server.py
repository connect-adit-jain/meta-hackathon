"""FastAPI server exposing the CodeDebug OpenEnv over HTTP.

Enhanced with:
  • CORS middleware for cross-origin requests
  • Session-based multi-environment support
  • Structured error handling
  • Task listing and metadata endpoints
  • Episode history endpoint
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import CodeDebugEnv
from env.models import Action, ActionType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
)

app = FastAPI(
    title="CodeDebug OpenEnv",
    version="1.0.0",
    description="AI Code Debugging & Fix — OpenEnv-compliant environment server",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Session management — supports multiple concurrent agents            #
# ------------------------------------------------------------------ #

_sessions: Dict[str, CodeDebugEnv] = {}
_default_env = CodeDebugEnv()


def _get_env(session_id: Optional[str]) -> CodeDebugEnv:
    if not session_id:
        return _default_env
    if session_id not in _sessions:
        _sessions[session_id] = CodeDebugEnv()
    return _sessions[session_id]


# ------------------------------------------------------------------ #
#  Request / response schemas                                          #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    session_id: Optional[str] = Field(default=None, description="Optional session for multi-agent support")


class ActionRequest(BaseModel):
    action_type: str
    payload: Optional[str] = None
    session_id: Optional[str] = None


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/", tags=["meta"])
def root():
    return {
        "status": "running",
        "environment": "CodeDebug OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/history", "/health"],
    }


@app.post("/reset", tags=["environment"])
def reset(request: ResetRequest = ResetRequest()):
    env = _get_env(request.session_id)
    try:
        obs = env.reset(task_id=request.task_id, difficulty=request.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"observation": obs.model_dump(), "session_id": request.session_id}


@app.post("/step", tags=["environment"])
def step(request: ActionRequest):
    env = _get_env(request.session_id)
    try:
        action = Action(
            action_type=ActionType(request.action_type),
            payload=request.payload,
        )
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{request.action_type}'. "
                   f"Valid: {[a.value for a in ActionType]}",
        )
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["environment"])
def state(session_id: Optional[str] = None):
    return _get_env(session_id).state()


@app.get("/tasks", tags=["environment"])
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for t in _default_env.tasks:
        tasks.append({
            "id": t["id"],
            "title": t.get("title", t["id"]),
            "difficulty": t["difficulty"],
            "description": t.get("description", ""),
        })
    return {"tasks": tasks, "total": len(tasks)}


@app.get("/history", tags=["environment"])
def episode_history(session_id: Optional[str] = None):
    """Return the full step-by-step episode history."""
    env = _get_env(session_id)
    return {
        "task_id": env.current_task["id"] if env.current_task else None,
        "done": env.done,
        "total_reward": round(env.total_reward, 4),
        "steps": env.episode_history,
    }


@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "healthy",
        "tasks_loaded": len(_default_env.tasks),
        "active_sessions": len(_sessions),
    }
