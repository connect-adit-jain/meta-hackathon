"""
FastAPI application for the Blood Bank Environment.

Exposes the BloodBankEnvironment over HTTP endpoints:
  POST /reset   → reset the environment, return initial observation
  POST /step    → execute an action, return (observation, reward, done, info)
  GET  /state   → return current episode state
  GET  /health  → health-check endpoint
  POST /grade   → return the graded score for the current/specified task

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from server.environment import BloodBankEnvironment
    from models import BloodBankAction
except ImportError:
    from .environment import BloodBankEnvironment
    from ..models import BloodBankAction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloodbank")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BloodBank Environment",
    description="Hospital blood bank inventory & allocation RL environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session for simplicity)
env = BloodBankEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_name: str = "basic_compatibility"
    max_steps: int = 30


class ActionRequest(BaseModel):
    request_id: int = -1
    donor_blood_type: str = "O+"
    units_to_allocate: int = 1
    skip: bool = False


class GradeRequest(BaseModel):
    task_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health-check endpoint."""
    return {"status": "ok", "environment": "bloodbank"}


@app.post("/reset")
async def reset(body: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial observation."""
    try:
        obs = env.reset(
            seed=body.seed,
            task_name=body.task_name,
            max_steps=body.max_steps,
        )
        return {
            "observation": asdict(obs),
            "info": {"episode_id": env.state.episode_id},
        }
    except Exception as e:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(body: ActionRequest):
    """Execute an action and return the result."""
    try:
        action = BloodBankAction(
            request_id=body.request_id,
            donor_blood_type=body.donor_blood_type,
            units_to_allocate=body.units_to_allocate,
            skip=body.skip,
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": asdict(obs),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        logger.exception("Step failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Return the current episode state."""
    return asdict(env.state)


@app.post("/grade")
async def grade(body: GradeRequest = GradeRequest()):
    """Return the graded score for the current or specified task."""
    try:
        score = env.grade(task_name=body.task_name)
        return {"task": body.task_name or env.state.task_name, "score": score}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
