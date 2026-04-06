"""Typed Pydantic models for the OpenEnv Code Debugging environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Valid action types for the environment."""
    ANALYZE_CODE = "analyze_code"
    SUGGEST_FIX = "suggest_fix"
    RUN_TESTS = "run_tests"
    SUBMIT_SOLUTION = "submit_solution"


class Action(BaseModel):
    """An action taken by the agent."""
    action_type: ActionType
    payload: Optional[str] = Field(
        default=None,
        description="For suggest_fix actions, the proposed corrected code.",
    )


class Observation(BaseModel):
    """Observable state returned to the agent after each step."""
    buggy_code: str = Field(description="Original buggy code snippet")
    error_message: str = Field(description="Error or symptom description")
    expected_output: str = Field(description="What the correct code should produce")
    current_code_state: str = Field(description="Latest version of the code")
    step_count: int = Field(description="Steps taken so far")
    max_steps: int = Field(description="Maximum steps allowed")
    test_results: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Results from the last run_tests action"
    )
    task_id: str = Field(description="Unique task identifier")
    difficulty: str = Field(description="easy | medium | hard")


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(description="Numeric reward value")
    reason: str = Field(description="Human-readable explanation")


class StepResult(BaseModel):
    """Full return tuple from env.step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]
