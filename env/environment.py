"""OpenEnv-compliant Code Debugging & Fix environment.

Enhanced with:
  • Full episode history recording for replay / grading
  • Structured logging
  • Robust error handling
  • Difficulty-aware step limits
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .executor import CodeExecutor
from .models import Action, ActionType, Observation, Reward

logger = logging.getLogger("openenv.environment")


class CodeDebugEnv:
    """
    An OpenEnv environment where an AI agent must debug and fix Python code.

    Lifecycle:  ``reset()`` → (``step()`` …) until ``done`` → ``state()`` at any point.
    """

    DIFFICULTY_STEPS = {"easy": 10, "medium": 12, "hard": 15}

    def __init__(self, task_dir: str = "tasks", max_steps: int = 10) -> None:
        self.task_dir = task_dir
        self.default_max_steps = max_steps
        self.max_steps = max_steps
        self.executor = CodeExecutor()
        self.tasks: List[Dict] = self._load_tasks()

        # Episode state
        self.current_task: Optional[Dict] = None
        self.current_code: str = ""
        self.step_count: int = 0
        self.done: bool = True
        self.total_reward: float = 0.0
        self.action_history: List[str] = []
        self.analysis_done: bool = False
        self.episode_info: Dict[str, Any] = {}
        self.episode_history: List[Dict[str, Any]] = []
        self._episode_start: float = 0.0

        if not self.tasks:
            logger.warning("No tasks found in '%s' — environment will fail on reset()", task_dir)
        else:
            logger.info("Environment initialised with %d tasks from '%s'", len(self.tasks), task_dir)

    # ------------------------------------------------------------------ #
    #  Task loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_tasks(self) -> List[Dict]:
        tasks: List[Dict] = []
        for difficulty in ("easy", "medium", "hard"):
            dir_path = os.path.join(self.task_dir, difficulty)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if not fname.endswith((".yaml", ".yml")):
                    continue
                with open(os.path.join(dir_path, fname), encoding="utf-8") as fh:
                    task = yaml.safe_load(fh)
                    task["difficulty"] = difficulty
                    tasks.append(task)
        return tasks

    # ------------------------------------------------------------------ #
    #  OpenEnv interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Observation:
        """Start a new episode and return the initial observation."""
        if not self.tasks:
            raise RuntimeError("No tasks loaded. Check your tasks/ directory.")

        if task_id:
            matches = [t for t in self.tasks if t["id"] == task_id]
            if not matches:
                raise ValueError(f"Task '{task_id}' not found")
            self.current_task = matches[0]
        elif difficulty:
            pool = [t for t in self.tasks if t["difficulty"] == difficulty]
            if not pool:
                raise ValueError(f"No tasks with difficulty '{difficulty}'")
            self.current_task = random.choice(pool)
        else:
            self.current_task = random.choice(self.tasks)

        # Difficulty-aware step limits
        self.max_steps = self.DIFFICULTY_STEPS.get(
            self.current_task["difficulty"], self.default_max_steps
        )

        self.current_code = self.current_task["buggy_code"]
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.action_history = []
        self.analysis_done = False
        self.episode_info = {}
        self.episode_history = []
        self._episode_start = time.time()

        logger.info("Episode reset — task=%s difficulty=%s max_steps=%d",
                     self.current_task["id"], self.current_task["difficulty"], self.max_steps)
        return self._observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute *action* and return ``(observation, reward, done, info)``."""
        if self.current_task is None:
            raise RuntimeError("No episode started. Call reset() first.")
        if self.done:
            raise RuntimeError("Episode finished. Call reset() first.")

        self.step_count += 1
        info: Dict[str, Any] = {
            "action": action.action_type.value,
            "step": self.step_count,
        }

        handler = {
            ActionType.ANALYZE_CODE: self._handle_analyze,
            ActionType.SUGGEST_FIX: lambda: self._handle_suggest_fix(action.payload or ""),
            ActionType.RUN_TESTS: lambda: self._handle_run_tests(info),
            ActionType.SUBMIT_SOLUTION: lambda: self._handle_submit(info),
        }[action.action_type]

        result = handler()
        if isinstance(result, tuple):
            reward, info = result
        else:
            reward = result

        self.total_reward += reward.value
        self.action_history.append(action.action_type.value)

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            info["termination_reason"] = "max_steps_reached"

        # Record episode history
        record = {
            **info,
            "reward": reward.value,
            "reward_reason": reward.reason,
            "cumulative_reward": round(self.total_reward, 4),
            "done": self.done,
            "elapsed": round(time.time() - self._episode_start, 3),
        }
        self.episode_history.append(record)
        self.episode_info = info

        logger.debug("step %d: %s → reward=%.2f done=%s",
                      self.step_count, action.action_type.value, reward.value, self.done)

        return self._observation(), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state (for grading / debugging)."""
        return {
            "task_id": self.current_task["id"] if self.current_task else None,
            "difficulty": self.current_task["difficulty"] if self.current_task else None,
            "buggy_code": self.current_task["buggy_code"] if self.current_task else None,
            "current_code": self.current_code,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "action_history": self.action_history,
            "episode_history": self.episode_history,
        }

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def _observation(self) -> Observation:
        assert self.current_task is not None
        return Observation(
            buggy_code=self.current_task["buggy_code"],
            error_message=self.current_task["error_message"],
            expected_output=self.current_task["expected_output"],
            current_code_state=self.current_code,
            step_count=self.step_count,
            max_steps=self.max_steps,
            task_id=self.current_task["id"],
            difficulty=self.current_task["difficulty"],
        )

    # ------------------------------------------------------------------ #
    #  Action handlers                                                     #
    # ------------------------------------------------------------------ #

    def _handle_analyze(self) -> Reward:
        if self.analysis_done:
            return Reward(value=-0.2, reason="Repeated analysis — no new information")
        self.analysis_done = True
        return Reward(value=0.2, reason="Meaningful code analysis performed")

    def _handle_suggest_fix(self, code: str) -> Reward:
        if not code.strip():
            return Reward(value=-0.2, reason="Empty fix suggestion")
        if code.strip() == self.current_code.strip():
            return Reward(value=-0.2, reason="Suggested code identical to current state")

        syntax = self.executor.check_syntax(code)
        if not syntax["valid"]:
            return Reward(value=-0.2, reason=f"Syntax error in fix: {syntax['error']}")

        self.current_code = code

        test_cases = self.current_task.get("test_cases", [])
        if not test_cases:
            return Reward(value=0.2, reason="Fix applied (no tests to verify)")

        passed = sum(1 for tc in test_cases if self._run_single(code, tc))
        total = len(test_cases)

        if passed == total:
            return Reward(value=1.0, reason=f"Fix passes all {total} tests")
        if passed > 0:
            ratio = passed / total
            return Reward(
                value=round(0.2 + 0.8 * ratio, 2),
                reason=f"Partial fix: {passed}/{total} tests pass",
            )
        return Reward(value=-0.2, reason="Fix does not pass any test")

    def _handle_run_tests(self, info: Dict) -> Tuple[Reward, Dict]:
        test_cases = self.current_task.get("test_cases", [])
        results, passed = self._evaluate_tests(self.current_code, test_cases)
        info["test_results"] = results
        info["tests_passed"] = passed
        info["tests_total"] = len(test_cases)
        return Reward(value=0.1, reason=f"Tests executed: {passed}/{len(test_cases)} passed"), info

    def _handle_submit(self, info: Dict) -> Tuple[Reward, Dict]:
        self.done = True
        test_cases = self.current_task.get("test_cases", [])
        results, passed = self._evaluate_tests(self.current_code, test_cases)
        info["test_results"] = results
        info["tests_passed"] = passed
        info["tests_total"] = len(test_cases)

        if passed == len(test_cases):
            info["termination_reason"] = "successful_fix"
            return Reward(value=5.0, reason="Correct solution — all tests pass"), info

        info["termination_reason"] = "incorrect_submission"
        return Reward(value=-1.0, reason=f"Incorrect: {passed}/{len(test_cases)} tests pass"), info

    # ------------------------------------------------------------------ #
    #  Test helpers                                                        #
    # ------------------------------------------------------------------ #

    def _run_single(self, code: str, tc: Dict) -> bool:
        res = self.executor.execute(code, tc["input"])
        return res["success"] and res["output"] == str(tc["expected"])

    def _evaluate_tests(self, code: str, test_cases: List[Dict]):
        results = []
        passed = 0
        for tc in test_cases:
            res = self.executor.execute(code, tc["input"])
            ok = res["success"] and res.get("output") == str(tc["expected"])
            results.append({
                "input": tc["input"],
                "expected": str(tc["expected"]),
                "actual": res.get("output"),
                "passed": ok,
                "error": res.get("error"),
            })
            if ok:
                passed += 1
        return results, passed
