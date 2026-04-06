#!/usr/bin/env python3
"""inference.py — Baseline agent for the CodeDebug OpenEnv.

Strict stdout format (no deviation):
  [START] task=<task_name> env=code-debug-fix model=<model_name>
  [STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Required env vars:  API_BASE_URL, MODEL_NAME, HF_TOKEN
Optional:           ENV_URL  (default http://localhost:7860)
                    USE_DIRECT=1  (bypass server, use env directly)
"""

from __future__ import annotations

import os
import sys
import time

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
USE_DIRECT = os.environ.get("USE_DIRECT", "0") == "1"

from openai import OpenAI

_llm = None
_llm_available = True


def _get_llm():
    global _llm, _llm_available
    if _llm is None and _llm_available:
        try:
            _llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "sk-placeholder")
        except Exception:
            _llm_available = False
    return _llm


# ------------------------------------------------------------------ #
#  LLM helpers                                                         #
# ------------------------------------------------------------------ #

def call_llm(messages: list[dict]) -> str:
    """Call LLM with retry + exponential backoff. Returns empty on failure."""
    client = _get_llm()
    if client is None:
        return ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2 ** attempt)
    return ""


def extract_code(text: str) -> str:
    """Strip markdown fences from LLM output."""
    for fence in ("```python", "```py", "```"):
        if fence in text:
            parts = text.split(fence, 1)
            if len(parts) > 1:
                code = parts[1].split("```", 1)[0].strip()
                if code:
                    return code
    return text.strip()


# ------------------------------------------------------------------ #
#  Environment adapters (HTTP vs Direct)                               #
# ------------------------------------------------------------------ #

_direct_env = None


def _get_direct_env():
    global _direct_env
    if _direct_env is None:
        from env.environment import CodeDebugEnv
        _direct_env = CodeDebugEnv()
    return _direct_env


def env_reset(task_id=None, difficulty=None):
    if USE_DIRECT:
        env = _get_direct_env()
        obs = env.reset(task_id=task_id, difficulty=difficulty)
        return obs.model_dump()
    else:
        import requests
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        if difficulty:
            payload["difficulty"] = difficulty
        resp = requests.post(f"{ENV_URL}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["observation"]


def env_step(action_type, payload=None):
    if USE_DIRECT:
        from env.models import Action, ActionType
        env = _get_direct_env()
        action = Action(action_type=ActionType(action_type), payload=payload)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    else:
        import requests
        body = {"action_type": action_type}
        if payload:
            body["payload"] = payload
        resp = requests.post(f"{ENV_URL}/step", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------ #
#  Prompt templates                                                    #
# ------------------------------------------------------------------ #

SYSTEM_ANALYZE = (
    "You are a world-class Python debugging expert. "
    "Identify EVERY bug in the code. For each bug state the line, "
    "what is wrong, and what the fix should be. Be precise."
)

SYSTEM_FIX = (
    "You are a world-class Python debugging expert. "
    "Return ONLY the corrected Python code. "
    "No explanations, no markdown fences, no comments about changes. "
    "Preserve original function signatures. Fix ALL bugs."
)

SYSTEM_REFIX = (
    "Your previous fix was partially correct but some tests still fail. "
    "Examine the failing test results and provide a corrected version. "
    "Return ONLY valid Python code."
)


# ------------------------------------------------------------------ #
#  Logging helpers (strict format)                                     #
# ------------------------------------------------------------------ #

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    err = "null" if error is None else str(error).replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}"
    )


# ------------------------------------------------------------------ #
#  Main agent loop                                                     #
# ------------------------------------------------------------------ #

def run() -> None:
    rewards: list[float] = []
    success = False
    step_count = 0

    try:
        # ── Reset ──
        obs = env_reset()
        task_name = obs["task_id"]

        # [START] — exactly once
        print(f"[START] task={task_name} env=code-debug-fix model={MODEL_NAME}")

        done = False
        analysis_text = ""
        last_test_info: dict = {}
        is_first_iter = True

        while not done:
            # ── 1. Analyse (first iteration only) ──
            if is_first_iter:
                is_first_iter = False
                step_count += 1

                messages = [
                    {"role": "system", "content": SYSTEM_ANALYZE},
                    {"role": "user", "content": (
                        f"Buggy code:\n{obs['buggy_code']}\n"
                        f"Error: {obs['error_message']}\n"
                        f"Expected: {obs['expected_output']}\n"
                        "Identify all bugs."
                    )},
                ]
                analysis_text = call_llm(messages)

                result = env_step("analyze_code")
                rval = result["reward"]["value"]
                done = result["done"]
                obs = result["observation"]
                rewards.append(rval)
                log_step(step_count, "analyze_code", rval, done)
                if done:
                    break

            # ── 2. Suggest fix ──
            step_count += 1

            if len(rewards) <= 1:
                # First fix — full context + analysis
                messages = [
                    {"role": "system", "content": SYSTEM_FIX},
                    {"role": "user", "content": (
                        f"Buggy code:\n{obs['buggy_code']}\n"
                        f"Error: {obs['error_message']}\n"
                        f"Expected: {obs['expected_output']}\n"
                        f"Bug analysis:\n{analysis_text}\n"
                        "Provide the corrected code."
                    )},
                ]
            else:
                # Retry — feed back failing test info
                fail_lines = ""
                for tr in last_test_info.get("test_results", []):
                    if not tr.get("passed"):
                        fail_lines += (
                            f"  Input: {tr['input']}  "
                            f"Expected: {tr['expected']}  "
                            f"Got: {tr.get('actual', 'ERROR')}\n"
                        )
                messages = [
                    {"role": "system", "content": SYSTEM_REFIX},
                    {"role": "user", "content": (
                        f"Current code:\n{obs['current_code_state']}\n"
                        f"Failing tests:\n{fail_lines}\n"
                        "Fix the remaining issues."
                    )},
                ]

            llm_output = call_llm(messages)
            fixed = extract_code(llm_output) if llm_output else ""

            if not fixed:
                # LLM unavailable — submit as-is so episode terminates
                result = env_step("submit_solution")
                rval = result["reward"]["value"]
                done = result["done"]
                rewards.append(rval)
                success = result["info"].get("termination_reason") == "successful_fix"
                log_step(step_count, "submit_solution", rval, done)
                break

            result = env_step("suggest_fix", fixed)
            rval = result["reward"]["value"]
            done = result["done"]
            obs = result["observation"]
            rewards.append(rval)
            log_step(step_count, "suggest_fix", rval, done)
            if done:
                break

            # ── 3. Run tests ──
            step_count += 1
            result = env_step("run_tests")
            rval = result["reward"]["value"]
            done = result["done"]
            last_test_info = result["info"]
            obs = result["observation"]
            rewards.append(rval)
            log_step(step_count, "run_tests", rval, done)
            if done:
                break

            # ── 4. Submit if all tests pass ──
            passed = last_test_info.get("tests_passed", 0)
            total = last_test_info.get("tests_total", 1)

            if passed == total:
                step_count += 1
                result = env_step("submit_solution")
                rval = result["reward"]["value"]
                done = result["done"]
                rewards.append(rval)
                success = result["info"].get("termination_reason") == "successful_fix"
                log_step(step_count, "submit_solution", rval, done)
                break

            # else: loop back for another fix attempt

    except Exception as exc:
        # Guarantee [END] is always printed even on crash
        reward_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success=false steps={len(rewards)} rewards={reward_str}")
        raise

    # Determine success from final reward
    if rewards and rewards[-1] >= 5.0:
        success = True

    # [END] — exactly once
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={reward_str}")


if __name__ == "__main__":
    run()
