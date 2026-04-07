#!/usr/bin/env python3
"""
inference.py – BloodBank Environment Inference Script

Runs an LLM-based agent against the BloodBank environment server and emits
standardised stdout logs in the mandated format:

    [START] task=<name> env=bloodbank model=<model>
    [STEP]  step=<n> action=<json> reward=XX.XX done=true/false error=.../null
    [END]   success=true/false steps=<n> score=XX.XXX rewards=0.00,0.00,...

Environment Variables:
    API_BASE_URL   – Chat completions endpoint (see examples below)
    MODEL_NAME     – Model identifier (see examples below)
    API_KEY        – Your API key for the chosen provider
    ENV_URL        – BloodBank server URL (default http://localhost:7860)
    TASK_NAME      – Task to run (default "basic_compatibility")
    MAX_STEPS      – Maximum steps per episode (default 30)

Supported Providers (examples):
    ┌──────────┬──────────────────────────────────────────────────────────┬─────────────────────────┐
    │ Provider │ API_BASE_URL                                           │ MODEL_NAME              │
    ├──────────┼──────────────────────────────────────────────────────────┼─────────────────────────┤
    │ Gemini   │ https://generativelanguage.googleapis.com/v1beta/openai/│ gemini-2.0-flash        │
    │ Claude   │ https://api.anthropic.com/v1/                          │ claude-sonnet-4-20250514│
    │ OpenAI   │ https://api.openai.com/v1                              │ gpt-4o-mini             │
    │ MiniMax  │ https://api.minimax.chat/v1                            │ MiniMax-Text-01         │
    │ Groq     │ https://api.groq.com/openai/v1                        │ llama-3.3-70b-versatile │
    │ Together │ https://api.together.xyz/v1                            │ meta-llama/Llama-3-70b  │
    └──────────┴──────────────────────────────────────────────────────────┴─────────────────────────┘

    Just set API_KEY, API_BASE_URL, and MODEL_NAME for your provider. That's it.

Designed to run within 20 minutes on 2 vCPU / 8 GB RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash")

# Generic API_KEY – works with ANY provider.
# Also checks legacy OPENAI_API_KEY and HF_TOKEN as fallbacks for compatibility.
API_KEY = (
    os.environ.get("API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("CLAUDE_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or ""
)

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASK_NAME = os.environ.get("TASK_NAME", "basic_compatibility")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "30"))
TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Logging helpers – exact mandated format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert hospital blood bank manager AI agent.

Your job is to manage a blood bank: allocate blood units to transfusion requests
while ensuring ABO/Rh compatibility, minimising wastage from expired units, and
prioritising emergency requests.

BLOOD TYPE COMPATIBILITY (donor → recipient):
- O- can donate to ALL types (universal donor)
- O+ can donate to O+, A+, B+, AB+
- A+ can donate to A+, AB+
- A- can donate to A+, A-, AB+, AB-
- B+ can donate to B+, AB+
- B- can donate to B+, B-, AB+, AB-
- AB+ can donate to AB+ only
- AB- can donate to AB+, AB- only

GUIDELINES:
1. ALWAYS ensure blood type compatibility (mismatches are fatal).
2. Prioritise EMERGENCY requests (priority=2) over URGENT (1) and ROUTINE (0).
3. Use near-expiry units first (FEFO: First-Expiry First-Out).
4. Maintain balanced stock across all blood types.
5. Only skip if there truly is no beneficial action available.

You will receive the current observation as JSON and must respond with ONLY
a valid JSON action object. No explanation, no markdown, just raw JSON.

Action format:
{
    "request_id": <int>,           // ID of the request to fulfil
    "donor_blood_type": "<str>",   // blood type to allocate from inventory
    "units_to_allocate": <int>,    // number of units (1-4)
    "skip": false                  // set true only if no good action exists
}
"""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    """Build a concise prompt from the observation."""
    inv = observation.get("inventory", {})
    reqs = observation.get("pending_requests", [])

    inv_summary = []
    for bt, expiries in inv.items():
        if expiries:
            inv_summary.append(f"  {bt}: {len(expiries)} units (expiry days: {expiries[:5]}{'...' if len(expiries) > 5 else ''})")
        else:
            inv_summary.append(f"  {bt}: 0 units")

    req_summary = []
    for r in sorted(reqs, key=lambda x: -x.get("priority", 0)):
        req_summary.append(
            f"  #{r['request_id']}: {r['blood_type']} × {r['units_needed']} "
            f"[{r.get('priority_label', 'ROUTINE')}] (waiting {r.get('age_steps', 0)} steps)"
        )

    donation = observation.get("donation_event")
    donation_line = f"\n🩸 DONATION EVENT: {donation}" if donation else ""

    prompt = f"""STEP {observation.get('step_number', '?')} — Blood Bank Status:

INVENTORY ({observation.get('total_units', 0)} total, {observation.get('units_expiring_soon', 0)} expiring within 3 days):
{chr(10).join(inv_summary)}

PENDING REQUESTS ({observation.get('total_requests', 0)} total):
{chr(10).join(req_summary) if req_summary else '  (none)'}
{donation_line}
Previous reward: {observation.get('reward', 0.0):.2f}
Message: {observation.get('message', '')}

Respond with a single JSON action object. Prioritise emergencies and use near-expiry stock."""

    return prompt


def parse_llm_action(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dict. Robust to markdown fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    # Fallback: skip
    return {"skip": True}


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    """Run one episode of the BloodBank environment with the LLM agent."""

    if not API_KEY:
        print(
            "ERROR: No API key found. Set one of these environment variables:\n"
            "  API_KEY        (generic, works with any provider)\n"
            "  GEMINI_API_KEY (for Google Gemini)\n"
            "  CLAUDE_API_KEY (for Anthropic Claude)\n"
            "  OPENAI_API_KEY (for OpenAI)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialise LLM client – the OpenAI Python library supports any
    # OpenAI-compatible endpoint (Gemini, Claude, Groq, Together, etc.)
    llm = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    # Create HTTP client for the environment
    env = httpx.Client(base_url=ENV_URL, timeout=30.0)

    rewards: List[float] = []
    success = False
    steps = 0
    score = 0.0

    try:
        log_start(task=TASK_NAME, env="bloodbank", model=MODEL_NAME)

        # Reset environment
        reset_resp = env.post("/reset", json={
            "task_name": TASK_NAME,
            "max_steps": MAX_STEPS,
        })
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        observation = reset_data["observation"]
        done = False

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            steps = step_num
            error_msg = None

            try:
                # Ask LLM for action
                user_prompt = build_user_prompt(observation)
                chat_resp = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=200,
                )
                raw_action = chat_resp.choices[0].message.content or '{"skip": true}'
                action_dict = parse_llm_action(raw_action)

                # Send action to environment
                step_resp = env.post("/step", json=action_dict)
                step_resp.raise_for_status()
                step_data = step_resp.json()

                observation = step_data["observation"]
                reward = step_data["reward"]
                done = step_data["done"]

            except Exception as e:
                error_msg = str(e)
                reward = 0.0
                action_dict = {"skip": True, "error": error_msg}
                done = False

            rewards.append(reward)
            action_json = json.dumps(action_dict, separators=(",", ":"))
            log_step(
                step=step_num,
                action=action_json,
                reward=reward,
                done=done,
                error=error_msg,
            )

        # Grade the episode
        try:
            grade_resp = env.post("/grade", json={"task_name": TASK_NAME})
            grade_resp.raise_for_status()
            score = grade_resp.json()["score"]
            success = True
        except Exception as e:
            score = 0.0
            success = False

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        success = False
        score = 0.0

    finally:
        env.close()
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    run_inference()
