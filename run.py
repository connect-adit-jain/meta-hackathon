#!/usr/bin/env python3
"""Unified entry-point for the CodeDebug OpenEnv project.

Usage
-----
    python run.py server          # start the HTTP server
    python run.py validate        # run self-validation
    python run.py inference       # run the LLM inference agent
    python run.py test            # run unit tests
    python run.py demo            # interactive demo in the terminal
"""

from __future__ import annotations

import sys


def _server() -> None:
    import uvicorn
    from config import SERVER_CONFIG

    uvicorn.run(
        "server:app",
        host=SERVER_CONFIG.host,
        port=SERVER_CONFIG.port,
        log_level=SERVER_CONFIG.log_level,
        reload=False,
    )


def _validate() -> None:
    from validate import run_checks

    run_checks()


def _inference() -> None:
    from inference import run

    run()


def _test() -> None:
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"]))


def _demo() -> None:
    from env import CodeDebugEnv, Action, ActionType
    from grader import Grader

    env = CodeDebugEnv()
    grader = Grader()

    for diff in ("easy", "medium", "hard"):
        obs = env.reset(difficulty=diff)
        print(f"\n{'═'*60}")
        print(f"  Task: {obs.task_id}  |  Difficulty: {obs.difficulty}")
        print(f"{'═'*60}")
        print(f"  Buggy code:\n{obs.buggy_code}")
        print(f"  Error: {obs.error_message}")

        history = []
        obs, r, d, i = env.step(Action(action_type=ActionType.ANALYZE_CODE))
        history.append(i)
        print(f"  → analyze   reward={r.value:+.1f}")

        if not d:
            import yaml, os

            # Load correct solution
            for root, _, files in os.walk("tasks"):
                for f in files:
                    if f.endswith(".yaml"):
                        with open(os.path.join(root, f)) as fh:
                            t = yaml.safe_load(fh)
                            if t["id"] == obs.task_id:
                                correct = t["correct_solution"]
                                break

            obs, r, d, i = env.step(
                Action(action_type=ActionType.SUGGEST_FIX, payload=correct)
            )
            history.append(i)
            print(f"  → fix       reward={r.value:+.1f}")

        if not d:
            obs, r, d, i = env.step(Action(action_type=ActionType.RUN_TESTS))
            history.append(i)
            print(f"  → tests     reward={r.value:+.1f}  pass={i.get('tests_passed')}/{i.get('tests_total')}")

        if not d:
            obs, r, d, i = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
            history.append(i)
            print(f"  → submit    reward={r.value:+.1f}  {i.get('termination_reason')}")

        score = grader.grade(env.state(), history)
        print(f"  ★ Score: {score:.4f}")

    print(f"\n{'═'*60}")
    print("  Demo complete!")
    print(f"{'═'*60}\n")


COMMANDS = {
    "server": _server,
    "validate": _validate,
    "inference": _inference,
    "test": _test,
    "demo": _demo,
}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "server"
    if cmd in ("-h", "--help") or cmd not in COMMANDS:
        print(__doc__)
        sys.exit(0)
    COMMANDS[cmd]()
