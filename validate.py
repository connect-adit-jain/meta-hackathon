#!/usr/bin/env python3
"""Self-validation script — mirrors ``openenv validate``.

Checks:
  1. openenv.yaml exists and is well-formed
  2. All referenced task files exist and parse correctly
  3. Environment can reset() for every task
  4. step() returns correct types for every action type
  5. Grader returns a score in [0.0, 1.0]
  6. Server module loads without import errors
  7. Dockerfile exists
  8. inference.py exists and parses
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def run_checks() -> None:
    print("\n╔══════════════════════════════════════════╗")
    print("║   OpenEnv Validation — CodeDebug v1.0   ║")
    print("╚══════════════════════════════════════════╝\n")

    # 1 — openenv.yaml
    yaml_path = ROOT / "openenv.yaml"
    try:
        meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        check("openenv.yaml exists & parses", True)
        check("openenv.yaml has 'name'", "name" in meta)
        check("openenv.yaml has 'environment'", "environment" in meta)
        check("openenv.yaml has 'grader'", "grader" in meta)
    except Exception as exc:
        check("openenv.yaml", False, str(exc))

    # 2 — Task files
    task_dir = ROOT / "tasks"
    task_count = 0
    for diff in ("easy", "medium", "hard"):
        d = task_dir / diff
        if not d.is_dir():
            check(f"tasks/{diff}/ directory", False, "missing")
            continue
        for fp in sorted(d.glob("*.yaml")):
            try:
                t = yaml.safe_load(fp.read_text(encoding="utf-8"))
                assert "id" in t and "buggy_code" in t and "test_cases" in t
                task_count += 1
            except Exception as exc:
                check(f"task {fp.name}", False, str(exc))
    check(f"Task files valid ({task_count} loaded)", task_count >= 9)

    # 3 — Environment reset + step
    try:
        sys.path.insert(0, str(ROOT))
        from env import CodeDebugEnv, Action, ActionType

        env = CodeDebugEnv(task_dir=str(task_dir))
        for task in env.tasks:
            obs = env.reset(task_id=task["id"])
            assert obs.task_id == task["id"]
        check("env.reset() works for all tasks", True)

        # Step with each action type
        env.reset()
        obs, r, d, i = env.step(Action(action_type=ActionType.ANALYZE_CODE))
        assert hasattr(obs, "buggy_code")
        assert hasattr(r, "value")
        check("env.step(analyze_code) returns valid types", True)

        obs, r, d, i = env.step(
            Action(action_type=ActionType.SUGGEST_FIX, payload="def f(): pass")
        )
        check("env.step(suggest_fix) returns valid types", True)

        obs, r, d, i = env.step(Action(action_type=ActionType.RUN_TESTS))
        assert "tests_passed" in i
        check("env.step(run_tests) returns test info", True)

        obs, r, d, i = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        assert d is True
        check("env.step(submit_solution) terminates episode", True)

    except Exception as exc:
        check("Environment lifecycle", False, str(exc))

    # 4 — Grader: determinism, variability, range
    try:
        from grader import Grader

        g = Grader()

        # Score must be in [0.0, 1.0]
        env.reset()
        env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        score1 = g.grade(env.state(), [{"tests_passed": 3, "tests_total": 3}])
        assert 0.0 <= score1 <= 1.0
        check(f"Grader score in [0,1] ({score1})", True)

        # Determinism: same input → same output
        score2 = g.grade(env.state(), [{"tests_passed": 3, "tests_total": 3}])
        check("Grader is deterministic", score1 == score2)

        # Variability: different inputs → different scores (no constant grader)
        score_bad = g.grade(
            {"step_count": 10, "max_steps": 10, "action_history": ["submit_solution"] * 10},
            [{"tests_passed": 0, "tests_total": 3}],
        )
        check(f"Grader varies by input ({score1} != {score_bad})", score1 != score_bad)

    except Exception as exc:
        check("Grader", False, str(exc))

    # 5 — Episode always terminates (no infinite loops)
    try:
        env.reset()
        for _ in range(env.max_steps + 5):
            if env.done:
                break
            env.step(Action(action_type=ActionType.ANALYZE_CODE))
        check("Episode terminates within max_steps", env.done)
    except Exception as exc:
        check("Episode termination", False, str(exc))

    # 6 — All correct solutions yield +5.0 reward
    try:
        all_correct = True
        for task in env.tasks:
            env.reset(task_id=task["id"])
            correct = task.get("correct_solution", "")
            if correct:
                env.step(Action(action_type=ActionType.SUGGEST_FIX, payload=correct))
                _, r, _, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
                if r.value != 5.0:
                    all_correct = False
                    check(f"Task {task['id']}: correct solution", False, f"reward={r.value}")
        if all_correct:
            check("All correct solutions score +5.0", True)
    except Exception as exc:
        check("Correct solutions", False, str(exc))

    # 7 — Server module
    try:
        import server  # noqa: F401
        check("server.py imports cleanly", True)
    except Exception as exc:
        check("server.py import", False, str(exc))

    # 8 — Dockerfile
    check("Dockerfile exists", (ROOT / "Dockerfile").exists())

    # 9 — inference.py syntax + format check
    try:
        src = (ROOT / "inference.py").read_text(encoding="utf-8")
        compile(src, "inference.py", "exec")
        check("inference.py parses cleanly", True)

        # Check stdout format markers exist in source
        has_start = "[START] task=" in src
        has_step = "[STEP] step=" in src
        has_end = "[END] success=" in src
        check("inference.py has [START] format", has_start)
        check("inference.py has [STEP] format", has_step)
        check("inference.py has [END] format", has_end)

        # Check required env vars are referenced
        for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
            check(f"inference.py uses {var}", var in src)

    except SyntaxError as exc:
        check("inference.py syntax", False, str(exc))

    # 10 — README
    readme = ROOT / "README.md"
    check("README.md exists", readme.exists())
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        check("README > 500 chars", len(content) > 500)

    # Summary
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"\n{'─'*44}")
    if failed == 0:
        print(f"  {PASS}  ALL {total} CHECKS PASSED")
    else:
        print(f"  {FAIL}  {failed}/{total} CHECKS FAILED")
    print()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_checks()
