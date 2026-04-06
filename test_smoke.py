"""Quick smoke test — run locally, no server needed."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from env import CodeDebugEnv, Action, ActionType
from grader.grader import Grader

env = CodeDebugEnv()
grader = Grader()

# Pre-load all correct solutions keyed by task id
solutions = {}
for root, dirs, files in os.walk("tasks"):
    for f in files:
        if f.endswith((".yaml", ".yml")):
            fp = os.path.join(root, f)
            with open(fp) as fh:
                t = yaml.safe_load(fh)
                solutions[t["id"]] = t["correct_solution"]

for diff in ("easy", "medium", "hard"):
    obs = env.reset(difficulty=diff)
    print(f"\n{'='*60}")
    print(f"Task: {obs.task_id} | Difficulty: {obs.difficulty}")

    history = []

    # Step 1: analyse
    obs, reward, done, info = env.step(Action(action_type=ActionType.ANALYZE_CODE))
    history.append(info)
    print(f"  analyze  -> reward={reward.value:+.1f}  done={done}")

    if not done:
        # Step 2: suggest correct solution
        correct = solutions[obs.task_id]
        obs, reward, done, info = env.step(
            Action(action_type=ActionType.SUGGEST_FIX, payload=correct)
        )
        history.append(info)
        print(f"  fix      -> reward={reward.value:+.1f}  done={done}")

    if not done:
        # Step 3: run tests
        obs, reward, done, info = env.step(Action(action_type=ActionType.RUN_TESTS))
        history.append(info)
        print(f"  tests    -> reward={reward.value:+.1f}  passed={info.get('tests_passed')}/{info.get('tests_total')}")

    if not done:
        # Step 4: submit
        obs, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        history.append(info)
        print(f"  submit   -> reward={reward.value:+.1f}  result={info.get('termination_reason')}")

    score = grader.grade(env.state(), history)
    print(f"  GRADER SCORE: {score}")

print("\nAll smoke tests passed!")
