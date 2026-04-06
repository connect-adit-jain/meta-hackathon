"""Full API integration test — hit every endpoint against a running server."""
import requests
import json

BASE = "http://localhost:7860"

print("=" * 60)
print("  CodeDebug OpenEnv — Full API Integration Test")
print("=" * 60)

# 1. Health
r = requests.get(f"{BASE}/health")
h = r.json()
print(f"\n[1] GET /health")
print(f"    Status: {h['status']}  Tasks loaded: {h['tasks_loaded']}")

# 2. List tasks
r = requests.get(f"{BASE}/tasks")
tasks = r.json()
print(f"\n[2] GET /tasks")
print(f"    Total tasks: {tasks['total']}")
for t in tasks["tasks"]:
    print(f"      {t['id']:12s} | {t['difficulty']:6s} | {t['title']}")

# 3. Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "easy_001"})
obs = r.json()["observation"]
print(f"\n[3] POST /reset (task_id=easy_001)")
print(f"    Task: {obs['task_id']}  Difficulty: {obs['difficulty']}")
print(f"    Error: {obs['error_message']}")

# 4. Analyze
r = requests.post(f"{BASE}/step", json={"action_type": "analyze_code"})
d = r.json()
print(f"\n[4] POST /step (analyze_code)")
print(f"    Reward: {d['reward']['value']}  Reason: {d['reward']['reason']}")

# 5. Suggest fix (correct solution)
fix = "def add(a, b):\n    return a + b\n"
r = requests.post(f"{BASE}/step", json={"action_type": "suggest_fix", "payload": fix})
d = r.json()
print(f"\n[5] POST /step (suggest_fix)")
print(f"    Reward: {d['reward']['value']}  Reason: {d['reward']['reason']}")

# 6. Run tests
r = requests.post(f"{BASE}/step", json={"action_type": "run_tests"})
d = r.json()
info = d["info"]
print(f"\n[6] POST /step (run_tests)")
print(f"    Passed: {info['tests_passed']}/{info['tests_total']}")

# 7. Submit solution
r = requests.post(f"{BASE}/step", json={"action_type": "submit_solution"})
d = r.json()
print(f"\n[7] POST /step (submit_solution)")
print(f"    Reward: {d['reward']['value']}  Done: {d['done']}")
print(f"    Result: {d['info']['termination_reason']}")

# 8. Episode history
r = requests.get(f"{BASE}/history")
h = r.json()
print(f"\n[8] GET /history")
print(f"    Total reward: {h['total_reward']}  Steps: {len(h['steps'])}")
for s in h["steps"]:
    print(f"      Step {s['step']}: {s['action']:15s} reward={s['reward']:+.1f}")

# 9. State
r = requests.get(f"{BASE}/state")
state = r.json()
print(f"\n[9] GET /state")
print(f"    Done: {state['done']}  Total reward: {state['total_reward']}")

print("\n" + "=" * 60)
print("  ALL ENDPOINTS WORKING!")
print("=" * 60)
