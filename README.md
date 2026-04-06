# CodeDebug OpenEnv 🐛→✅

A **production-grade, OpenEnv-compliant** environment where an AI agent
iteratively debugs and fixes buggy Python code through analysis, patching,
testing, and submission — with dense reward feedback at every step.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        inference.py                             │
│              LLM Agent (OpenAI-compatible API)                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Analyse  │→ │Suggest Fix│→ │Run Tests │→ │Submit / Retry│  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP (POST /reset, /step)
┌────────────────────────▼────────────────────────────────────────┐
│                       server.py                                 │
│          FastAPI + CORS + Session Management                    │
│   /reset  /step  /state  /tasks  /history  /health  /docs      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  env/environment.py                              │
│         CodeDebugEnv (reset / step / state)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Task Loader │  │ Reward Logic │  │ Episode History Recorder│ │
│  └──────┬──────┘  └──────────────┘  └───────────────────────┘  │
│         │                                                       │
│  ┌──────▼──────┐  ┌──────────────┐                              │
│  │ tasks/*.yaml│  │ executor.py  │  Sandboxed subprocess runner │
│  └─────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   grader/grader.py                               │
│  Deterministic scoring:  60% correctness                        │
│                          20% efficiency                         │
│                          20% action quality                     │
│  Score range: [0.0, 1.0]                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Observation Space

| Field                | Type             | Description                              |
|----------------------|------------------|------------------------------------------|
| `buggy_code`         | `str`            | Original buggy code snippet              |
| `error_message`      | `str`            | Error or symptom description             |
| `expected_output`    | `str`            | What the correct code should produce     |
| `current_code_state` | `str`            | Latest version of the code (mutates)     |
| `step_count`         | `int`            | Steps taken so far                       |
| `max_steps`          | `int`            | Maximum steps (difficulty-adjusted)      |
| `test_results`       | `list[dict]|null`| Results from the last `run_tests` action |
| `task_id`            | `str`            | Unique task identifier                   |
| `difficulty`         | `str`            | `easy` / `medium` / `hard`              |

## Action Space

| Action            | Payload       | Reward                                           |
|-------------------|---------------|--------------------------------------------------|
| `analyze_code`    | —             | +0.2 (first time), −0.2 (repeat)                |
| `suggest_fix`     | corrected code| +1.0 (all pass), proportional partial, −0.2 fail|
| `run_tests`       | —             | +0.1                                             |
| `submit_solution` | —             | +5.0 (correct), −1.0 (incorrect)                 |

---

## Tasks — 15 challenges across 3 difficulty tiers

### Easy (5 tasks) — Syntax Errors
| ID | Title | Bug |
|----|-------|-----|
| `easy_001` | Missing Operand | `return a +` |
| `easy_002` | Missing Colon | `def greet(name)` |
| `easy_003` | Unclosed Parenthesis | `return (a * b` |
| `easy_004` | Indentation Error | Body not indented |
| `easy_005` | Missing Quote | Unclosed string literal |

### Medium (5 tasks) — Logical Bugs
| ID | Title | Bug |
|----|-------|-----|
| `medium_001` | Off-By-One Factorial | `range(1, n)` → `range(1, n+1)` |
| `medium_002` | Wrong Comparison | `<` instead of `>` in find_max |
| `medium_003` | Swapped FizzBuzz | Fizz/Buzz labels reversed |
| `medium_004` | Reverse String | `s[::1]` → `s[::-1]` |
| `medium_005` | Assignment vs Increment | `counts[w] = 1` → `+= 1` |

### Hard (5 tasks) — Multi-Step Reasoning
| ID | Title | Bugs |
|----|-------|------|
| `hard_001` | Prime Counter | Inverted returns + off-by-one across 2 functions |
| `hard_002` | Flatten List | `append` → `extend` in recursive flatten |
| `hard_003` | Matrix Transpose | Wrong result dimensions (rows↔cols) |
| `hard_004` | Binary Search | 3 interacting bugs (boundary, condition, step) |
| `hard_005` | Memoised Fibonacci | Wrong base case + wrong recursion (`n-3`→`n-2`) |

---

## Grading

Scores are **deterministic** and normalised to **[0.0, 1.0]**:

| Component | Weight | Metric |
|-----------|--------|--------|
| **Correctness** | 60% | Proportion of tests passed |
| **Efficiency** | 20% | Fewer steps → higher score |
| **Action Quality** | 20% | Unique action variety, no repeats |

---

## Quick Start

### Local Development

```bash
# Install
pip install -r requirements.txt

# Start server
python run.py server

# Run tests
python run.py test

# Self-validate (mirrors openenv validate)
python run.py validate

# Interactive demo
python run.py demo
```

### Inference (with LLM)

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_..."
python run.py inference
```

### Docker

```bash
docker build -t codedebug-openenv .
docker run -p 7860:7860 codedebug-openenv
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "analyze_code"}'

# List tasks
curl http://localhost:7860/tasks

# Episode history
curl http://localhost:7860/history

# Swagger docs
open http://localhost:7860/docs
```

### Python API

```python
from env import CodeDebugEnv, Action, ActionType
from grader import Grader

env = CodeDebugEnv()
grader = Grader()

obs = env.reset(difficulty="hard")
obs, reward, done, info = env.step(Action(action_type=ActionType.ANALYZE_CODE))
obs, reward, done, info = env.step(
    Action(action_type=ActionType.SUGGEST_FIX, payload="def is_prime(n):...")
)
obs, reward, done, info = env.step(Action(action_type=ActionType.RUN_TESTS))
obs, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))

score = grader.grade(env.state(), env.episode_history)
print(f"Score: {score:.4f}")  # [0.0, 1.0]
```

---

## Project Structure

```
├── env/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Pydantic: Observation, Action, Reward
│   ├── executor.py          # Sandboxed subprocess code runner
│   └── environment.py       # Core OpenEnv (reset/step/state + history)
├── tasks/
│   ├── easy/                # 5 syntax-error tasks
│   ├── medium/              # 5 logical-bug tasks
│   └── hard/                # 5 multi-step-bug tasks
├── grader/
│   ├── __init__.py
│   └── grader.py            # Deterministic scorer [0.0, 1.0]
├── tests/
│   ├── test_executor.py     # Executor unit tests
│   ├── test_environment.py  # Environment lifecycle tests
│   └── test_grader.py       # Grader unit tests
├── config.py                # Centralised configuration
├── inference.py             # LLM agent (multi-turn, retry logic)
├── server.py                # FastAPI (CORS, sessions, /docs)
├── validate.py              # Self-validation (openenv validate)
├── run.py                   # Unified CLI entry point
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # HF Spaces compatible
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## License

MIT
