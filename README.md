---
title: BloodBank Environment
emoji: 🩸
colorFrom: red
colorTo: white
sdk: docker
app_port: 7860
license: mit
---

# 🩸 BloodBank – Hospital Blood Bank RL Environment

> **Meta PyTorch OpenEnv Hackathon × SST 2026 – Round 1 Submission**

An OpenEnv-compatible reinforcement learning environment that simulates a real-world **hospital blood bank inventory and allocation system**. The agent must manage perishable blood unit inventory across 8 ABO/Rh blood types, fulfil transfusion requests while ensuring compatibility, prioritise emergencies, minimise expiry wastage, and adapt to stochastic donation-camp inflows.

---

## 🏥 Real-World Motivation

Every year, **millions of blood units expire unused** in hospital blood banks while patients die waiting for transfusions. In **India alone**:

- 🩸 **12,000+ lives** are lost annually due to blood shortages
- 🗑️ **~6% of collected blood** expires before use due to poor stock rotation
- ⚠️ Transfusion errors (ABO incompatibility) remain a leading cause of transfusion-related fatalities

Blood bank managers face a complex daily optimisation problem:
- **Perishable inventory** — Red blood cells (RBCs) expire in 35 days
- **8 blood types** with strict ABO/Rh compatibility rules
- **Variable demand** — emergencies arrive unpredictably
- **Stochastic supply** — donation camps add random bursts of stock
- **Multiple priorities** — emergencies vs. surgeries vs. elective procedures

This environment captures these real operational challenges, enabling RL agents to learn policies that could **directly save lives** if deployed in hospital logistics systems.

---

## 📐 Environment Specification

### Action Space

The agent produces a `BloodBankAction` each step:

| Field | Type | Description |
|---|---|---|
| `request_id` | `int` | ID of the transfusion request to fulfil |
| `donor_blood_type` | `str` | Blood type to allocate from inventory (e.g. `"O+"`) |
| `units_to_allocate` | `int` | Number of units to give (1–4) |
| `skip` | `bool` | If `True`, agent deliberately does nothing this step |

### Observation Space

The agent receives a `BloodBankObservation` containing:

| Field | Type | Description |
|---|---|---|
| `inventory` | `Dict[str, List[int]]` | Blood type → sorted list of days-to-expiry |
| `pending_requests` | `List[Dict]` | Active transfusion requests with ID, blood type, units needed, priority, age |
| `step_number` | `int` | Current step in the episode |
| `total_units` | `int` | Total blood units across all types |
| `total_requests` | `int` | Number of pending requests |
| `units_expiring_soon` | `int` | Units with ≤ 3 days remaining |
| `reward` | `float` | Reward from the previous step |
| `done` | `bool` | Whether the episode has ended |
| `message` | `str` | Human-readable status message |
| `donation_event` | `str \| null` | Description of any donation camp that occurred |

### State

Episode metadata returned by `GET /state`:

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str` | UUID for the episode |
| `step_count` | `int` | Steps elapsed |
| `total_fulfilled` | `int` | Requests successfully fulfilled |
| `total_requests_generated` | `int` | Total requests created |
| `total_expired` | `int` | Units that expired |
| `total_incompatible` | `int` | Incompatible allocation attempts |
| `total_emergency_fulfilled` | `int` | Emergency requests fulfilled |
| `cumulative_reward` | `float` | Running reward total |

### ABO/Rh Compatibility Matrix

```
Donor →  O+   O-   A+   A-   B+   B-   AB+  AB-
Recip ↓
O+       ✅   ✅   ❌   ❌   ❌   ❌   ❌   ❌
O-       ❌   ✅   ❌   ❌   ❌   ❌   ❌   ❌
A+       ✅   ✅   ✅   ✅   ❌   ❌   ❌   ❌
A-       ❌   ✅   ❌   ✅   ❌   ❌   ❌   ❌
B+       ✅   ✅   ❌   ❌   ✅   ✅   ❌   ❌
B-       ❌   ✅   ❌   ❌   ❌   ✅   ❌   ❌
AB+      ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
AB-      ❌   ✅   ❌   ✅   ❌   ✅   ❌   ✅
```

---

## 🎯 Tasks & Graders

### Task 1: Basic Compatibility & Fulfillment (Easy)

**Objective:** Correctly match blood types and maximise request fulfillment.

**Grader formula:**
```
score = 0.6 × compatibility_score + 0.4 × fulfillment_rate
```
- `compatibility_score` = 1.0 if zero incompatible allocations, else 0.0
- `fulfillment_rate` = fulfilled / total_requests_generated

**Success criteria:** Score ≥ 0.7 requires zero incompatible allocations and ≥ 25% fulfillment.

---

### Task 2: Expiry-Aware Stock Rotation (Medium)

**Objective:** Use near-expiry units first (FEFO) and minimise wastage.

**Grader formula:**
```
score = 0.5 × near_expiry_ratio + 0.3 × (1 - wastage_rate) + 0.2 × fulfillment_rate
```
- `near_expiry_ratio` = near-expiry units used / total units allocated
- `wastage_rate` = wasted units / (allocated + wasted)

**Success criteria:** Score ≥ 0.6 requires consistent use of expiring stock and < 10% wastage.

---

### Task 3: Adaptive Management Under Uncertainty (Hard)

**Objective:** Handle emergencies, stochastic donations, and maintain stock balance simultaneously.

**Grader formula:**
```
score = 0.25 × emergency_rate + 0.20 × (1 - wastage_rate) + 0.15 × fulfillment_rate
     + 0.15 × compatibility + 0.15 × stock_balance + 0.10 × emergency_speed
```

**Success criteria:**
- Emergency fulfillment > 90%
- Wastage < 5%
- Zero incompatible allocations
- Balanced stock (coefficient of variation < 0.5)
- Average emergency response ≤ 1 step

---

## 💰 Reward Function

The environment provides **dense, per-step rewards** with clear partial progress signals:

| Signal | Reward | Description |
|---|---|---|
| Correct compatibility | +2.0 | Allocated a compatible blood type |
| Near-expiry used | +1.5 per unit | Used a unit with ≤ 5 days to expiry |
| Emergency fast response | +3.0 | Fulfilled an emergency on the same step it arrived |
| General fulfillment | +1.0 per unit | Each unit successfully allocated |
| Stock balance bonus | +0.5 | Coefficient of variation < 0.5 across types |
| Donation camp adaptation | +0.5 | Bonus when donation camp occurs |
| **Per-step cost** | **-0.1** | Encourages efficiency, discourages stalling |
| Skip penalty | -0.3 | Choosing to do nothing |
| Invalid action | -1.0 | Non-existent request ID or blood type |
| Incompatible allocation | **-10.0** | Life-critical error — wrong blood type match |
| Emergency delay | -3.0 per step | Emergency request waiting > 1 step |
| Unit expired/wasted | -2.0 per unit | Stock that expired unused |

**Reward profile by agent quality:**
| Agent | Expected Cumulative Reward (30 steps) |
|---|---|
| Random | -30 to -10 |
| Heuristic (FEFO + priority) | +10 to +30 |
| Trained RL / Strong LLM | +40 to +70 |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerised deployment)

### Local Development

```bash
# Clone the repository
git clone <repo-url> && cd bloodbank

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, test with curl
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_name": "basic_compatibility"}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"request_id": 0, "donor_blood_type": "O-", "units_to_allocate": 1, "skip": false}'
curl http://localhost:8000/state
curl http://localhost:8000/health
```

### Docker

```bash
# Build the image
docker build -t bloodbank:latest .

# Run the container
docker run -p 8000:8000 bloodbank:latest

# Verify health
curl http://localhost:8000/health
# → {"status":"ok","environment":"bloodbank"}

# Verify reset
curl -X POST http://localhost:8000/reset
# → {"observation": {...}, "info": {"episode_id": "..."}}
```

### Hugging Face Space

Push to a HF Space using the provided Dockerfile. The Space will automatically
serve the environment on port 8000.

```bash
# Using the HF CLI
huggingface-cli repo create bloodbank --type space --space-sdk docker
git remote add hf https://huggingface.co/spaces/<your-username>/bloodbank
git push hf main
```

---

## 🤖 Running the Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:8000"
export TASK_NAME="basic_compatibility"
export MAX_STEPS=30

# Run inference (server must be running)
python inference.py
```

**Expected output format:**
```
[START] task=basic_compatibility env=bloodbank model=meta-llama/Llama-3-8B-Instruct
[STEP] step=1 action={"request_id":0,"donor_blood_type":"O-","units_to_allocate":1,"skip":false} reward=6.50 done=false error=null
[STEP] step=2 action={"request_id":3,"donor_blood_type":"A+","units_to_allocate":2,"skip":false} reward=4.10 done=false error=null
...
[END] success=true steps=30 score=0.850 rewards=6.50,4.10,...
```

---

## 📊 Pre-Validation Checklist

| Check | Command | Expected |
|---|---|---|
| Docker builds | `docker build -t bloodbank .` | No errors |
| Health check | `curl http://localhost:8000/health` | `{"status":"ok"}` |
| Reset returns 200 | `curl -X POST http://localhost:8000/reset` | 200 OK with observation |
| Step works | `curl -X POST http://localhost:8000/step -d '{"skip":true}'` | 200 OK |
| Grader varies | Run random vs. heuristic agents | Different scores |
| Inference logs | `python inference.py` | Valid [START]/[STEP]/[END] |
| Score in [0,1] | `curl -X POST http://localhost:8000/grade` | `{"score": 0.XXX}` |

---

## 📁 Repository Structure

```
bloodbank/
├── openenv.yaml          # OpenEnv manifest (spec_version, tasks, runtime)
├── Dockerfile             # Multi-stage Docker build for HF Space
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and deps
├── models.py              # Action, Observation, State dataclasses + constants
├── client.py              # Sync & async HTTP clients
├── inference.py           # LLM agent inference script with mandated logging
├── __init__.py            # Package exports
├── .dockerignore          # Docker build exclusions
├── README.md              # This file
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI server (/reset, /step, /state, /health, /grade)
    └── environment.py     # Core environment simulation + graders
```

---

## ⚖️ License

MIT License. This is an original work created for the Meta PyTorch OpenEnv Hackathon × SST 2026.

---

## 🙏 Acknowledgments

- [Meta PyTorch OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework
- Blood type compatibility data from WHO and Indian Red Cross Society guidelines
- Indian blood type distribution based on published epidemiological studies
