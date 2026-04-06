"""
Blood Bank Environment – Core Logic.

Implements the full environment simulation:
  • Manages perishable blood inventory across 8 ABO/Rh types.
  • Generates stochastic transfusion requests with varying priority.
  • Processes agent allocation actions and computes dense rewards.
  • Tracks metrics for the three graded tasks.

The environment is lightweight and runs entirely on CPU with numpy/random.
"""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use relative imports when running inside the package; fall back for standalone
try:
    from models import (
        BLOOD_TYPE_DISTRIBUTION,
        BLOOD_TYPE_INDEX,
        BLOOD_TYPES,
        COMPATIBILITY,
        NUM_BLOOD_TYPES,
        PRIORITY_EMERGENCY,
        PRIORITY_ROUTINE,
        PRIORITY_URGENT,
        BloodBankAction,
        BloodBankObservation,
        BloodBankState,
        BloodRequest,
        BloodUnit,
    )
except ImportError:
    from ..models import (
        BLOOD_TYPE_DISTRIBUTION,
        BLOOD_TYPE_INDEX,
        BLOOD_TYPES,
        COMPATIBILITY,
        NUM_BLOOD_TYPES,
        PRIORITY_EMERGENCY,
        PRIORITY_ROUTINE,
        PRIORITY_URGENT,
        BloodBankAction,
        BloodBankObservation,
        BloodBankState,
        BloodRequest,
        BloodUnit,
    )


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_STEPS_DEFAULT = 30
INITIAL_UNITS_PER_TYPE_RANGE = (3, 8)
EXPIRY_RANGE = (1, 35)   # RBC shelf-life in days
REQUESTS_PER_STEP = (1, 3)
DONATION_CAMP_PROB = 0.10  # 10 % per step
DONATION_CAMP_UNITS = (5, 15)
NEAR_EXPIRY_THRESHOLD = 5  # days

# Reward shaping constants
REWARD_CORRECT_COMPAT = 2.0
REWARD_NEAR_EXPIRY_USED = 1.5
REWARD_EMERGENCY_FAST = 3.0
REWARD_FULFILLMENT = 1.0
REWARD_STOCK_BALANCE = 0.5
REWARD_DONATION_ADAPT = 0.5

PENALTY_INCOMPATIBLE = -10.0     # life-critical error
PENALTY_EMERGENCY_DELAY = -3.0   # emergency waiting > 1 step
PENALTY_WASTAGE = -2.0           # per expired unit
PENALTY_STEP = -0.1              # small per-step cost
PENALTY_SKIP = -0.3              # choosing to skip
PENALTY_INVALID = -1.0           # invalid action (no such request, etc.)


class BloodBankEnvironment:
    """
    Simulates a hospital blood bank for RL agent training.

    Methods
    -------
    reset(seed, task_name) → BloodBankObservation
    step(action)           → Tuple[BloodBankObservation, float, bool, dict]
    state                  → BloodBankState   (property)
    """

    def __init__(self) -> None:
        self._state = BloodBankState()
        self._inventory: Dict[str, List[BloodUnit]] = {}
        self._requests: List[BloodRequest] = []
        self._next_unit_id: int = 0
        self._next_request_id: int = 0
        self._rng: random.Random = random.Random()
        self._np_rng: np.random.Generator = np.random.default_rng()
        self._max_steps: int = MAX_STEPS_DEFAULT
        self._donation_event: Optional[str] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        task_name: str = "basic_compatibility",
        max_steps: int = MAX_STEPS_DEFAULT,
        **kwargs: Any,
    ) -> BloodBankObservation:
        """Initialise a new episode and return the first observation."""
        # Seed RNGs
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
        else:
            self._rng = random.Random()
            self._np_rng = np.random.default_rng()

        self._max_steps = max_steps
        self._next_unit_id = 0
        self._next_request_id = 0
        self._donation_event = None

        # Initialise inventory
        self._inventory = {bt: [] for bt in BLOOD_TYPES}
        for bt in BLOOD_TYPES:
            n_units = self._rng.randint(*INITIAL_UNITS_PER_TYPE_RANGE)
            for _ in range(n_units):
                self._inventory[bt].append(
                    BloodUnit(
                        unit_id=self._next_unit_id,
                        blood_type=bt,
                        days_to_expiry=self._rng.randint(*EXPIRY_RANGE),
                    )
                )
                self._next_unit_id += 1

        # Generate initial requests
        self._requests = []
        self._generate_requests()

        # Reset state tracker
        self._state = BloodBankState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
        )

        return self._make_observation(reward=0.0, done=False, message="Episode started. Manage the blood bank wisely.")

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, action: BloodBankAction
    ) -> Tuple[BloodBankObservation, float, bool, dict]:
        """
        Execute one step: process the agent's allocation action, advance
        the simulation clock, generate new requests, expire old units.

        Returns (observation, reward, done, info).
        """
        reward = PENALTY_STEP  # baseline per-step penalty
        info: Dict[str, Any] = {}
        message_parts: List[str] = []

        # ---- 1. Process agent action ----
        if action.skip:
            reward += PENALTY_SKIP
            message_parts.append("Agent chose to skip this step.")
        else:
            action_reward, action_msg = self._process_action(action)
            reward += action_reward
            message_parts.append(action_msg)

        # ---- 2. Age requests (increase pending time) ----
        for req in self._requests:
            req.age_steps += 1
            if req.priority == PRIORITY_EMERGENCY and req.age_steps > 1:
                reward += PENALTY_EMERGENCY_DELAY
                message_parts.append(
                    f"⚠ Emergency request #{req.request_id} delayed ({req.age_steps} steps)."
                )

        # ---- 3. Advance expiry & remove expired units ----
        expired_count = self._advance_expiry()
        if expired_count > 0:
            reward += PENALTY_WASTAGE * expired_count
            self._state.total_expired += expired_count
            self._state.total_wasted_units += expired_count
            message_parts.append(f"🗑 {expired_count} unit(s) expired and discarded.")

        # ---- 4. Stochastic donation camp ----
        self._donation_event = None
        if self._rng.random() < DONATION_CAMP_PROB:
            donated = self._donation_camp_event()
            self._donation_event = donated
            reward += REWARD_DONATION_ADAPT
            message_parts.append(f"🩸 Donation camp! {donated}")

        # ---- 5. Generate new requests ----
        self._generate_requests()

        # ---- 6. Stock balance bonus ----
        balance_bonus = self._compute_stock_balance_reward()
        reward += balance_bonus

        # ---- 7. Update state ----
        self._state.step_count += 1
        self._state.cumulative_reward += reward

        done = self._state.step_count >= self._max_steps
        self._state.done = done

        if done:
            message_parts.append("Episode finished.")

        info["step_reward"] = reward
        info["cumulative_reward"] = self._state.cumulative_reward

        obs = self._make_observation(
            reward=reward,
            done=done,
            message=" | ".join(message_parts),
        )
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> BloodBankState:
        return deepcopy(self._state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_action(self, action: BloodBankAction) -> Tuple[float, str]:
        """Validate and execute an allocation action. Return (reward, message)."""
        # Find requested request
        req = None
        for r in self._requests:
            if r.request_id == action.request_id:
                req = r
                break

        if req is None:
            return PENALTY_INVALID, f"❌ Invalid request ID {action.request_id}."

        donor_bt = action.donor_blood_type
        if donor_bt not in BLOOD_TYPE_INDEX:
            return PENALTY_INVALID, f"❌ Unknown blood type '{donor_bt}'."

        recipient_bt = req.blood_type
        donor_idx = BLOOD_TYPE_INDEX[donor_bt]
        recip_idx = BLOOD_TYPE_INDEX[recipient_bt]

        # ---- Compatibility check ----
        if not COMPATIBILITY[donor_idx][recip_idx]:
            self._state.total_incompatible += 1
            return PENALTY_INCOMPATIBLE, (
                f"💀 INCOMPATIBLE: {donor_bt} → {recipient_bt}! Life-critical error."
            )

        # ---- Inventory availability ----
        available = self._inventory.get(donor_bt, [])
        units_needed = min(action.units_to_allocate, req.units_needed)
        if len(available) < units_needed:
            units_needed = len(available)
        if units_needed == 0:
            return PENALTY_INVALID, f"❌ No {donor_bt} units in stock."

        # ---- Sort by expiry (FEFO: First-Expiry First-Out) ----
        available.sort(key=lambda u: u.days_to_expiry)
        allocated_units = available[:units_needed]
        self._inventory[donor_bt] = available[units_needed:]

        # Reward: correct compatibility
        reward = REWARD_CORRECT_COMPAT

        # Reward: near-expiry units used first
        near_expiry_count = sum(
            1 for u in allocated_units if u.days_to_expiry <= NEAR_EXPIRY_THRESHOLD
        )
        if near_expiry_count > 0:
            reward += REWARD_NEAR_EXPIRY_USED * near_expiry_count
            self._state.near_expiry_used += near_expiry_count

        # Reward: fulfillment
        reward += REWARD_FULFILLMENT * units_needed

        # Track metrics
        self._state.total_allocated_units += units_needed
        self._state.total_fulfilled += 1

        # Emergency tracking
        if req.priority == PRIORITY_EMERGENCY:
            self._state.total_emergency_fulfilled += 1
            self._state.emergency_response_steps.append(req.age_steps)
            if req.age_steps == 0:
                reward += REWARD_EMERGENCY_FAST
                msg_priority = "🚑 Emergency fulfilled IMMEDIATELY!"
            else:
                msg_priority = f"🚑 Emergency fulfilled (delayed {req.age_steps} steps)."
        elif req.priority == PRIORITY_URGENT:
            msg_priority = "⚡ Urgent request fulfilled."
        else:
            msg_priority = "✅ Routine request fulfilled."

        # Remove fulfilled request
        req.units_needed -= units_needed
        if req.units_needed <= 0:
            self._requests = [r for r in self._requests if r.request_id != req.request_id]

        return reward, f"{msg_priority} Allocated {units_needed}× {donor_bt} → {recipient_bt}."

    def _generate_requests(self) -> None:
        """Add new random transfusion requests."""
        n_new = self._rng.randint(*REQUESTS_PER_STEP)
        types_pool = list(BLOOD_TYPE_DISTRIBUTION.keys())
        weights = list(BLOOD_TYPE_DISTRIBUTION.values())

        for _ in range(n_new):
            bt = self._rng.choices(types_pool, weights=weights, k=1)[0]
            priority = self._rng.choices(
                [PRIORITY_ROUTINE, PRIORITY_URGENT, PRIORITY_EMERGENCY],
                weights=[0.50, 0.35, 0.15],
                k=1,
            )[0]
            units_needed = self._rng.randint(1, 4)
            self._requests.append(
                BloodRequest(
                    request_id=self._next_request_id,
                    blood_type=bt,
                    units_needed=units_needed,
                    priority=priority,
                )
            )
            self._next_request_id += 1
            self._state.total_requests_generated += 1
            if priority == PRIORITY_EMERGENCY:
                self._state.total_emergency_generated += 1

    def _advance_expiry(self) -> int:
        """Decrease days_to_expiry by 1 for all units; remove expired ones."""
        expired = 0
        for bt in BLOOD_TYPES:
            new_list: List[BloodUnit] = []
            for unit in self._inventory[bt]:
                unit.days_to_expiry -= 1
                if unit.days_to_expiry <= 0:
                    expired += 1
                else:
                    new_list.append(unit)
            self._inventory[bt] = new_list
        return expired

    def _donation_camp_event(self) -> str:
        """Simulate a random donation camp adding units to inventory."""
        n_donated = self._rng.randint(*DONATION_CAMP_UNITS)
        types_pool = list(BLOOD_TYPE_DISTRIBUTION.keys())
        weights = list(BLOOD_TYPE_DISTRIBUTION.values())
        donated_types: Dict[str, int] = {}

        for _ in range(n_donated):
            bt = self._rng.choices(types_pool, weights=weights, k=1)[0]
            self._inventory[bt].append(
                BloodUnit(
                    unit_id=self._next_unit_id,
                    blood_type=bt,
                    days_to_expiry=self._rng.randint(20, 35),
                )
            )
            self._next_unit_id += 1
            donated_types[bt] = donated_types.get(bt, 0) + 1

        parts = [f"{v}× {k}" for k, v in donated_types.items()]
        return f"Received {n_donated} units: {', '.join(parts)}"

    def _compute_stock_balance_reward(self) -> float:
        """Reward for maintaining balanced stock across blood types."""
        counts = [len(self._inventory[bt]) for bt in BLOOD_TYPES]
        if sum(counts) == 0:
            return 0.0
        cv = float(np.std(counts) / (np.mean(counts) + 1e-8))  # coefficient of variation
        # Lower CV → better balance → higher reward
        if cv < 0.5:
            return REWARD_STOCK_BALANCE
        elif cv < 1.0:
            return REWARD_STOCK_BALANCE * 0.5
        return 0.0

    def _make_observation(
        self, reward: float, done: bool, message: str
    ) -> BloodBankObservation:
        """Build an observation dict from current state."""
        inventory_view: Dict[str, List[int]] = {}
        total_units = 0
        expiring_soon = 0
        for bt in BLOOD_TYPES:
            expiries = sorted([u.days_to_expiry for u in self._inventory[bt]])
            inventory_view[bt] = expiries
            total_units += len(expiries)
            expiring_soon += sum(1 for d in expiries if d <= 3)

        pending = []
        for req in self._requests:
            pending.append({
                "request_id": req.request_id,
                "blood_type": req.blood_type,
                "units_needed": req.units_needed,
                "priority": req.priority,
                "priority_label": (
                    "EMERGENCY" if req.priority == PRIORITY_EMERGENCY
                    else "URGENT" if req.priority == PRIORITY_URGENT
                    else "ROUTINE"
                ),
                "age_steps": req.age_steps,
            })

        return BloodBankObservation(
            inventory=inventory_view,
            pending_requests=pending,
            step_number=self._state.step_count,
            total_units=total_units,
            total_requests=len(pending),
            units_expiring_soon=expiring_soon,
            reward=reward,
            done=done,
            message=message,
            donation_event=self._donation_event,
        )

    # ------------------------------------------------------------------
    # Graders
    # ------------------------------------------------------------------
    def grade_basic_compatibility(self) -> float:
        """
        Task 1 grader: Basic Compatibility & Fulfillment.

        Score = 0.6 × (1 if zero incompatible else 0)
              + 0.4 × fulfillment_rate
        """
        s = self._state
        compat_score = 1.0 if s.total_incompatible == 0 else 0.0
        fulfill_rate = (
            s.total_fulfilled / max(s.total_requests_generated, 1)
        )
        return round(min(1.0, 0.6 * compat_score + 0.4 * fulfill_rate), 4)

    def grade_expiry_aware_rotation(self) -> float:
        """
        Task 2 grader: Expiry-Aware Stock Rotation.

        Score = 0.5 × near_expiry_usage_ratio
              + 0.3 × (1 - wastage_rate)
              + 0.2 × fulfillment_rate
        """
        s = self._state
        near_ratio = (
            s.near_expiry_used / max(s.total_allocated_units, 1)
        )
        wastage_rate = (
            s.total_wasted_units / max(s.total_allocated_units + s.total_wasted_units, 1)
        )
        fulfill_rate = (
            s.total_fulfilled / max(s.total_requests_generated, 1)
        )
        score = 0.5 * near_ratio + 0.3 * (1.0 - wastage_rate) + 0.2 * fulfill_rate
        return round(min(1.0, max(0.0, score)), 4)

    def grade_adaptive_management(self) -> float:
        """
        Task 3 grader: Adaptive Management Under Uncertainty.

        Score = 0.25 × emergency_fulfillment_rate
              + 0.20 × (1 - wastage_rate)
              + 0.15 × fulfillment_rate
              + 0.15 × compatibility_score
              + 0.15 × stock_balance_score
              + 0.10 × emergency_speed_score
        """
        s = self._state

        # Emergency fulfillment rate
        emerg_rate = (
            s.total_emergency_fulfilled / max(s.total_emergency_generated, 1)
        )

        # Wastage
        total_processed = s.total_allocated_units + s.total_wasted_units
        wastage_rate = s.total_wasted_units / max(total_processed, 1)

        # General fulfillment
        fulfill_rate = s.total_fulfilled / max(s.total_requests_generated, 1)

        # Compatibility
        compat_score = 1.0 if s.total_incompatible == 0 else max(0.0, 1.0 - s.total_incompatible * 0.2)

        # Stock balance: compute CV from current inventory
        counts = [len(self._inventory[bt]) for bt in BLOOD_TYPES]
        if sum(counts) > 0:
            cv = float(np.std(counts) / (np.mean(counts) + 1e-8))
            balance_score = max(0.0, 1.0 - cv)
        else:
            balance_score = 0.0

        # Emergency response speed (lower is better)
        if s.emergency_response_steps:
            avg_response = sum(s.emergency_response_steps) / len(s.emergency_response_steps)
            speed_score = max(0.0, 1.0 - avg_response * 0.3)
        else:
            speed_score = 0.5  # neutral if no emergencies encountered

        score = (
            0.25 * emerg_rate
            + 0.20 * (1.0 - wastage_rate)
            + 0.15 * fulfill_rate
            + 0.15 * compat_score
            + 0.15 * balance_score
            + 0.10 * speed_score
        )
        return round(min(1.0, max(0.0, score)), 4)

    def grade(self, task_name: Optional[str] = None) -> float:
        """Return graded score for the specified or current task."""
        task = task_name or self._state.task_name
        if task == "basic_compatibility":
            return self.grade_basic_compatibility()
        elif task == "expiry_aware_rotation":
            return self.grade_expiry_aware_rotation()
        elif task == "adaptive_management":
            return self.grade_adaptive_management()
        else:
            raise ValueError(f"Unknown task: {task}")
