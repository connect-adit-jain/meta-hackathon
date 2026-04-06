"""
Blood Bank Environment – Typed Data Models.

Defines the Action, Observation, and State dataclasses used by both the
server-side environment and the client. All types are Pydantic-compatible
for JSON serialisation over HTTP / WebSockets.

Blood Type Encoding (index → type):
    0: O+   1: O-   2: A+   3: A-
    4: B+   5: B-   6: AB+  7: AB-
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Blood-type constants
# ---------------------------------------------------------------------------
BLOOD_TYPES: List[str] = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
BLOOD_TYPE_INDEX: Dict[str, int] = {bt: i for i, bt in enumerate(BLOOD_TYPES)}
NUM_BLOOD_TYPES: int = len(BLOOD_TYPES)

# ABO/Rh compatibility matrix: COMPATIBLE[donor][recipient] == True means safe.
# Rows = donor type index, Cols = recipient type index.
COMPATIBILITY: List[List[bool]] = [
    # O+     O-     A+     A-     B+     B-     AB+    AB-
    [True,  False, True,  False, True,  False, True,  False],  # O+
    [True,  True,  True,  True,  True,  True,  True,  True ],  # O-  (universal)
    [False, False, True,  False, False, False, True,  False],  # A+
    [False, False, True,  True,  False, False, True,  True ],  # A-
    [False, False, False, False, True,  False, True,  False],  # B+
    [False, False, False, False, True,  True,  True,  True ],  # B-
    [False, False, False, False, False, False, True,  False],  # AB+
    [False, False, False, False, False, False, True,  True ],  # AB-
]

# Realistic Indian blood-type distribution (approx. percentages)
BLOOD_TYPE_DISTRIBUTION: Dict[str, float] = {
    "O+": 0.37, "O-": 0.02, "A+": 0.22, "A-": 0.06,
    "B+": 0.21, "B-": 0.02, "AB+": 0.07, "AB-": 0.03,
}


# ---------------------------------------------------------------------------
# Request priority levels
# ---------------------------------------------------------------------------
PRIORITY_EMERGENCY: int = 2  # life-threatening, must fulfil ASAP
PRIORITY_URGENT: int = 1     # surgery / scheduled, moderate time window
PRIORITY_ROUTINE: int = 0    # elective / cross-match, flexible


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BloodRequest:
    """A single pending transfusion request."""
    request_id: int
    blood_type: str          # requested blood type (e.g. "A+")
    units_needed: int        # how many units required (1-4)
    priority: int            # 0=routine, 1=urgent, 2=emergency
    age_steps: int = 0       # how many steps this request has been pending


@dataclass
class BloodUnit:
    """A single blood unit in inventory."""
    unit_id: int
    blood_type: str
    days_to_expiry: int      # RBC shelf life countdown (1-35 days)


@dataclass
class BloodBankAction:
    """
    Agent's action for a single step.

    The agent selects which request to fulfil and which donor blood type
    to allocate from inventory.  Setting `skip = True` means the agent
    deliberately chooses not to fulfil any request this step (e.g. waiting
    for better stock).
    """
    request_id: int = -1             # ID of request to fulfil
    donor_blood_type: str = "O+"     # blood type to allocate from inventory
    units_to_allocate: int = 1       # number of units to give
    skip: bool = False               # True → do nothing this step


@dataclass
class BloodBankObservation:
    """
    What the agent sees after each step (or at reset).
    """
    inventory: Dict[str, List[int]]    # blood_type → list of days_to_expiry
    pending_requests: List[Dict]       # list of request dicts
    step_number: int = 0
    total_units: int = 0               # total units across all types
    total_requests: int = 0            # pending request count
    units_expiring_soon: int = 0       # units with ≤ 3 days left
    reward: float = 0.0
    done: bool = False
    message: str = ""
    donation_event: Optional[str] = None  # description of recent donation camp


@dataclass
class BloodBankState:
    """
    Internal episode state – returned via the state() property.
    """
    episode_id: str = ""
    step_count: int = 0
    total_fulfilled: int = 0
    total_requests_generated: int = 0
    total_expired: int = 0
    total_wasted_units: int = 0
    total_incompatible: int = 0
    total_emergency_fulfilled: int = 0
    total_emergency_generated: int = 0
    emergency_response_steps: List[int] = field(default_factory=list)
    near_expiry_used: int = 0          # near-expiry units used (≤ 5 days)
    total_allocated_units: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    task_name: str = "basic_compatibility"
