"""Deterministic grader for the Code-Debug OpenEnv environment.

Returns a normalised score in [0.0, 1.0] based on:
  • Correctness     (60 %) — proportion of test-cases the final code passes.
  • Efficiency      (20 %) — fewer steps → higher score.
  • Action quality  (20 %) — penalises repeated / wasted actions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class Grader:
    """Stateless, deterministic grader."""

    W_CORRECT = 0.60
    W_EFFICIENCY = 0.20
    W_QUALITY = 0.20

    def grade(
        self,
        env_state: Dict[str, Any],
        episode_history: List[Dict[str, Any]],
    ) -> float:
        """Return a score in ``[0.0, 1.0]``.

        Parameters
        ----------
        env_state:
            The dict returned by ``env.state()`` at the end of the episode.
        episode_history:
            A list of *info* dicts collected from each ``env.step()`` call.
        """
        if not episode_history:
            return 0.0

        correctness = self._score_correctness(episode_history)
        efficiency = self._score_efficiency(env_state)
        quality = self._score_quality(env_state)

        score = (
            self.W_CORRECT * correctness
            + self.W_EFFICIENCY * efficiency
            + self.W_QUALITY * quality
        )
        return round(min(1.0, max(0.0, score)), 4)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_correctness(history: List[Dict]) -> float:
        """Proportion of tests passed in the final submission / last step."""
        last = history[-1]
        passed = last.get("tests_passed", 0)
        total = max(last.get("tests_total", 1), 1)
        return passed / total

    @staticmethod
    def _score_efficiency(state: Dict) -> float:
        """1.0 when solved in 1 step, 0.0 when all steps exhausted."""
        used = state.get("step_count", state.get("max_steps", 10))
        mx = max(state.get("max_steps", 10), 1)
        return max(0.0, 1.0 - used / mx)

    @staticmethod
    def _score_quality(state: Dict) -> float:
        """Penalise repeated or unproductive action patterns."""
        actions = state.get("action_history", [])
        if not actions:
            return 0.0

        counts = Counter(actions)
        total = len(actions)

        # Ideal flow: analyse → fix → test → submit  (4 unique)
        unique_ratio = min(len(counts) / 4.0, 1.0)

        # Penalise excessive repeats of the same action
        max_repeat = max(counts.values())
        repeat_penalty = max(0.0, 1.0 - (max_repeat - 1) / total) if total > 1 else 1.0

        return 0.5 * unique_ratio + 0.5 * repeat_penalty
