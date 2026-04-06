"""Unit tests for grader.grader.Grader."""

import pytest
from grader.grader import Grader


@pytest.fixture
def grader():
    return Grader()


class TestGrader:
    def test_empty_history(self, grader):
        assert grader.grade({}, []) == 0.0

    def test_perfect_solve(self, grader):
        state = {"step_count": 4, "max_steps": 10, "action_history": ["analyze_code", "suggest_fix", "run_tests", "submit_solution"]}
        history = [{"tests_passed": 3, "tests_total": 3}]
        score = grader.grade(state, history)
        assert 0.8 <= score <= 1.0

    def test_no_tests_passed(self, grader):
        state = {"step_count": 10, "max_steps": 10, "action_history": ["submit_solution"]}
        history = [{"tests_passed": 0, "tests_total": 3}]
        score = grader.grade(state, history)
        assert score < 0.3

    def test_partial_pass(self, grader):
        state = {"step_count": 5, "max_steps": 10, "action_history": ["analyze_code", "suggest_fix", "run_tests", "suggest_fix", "submit_solution"]}
        history = [{"tests_passed": 2, "tests_total": 4}]
        score = grader.grade(state, history)
        assert 0.3 <= score <= 0.7

    def test_score_in_range(self, grader):
        """Score must always be in [0.0, 1.0]."""
        test_cases = [
            ({"step_count": 1, "max_steps": 10, "action_history": ["submit_solution"]}, [{"tests_passed": 5, "tests_total": 5}]),
            ({"step_count": 10, "max_steps": 10, "action_history": ["analyze_code"] * 10}, [{"tests_passed": 0, "tests_total": 1}]),
        ]
        for state, history in test_cases:
            score = grader.grade(state, history)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {state}"

    def test_efficiency_matters(self, grader):
        """Solving in fewer steps should yield a higher score."""
        fast = {"step_count": 2, "max_steps": 10, "action_history": ["suggest_fix", "submit_solution"]}
        slow = {"step_count": 9, "max_steps": 10, "action_history": ["analyze_code"] * 7 + ["suggest_fix", "submit_solution"]}
        hist = [{"tests_passed": 3, "tests_total": 3}]
        assert grader.grade(fast, hist) > grader.grade(slow, hist)

    def test_quality_penalises_repeats(self, grader):
        """Repeating the same action should lower the quality score."""
        clean = {"step_count": 4, "max_steps": 10, "action_history": ["analyze_code", "suggest_fix", "run_tests", "submit_solution"]}
        messy = {"step_count": 4, "max_steps": 10, "action_history": ["analyze_code", "analyze_code", "analyze_code", "submit_solution"]}
        hist = [{"tests_passed": 3, "tests_total": 3}]
        assert grader.grade(clean, hist) >= grader.grade(messy, hist)
