"""Unit tests for env.environment.CodeDebugEnv."""

import pytest
from env.environment import CodeDebugEnv
from env.models import Action, ActionType, Observation, Reward


@pytest.fixture
def env():
    return CodeDebugEnv()


class TestReset:
    def test_reset_random(self, env):
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.step_count == 0
        assert obs.buggy_code != ""

    def test_reset_by_difficulty(self, env):
        for diff in ("easy", "medium", "hard"):
            obs = env.reset(difficulty=diff)
            assert obs.difficulty == diff

    def test_reset_by_task_id(self, env):
        obs = env.reset(task_id="easy_001")
        assert obs.task_id == "easy_001"

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError, match="not found"):
            env.reset(task_id="nonexistent_999")

    def test_reset_clears_state(self, env):
        env.reset()
        env.step(Action(action_type=ActionType.ANALYZE_CODE))
        obs = env.reset()
        assert obs.step_count == 0


class TestStep:
    def test_analyze_first_time(self, env):
        env.reset()
        obs, reward, done, info = env.step(Action(action_type=ActionType.ANALYZE_CODE))
        assert reward.value == 0.2
        assert done is False

    def test_analyze_repeated(self, env):
        env.reset()
        env.step(Action(action_type=ActionType.ANALYZE_CODE))
        _, reward, _, _ = env.step(Action(action_type=ActionType.ANALYZE_CODE))
        assert reward.value == -0.2

    def test_suggest_fix_empty(self, env):
        env.reset()
        _, reward, _, _ = env.step(Action(action_type=ActionType.SUGGEST_FIX, payload=""))
        assert reward.value == -0.2

    def test_suggest_fix_syntax_error(self, env):
        env.reset()
        _, reward, _, _ = env.step(
            Action(action_type=ActionType.SUGGEST_FIX, payload="def f(:")
        )
        assert reward.value == -0.2

    def test_suggest_fix_correct(self, env):
        env.reset(task_id="easy_001")
        _, reward, _, _ = env.step(
            Action(
                action_type=ActionType.SUGGEST_FIX,
                payload="def add(a, b):\n    return a + b\n",
            )
        )
        assert reward.value >= 1.0

    def test_run_tests(self, env):
        env.reset()
        _, _, _, info = env.step(Action(action_type=ActionType.RUN_TESTS))
        assert "tests_passed" in info
        assert "tests_total" in info

    def test_submit_correct(self, env):
        env.reset(task_id="easy_001")
        env.step(
            Action(
                action_type=ActionType.SUGGEST_FIX,
                payload="def add(a, b):\n    return a + b\n",
            )
        )
        _, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        assert done is True
        assert reward.value == 5.0
        assert info["termination_reason"] == "successful_fix"

    def test_submit_incorrect(self, env):
        env.reset()
        _, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        assert done is True
        assert reward.value == -1.0

    def test_step_after_done(self, env):
        env.reset()
        env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        with pytest.raises(RuntimeError, match="finished"):
            env.step(Action(action_type=ActionType.ANALYZE_CODE))


class TestState:
    def test_state_after_reset(self, env):
        env.reset(task_id="easy_001")
        state = env.state()
        assert state["task_id"] == "easy_001"
        assert state["step_count"] == 0
        assert state["done"] is False

    def test_episode_history_recorded(self, env):
        env.reset()
        env.step(Action(action_type=ActionType.ANALYZE_CODE))
        env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
        state = env.state()
        assert len(state["episode_history"]) == 2


class TestAllTasks:
    """Verify every task loads and the correct solution passes all tests."""

    def test_all_correct_solutions_pass(self, env):
        import yaml, os

        solutions = {}
        for root, _, files in os.walk("tasks"):
            for f in files:
                if f.endswith(".yaml"):
                    with open(os.path.join(root, f)) as fh:
                        t = yaml.safe_load(fh)
                        solutions[t["id"]] = t["correct_solution"]

        for task in env.tasks:
            env.reset(task_id=task["id"])
            correct = solutions[task["id"]]
            env.step(Action(action_type=ActionType.SUGGEST_FIX, payload=correct))
            _, reward, done, info = env.step(Action(action_type=ActionType.SUBMIT_SOLUTION))
            assert reward.value == 5.0, (
                f"Task {task['id']}: expected correct submission but got reward={reward.value}"
            )
