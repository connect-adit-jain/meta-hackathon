"""Unit tests for env.executor.CodeExecutor."""

import pytest
from env.executor import CodeExecutor


@pytest.fixture
def executor():
    return CodeExecutor(timeout=5)


class TestCheckSyntax:
    def test_valid_code(self, executor):
        result = executor.check_syntax("x = 1 + 2")
        assert result["valid"] is True
        assert result["error"] is None

    def test_syntax_error(self, executor):
        result = executor.check_syntax("def f(\n")
        assert result["valid"] is False
        assert result["error"] is not None

    def test_empty_code(self, executor):
        result = executor.check_syntax("")
        assert result["valid"] is True


class TestExecute:
    def test_simple_expression(self, executor):
        result = executor.execute("x = 42", "x")
        assert result["success"] is True
        assert result["output"] == "42"

    def test_function_call(self, executor):
        code = "def add(a, b): return a + b"
        result = executor.execute(code, "add(3, 4)")
        assert result["success"] is True
        assert result["output"] == "7"

    def test_runtime_error(self, executor):
        result = executor.execute("x = 1 / 0", "x")
        assert result["success"] is False
        assert "ZeroDivision" in result["error"]

    def test_syntax_error_in_code(self, executor):
        result = executor.execute("def f(:", "f()")
        assert result["success"] is False

    def test_timeout(self):
        exe = CodeExecutor(timeout=2)
        result = exe.execute("import time; time.sleep(10)", "1")
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
