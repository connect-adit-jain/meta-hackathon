"""Safe sandboxed Python code executor used by the environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict


class CodeExecutor:
    """Execute untrusted Python snippets in a subprocess with a timeout."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #

    def execute(self, code: str, test_expression: str) -> Dict:
        """Run *code* then ``print(<test_expression>)`` and return the result."""
        full_code = f"{code}\nprint({test_expression})"
        return self._run(full_code)

    def check_syntax(self, code: str) -> Dict:
        """Return ``{valid: bool, error: str|None}``."""
        try:
            compile(code, "<agent_code>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as exc:
            return {"valid": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _run(self, code: str) -> Dict:
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            )
            tmp.write(code)
            tmp.flush()
            tmp.close()

            result = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout.strip(),
                    "error": None,
                }
            return {
                "success": False,
                "output": None,
                "error": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": None, "error": "Execution timed out"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "output": None, "error": str(exc)}
        finally:
            if tmp and os.path.exists(tmp.name):
                os.unlink(tmp.name)
