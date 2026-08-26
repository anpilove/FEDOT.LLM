"""Cheap post-patch import check. Harness-only; not a hunt signal."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


def import_error(checkout: Path, rel: str, *, timeout_s: float = 20) -> str | None:
    """None if the patched module imports. Error text if it does not.

    Skips when the file is missing (unit tests that mock apply).
    """

    target = checkout / rel
    if not target.is_file():
        return None
    try:
        ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    mod = rel.replace("/", ".").removesuffix(".py")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    if not mod.startswith("fedot."):
        return None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            env=env,
            cwd=str(checkout),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "import timeout"
    if proc.returncode == 0:
        return None
    lines = (proc.stderr or proc.stdout or "import failed").strip().splitlines()
    return (lines[-1] if lines else "import failed")[:300]
