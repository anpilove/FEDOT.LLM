"""Cheap post-patch import check. Harness-only; not a hunt signal."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import (
    clean_subprocess_env,
    fedot_python,
    run_worker,
)


def import_error(checkout: Path, rel: str, *, timeout_s: float = 20) -> str | None:
    """None if the patched module imports. Error text if it does not.

    Skips when the file is missing (unit tests that mock apply).
    """

    target = checkout / rel
    if not target.is_file():
        return None
    if target.suffix == ".json":
        try:
            json.loads(target.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            return f"JSONDecodeError: {exc.msg} at line {exc.lineno}"
        return None
    if target.suffix != ".py":
        return f"unsupported patched file type: {target.suffix or '<none>'}"
    try:
        ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    mod = rel.replace("/", ".").removesuffix(".py")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    if not mod.startswith("fedot."):
        return None
    env = clean_subprocess_env(checkout, repo_root=repo_root())
    proc = run_worker(
        [fedot_python(checkout), "-c", f"import {mod}"],
        cwd=checkout,
        env=env,
        timeout=timeout_s,
    )
    if proc.timed_out:
        return "import timeout"
    if proc.returncode == 0:
        return None
    lines = (proc.stderr or proc.stdout or "import failed").strip().splitlines()
    return (lines[-1] if lines else "import failed")[:300]
