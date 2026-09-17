"""Trusted research runner for Python snippets against one FEDOT experiment."""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import clean_subprocess_env, fedot_python
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.execution.snippet_policy import redact_snippet_output, snippet_blocked
from fedotllm.agents.evolve.types import SnippetResult

MAX_STEPS = int(os.environ.get("FEDOTLLM_EXPLORE_STEPS", "6"))
SNIPPET_TIMEOUT_S = float(os.environ.get("FEDOTLLM_EXPLORE_TIMEOUT", "120"))
MAX_OUTPUT_CHARS = int(os.environ.get("FEDOTLLM_EXPLORE_OUTPUT", "10000"))
WORKER_MODULE = "fedotllm.agents.evolve.execution._snippet_worker"


def _echo_last_expression(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name) and last.value.func.id == "print":
        return code
    lines = code.splitlines()
    expr = "\n".join(lines[last.lineno - 1 : last.end_lineno])
    if expr[: len(expr) - len(expr.lstrip())]:
        return code
    return "\n".join(lines[: last.lineno - 1] + [f"print({expr.strip()})"] + lines[last.end_lineno :])


def _append_trace(path: Path | None, result: SnippetResult) -> None:
    if path is None:
        raw = os.environ.get("EVOLVE_AGENT_TRACE")
        path = Path(raw) if raw else None
    if path is None:
        return
    row = {
        "event": "snippet",
        "status": result.status,
        "code": result.code,
        "stdout": redact_snippet_output(result.stdout),
        "stderr": redact_snippet_output(result.stderr),
        "exit_code": result.exit_code,
        "duration_s": result.duration_s,
        "detail": result.detail,
        "target_reached": result.target_reached,
    }
    append_journal(path, row)


def run_fedot_snippet(
    checkout: Path,
    code: str,
    history: list[str] | None = None,
    *,
    trace_path: Path | None = None,
    timeout_s: float | None = None,
    trace_target: dict | None = None,
) -> SnippetResult:
    """Execute trusted model-written Python in a disposable FEDOT checkout."""

    checkout = checkout.resolve()
    blob = code or ""
    blocked = snippet_blocked(blob)
    if blocked:
        result = SnippetResult(
            status="blocked",
            code=blob,
            detail=f"<blocked: evaluator / catalog import ({blocked})>",
        )
        _append_trace(trace_path, result)
        return result
    try:
        python = fedot_python(checkout)
    except (FileNotFoundError, OSError) as exc:
        result = SnippetResult(status="unavailable", code=blob, detail=f"<unavailable: {exc}>")
        _append_trace(trace_path, result)
        return result

    combined = "\n".join([*(history or []), _echo_last_expression(blob)])
    limit = SNIPPET_TIMEOUT_S if timeout_s is None else float(timeout_s)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="evolve-snippet-") as temp:
        script = Path(temp) / "snippet.py"
        script.write_text(combined, encoding="utf-8")
        cmd = [
            python,
            "-m",
            WORKER_MODULE,
            "--checkout",
            str(checkout),
            "--script",
            str(script),
        ]
        target_result = Path(temp) / "target-result.json"
        if trace_target:
            target_file = (checkout / trace_target["file_path"]).resolve()
            if not target_file.is_relative_to(checkout / "fedot"):
                raise ValueError("trace target must belong to this FEDOT checkout")
            cmd.extend(["--target-file", str(target_file), "--target-symbol", trace_target["symbol"],
                        "--target-result", str(target_result)])
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=checkout,
                env=clean_subprocess_env(checkout, repo_root=repo_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=limit)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
                result = SnippetResult(
                    status="timeout",
                    code=blob,
                    stdout=redact_snippet_output((stdout or "")[-MAX_OUTPUT_CHARS:]),
                    stderr=redact_snippet_output((stderr or "")[-MAX_OUTPUT_CHARS:]),
                    exit_code=None,
                    duration_s=time.perf_counter() - started,
                    detail=f"<timed out after {limit:g}s>",
                )
            else:
                result = SnippetResult(
                    status="ok" if proc.returncode == 0 else "runtime_error",
                    code=blob,
                    stdout=redact_snippet_output((stdout or "")[-MAX_OUTPUT_CHARS:]),
                    stderr=redact_snippet_output((stderr or "")[-MAX_OUTPUT_CHARS:]),
                    exit_code=proc.returncode,
                    duration_s=time.perf_counter() - started,
                    detail="" if proc.returncode == 0 else f"process exited {proc.returncode}",
                )
        except OSError as exc:
            result = SnippetResult(
                status="unavailable",
                code=blob,
                duration_s=time.perf_counter() - started,
                detail=f"<could not run: {exc}>",
            )
        if trace_target and target_result.exists():
            result.target_reached = json.loads(target_result.read_text())["target_reached"]
    _append_trace(trace_path, result)
    return result
