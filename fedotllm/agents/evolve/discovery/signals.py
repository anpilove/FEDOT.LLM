"""Static lint and FEDOT pytest evidence used by discovery and gates."""

from __future__ import annotations

import ast
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from fedotllm.agents.evolve.discovery.context import inspect_trace
from fedotllm.agents.evolve.execution.process import clean_subprocess_env, fedot_python
from fedotllm.agents.evolve.types import MatchSite, TestResult

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FAILED = re.compile(r"^(FAILED|ERROR) (test/\S+)", re.MULTILINE)
_TEST_LEAD_LIMIT = 8


def parse_pytest_output(text: str, checkout: Path) -> list[MatchSite]:
    """Leads from FEDOT checkout pytest output. Ignores frames outside fedot/."""

    nodes = [match.group(2) for match in _FAILED.finditer(text or "")]
    frames = inspect_trace(text or "", checkout=checkout)
    why = nodes[0] if nodes else "pytest failure"
    leads: list[MatchSite] = []
    seen: set[tuple[str, int]] = set()
    for frame in reversed(frames):
        if not frame["file"].startswith("fedot/"):
            continue
        key = (frame["file"], int(frame["line"]))
        if key in seen:
            continue
        seen.add(key)
        leads.append(
            MatchSite(
                channel="fedot_test",
                file_path=frame["file"],
                line=int(frame["line"]),
                why=f"{why} in {frame['func']}",
            )
        )
        if len(leads) >= _TEST_LEAD_LIMIT:
            break
    return leads


def _as_text(blob: str | bytes | None) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", errors="replace")
    return blob


def failed_pytest_nodes(text: str) -> set[str]:
    return {match.group(2) for match in _FAILED.finditer(text or "")}


def pytest_failure_excerpt(
    text: str,
    failed_nodes: set[str],
    *,
    max_chars: int = 6_000,
) -> str:
    """Keep the actual assertion/traceback for every failed pytest node.

    The end of pytest output is mostly warnings and the short summary.  Sending
    only that tail made Fixer guess why a contract failed.  Extract each failure
    section by its test-function header and retain summary lines as a fallback.
    """

    clean = _ANSI.sub("", text or "")
    if not clean.strip() or not failed_nodes:
        return ""
    summary = [
        line.strip()
        for line in clean.splitlines()
        if line.startswith(("FAILED ", "ERROR "))
        and any(node in line for node in failed_nodes)
    ]
    blocks: list[str] = []
    per_node = max(600, max_chars // max(1, len(failed_nodes)))
    for node in sorted(failed_nodes):
        test_name = node.rsplit("::", 1)[-1]
        header = re.search(
            rf"(?m)^_{{3,}}\s+[^\n]*{re.escape(test_name)}[^\n]*\s+_{{3,}}\s*$",
            clean,
        )
        if header is not None:
            rest = clean[header.start() :]
            boundary = re.search(
                r"(?m)^_{3,}\s+[^\n]+\s+_{3,}\s*$|^={3,}\s+(?:warnings summary|short test summary info)",
                rest[header.end() - header.start() :],
            )
            end = (
                header.end() - header.start() + boundary.start()
                if boundary is not None
                else len(rest)
            )
            detail = rest[:end].strip()
            # Captured model logs can dwarf the traceback. Keep the assertion
            # before applying the budget, rather than returning a log tail.
            detail = re.split(
                r"(?m)^-{3,}\s+Captured (?:stdout|stderr|log) (?:setup|call|teardown)\s+-{3,}\s*$",
                detail,
                maxsplit=1,
            )[0].strip()
        else:
            detail = next((line for line in summary if node in line), node)
        if len(detail) > per_node:
            tail_size = max(300, per_node - 240)
            detail = (
                detail[:180]
                + "\n...[pytest traceback frames trimmed]...\n"
                + detail[-tail_size:]
            )
        blocks.append(f"[{node}]\n{detail}")
    if summary:
        blocks.append("[pytest short summary]\n" + "\n".join(summary))
    return "\n\n".join(blocks)[:max_chars]


def pytest_contract_source(
    checkout: Path,
    failed_nodes: set[str],
    *,
    max_chars: int = 3_000,
) -> str:
    """Return the exact frozen test functions named by pytest failures.

    Pytest can render a multiline assertion as only ``assert`` in captured
    output.  The unit test itself is public FEDOT contract evidence, unlike the
    hidden Evolve benchmark, so provide its bounded source to the repair model
    instead of asking it to guess what the assertion checked.
    """

    if not failed_nodes or max_chars <= 0:
        return ""
    root = checkout.resolve()
    test_root = (root / "test").resolve()
    blocks: list[str] = []
    per_node = max(400, max_chars // max(1, len(failed_nodes)))
    for node_id in sorted(failed_nodes):
        parts = node_id.split("::")
        if len(parts) < 2 or not parts[0].startswith("test/"):
            continue
        path = (root / parts[0]).resolve()
        try:
            path.relative_to(test_root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            from fedotllm.agents.evolve.evaluation.test_contracts import TEST_PATH, repaired_contract_source

            if parts[0] == TEST_PATH:
                source = repaired_contract_source(source)
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        test_name = parts[-1].split("[", 1)[0]
        function = next(
            (
                item
                for item in ast.walk(tree)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == test_name
            ),
            None,
        )
        if function is None:
            continue
        decorators = [item.lineno for item in function.decorator_list]
        start = min(decorators or [function.lineno])
        end = function.end_lineno or function.lineno
        lines = source.splitlines()
        body = "\n".join(
            f"{number:5d}|{lines[number - 1]}"
            for number in range(start, min(end, len(lines)) + 1)
        )
        if len(body) > per_node:
            body = body[:per_node].rsplit("\n", 1)[0]
        blocks.append(f"[{node_id}]\n{body}")
    return "\n\n".join(blocks)[:max_chars]


DEFAULT_PYTEST_TIMEOUT_S = 300.0


def pytest_result(
    checkout: Path, *, timeout_s: float | None = None, maxfail: int = 8
) -> TestResult:
    test_root = checkout / "test" / "unit"
    if not test_root.is_dir():
        return TestResult(
            status="missing_tests",
            exit_code=None,
            output="FEDOT test/unit directory is missing",
        )
    # FEDOT's frozen unit suite takes roughly two minutes on the reference
    # macOS environment.  A 120 second default therefore timed out a healthy
    # run immediately before pytest produced its result and turned every
    # candidate into an infrastructure DROP.
    limit = (
        timeout_s
        if timeout_s is not None
        else float(
            os.environ.get("EVOLVE_AGENT_PYTEST_TIMEOUT", str(DEFAULT_PYTEST_TIMEOUT_S))
        )
    )
    env = clean_subprocess_env(checkout, repo_root=Path(__file__).resolve().parents[4])
    cmd = [
        fedot_python(checkout),
        "-m",
        "pytest",
        "test/unit",
        "-q",
        "--tb=short",
        f"--maxfail={maxfail}",
        "--no-header",
    ]
    started = time.perf_counter()
    cmd_s = " ".join(shlex.quote(part) for part in cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            timeout=limit,
            check=False,
        )
        text = _as_text(proc.stdout) + "\n" + _as_text(proc.stderr)
        if proc.returncode == 0:
            status = "passed"
        elif proc.returncode == 1:
            status = "incomplete" if "stopping after" in text else "test_failures"
        elif proc.returncode == 2:
            status = "collection_error"
        elif proc.returncode == 5:
            status = "missing_tests"
        else:
            status = "execution_error"
        return TestResult(
            status=status,
            exit_code=proc.returncode,
            failed_nodes=failed_pytest_nodes(text),
            output=text,
            duration_s=time.perf_counter() - started,
            cmd=cmd_s,
            leads=parse_pytest_output(text, checkout),
        )
    except subprocess.TimeoutExpired as exc:
        text = _as_text(exc.stdout) + "\n" + _as_text(exc.stderr)
        return TestResult(
            status="timeout",
            exit_code=None,
            failed_nodes=failed_pytest_nodes(text),
            output=text,
            duration_s=time.perf_counter() - started,
            cmd=cmd_s,
            leads=parse_pytest_output(text, checkout),
        )


def pytest_snapshot(
    checkout: Path, *, timeout_s: float | None = None, maxfail: int = 8
) -> TestResult:
    return pytest_result(checkout, timeout_s=timeout_s, maxfail=maxfail)
