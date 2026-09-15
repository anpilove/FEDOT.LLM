"""Read-only, first-class repository tools used by Scout/Verifier/Fixer."""

from __future__ import annotations

import ast
import re
import subprocess
import time
from pathlib import Path

from fedotllm.agents.evolve.discovery.context import inspect_trace, show_source
from fedotllm.agents.evolve.execution.guard import deny_write
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.discovery.repo_map import (
    format_map,
    iter_symbols,
    repo_map,
    search_callers,
)
from fedotllm.agents.evolve.execution.run_code import _clean_env, fedot_python
from fedotllm.agents.evolve.types import SnippetResult, TestResult

MAX_TOOL_OUTPUT = 8_000
_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)")
_MISSING_IMPORT = re.compile(r"ImportError: cannot import name ['\"]([^'\"]+)")
_MISSING_ATTRIBUTE = re.compile(
    r"AttributeError: (?:type object )?['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)"
)
_BAD_CALL = re.compile(
    r"TypeError: ([A-Za-z_][A-Za-z0-9_.]*)\(\) "
    r"(?:got an unexpected keyword argument|takes |missing )"
)


def _symbol_query(query: str) -> str:
    text = (query or "").strip()
    text = re.sub(r"^(?:async\s+def|class|def)\s+", "", text)
    text = text.rstrip(":").split("(", 1)[0].strip()
    return text


def _exact_symbol_source(checkout: Path, symbol) -> str:
    path = checkout / symbol.file_path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tree = ast.parse("\n".join(lines) + "\n", filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return ""
    node = None
    if symbol.kind == "class":
        node = next(
            (item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == symbol.name),
            None,
        )
    elif symbol.kind == "function":
        node = next(
            (
                item
                for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == symbol.name
            ),
            None,
        )
    elif symbol.kind == "method":
        owner = next(
            (
                item
                for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == symbol.parent
            ),
            None,
        )
        if owner is not None:
            node = next(
                (
                    item
                    for item in owner.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == symbol.name
                ),
                None,
            )
    if node is None:
        return ""
    start = max(0, int(node.lineno) - 1)
    end = min(len(lines), int(getattr(node, "end_lineno", node.lineno) or node.lineno))
    numbered = [f"{index + 1:5d}|{lines[index]}" for index in range(start, end)]
    return f"# {symbol.file_path} — exact {symbol.kind} {symbol.parent + '.' if symbol.parent else ''}{symbol.name}\n" + "\n".join(numbered)


def _imported_names(code: str, module: str) -> list[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or (node.module or "") != module:
            continue
        names.extend(alias.name for alias in node.names if alias.name != "*")
    return list(dict.fromkeys(names))


def _failure_queries(code: str, error: str) -> list[str]:
    queries: list[str] = []
    missing_module = _MISSING_MODULE.search(error)
    if missing_module:
        queries.extend(_imported_names(code, missing_module.group(1)))
    missing_import = _MISSING_IMPORT.search(error)
    if missing_import:
        queries.append(missing_import.group(1))
    missing_attribute = _MISSING_ATTRIBUTE.search(error)
    if missing_attribute:
        queries.append(missing_attribute.group(1))
    bad_call = _BAD_CALL.search(error)
    if bad_call:
        queries.append(bad_call.group(1))
    return list(dict.fromkeys(query for query in queries if query))[:3]


def _exact_api_source(checkout: Path, query: str) -> str:
    """Return the exact class/method named by a Python exception when possible."""

    owner = ""
    name = query
    constructor_owner = ""
    if "." in query:
        owner, name = query.rsplit(".", 1)
        if name == "__init__":
            constructor_owner = owner
            name = owner
            owner = ""
    symbols = list(iter_symbols(checkout))
    if owner:
        exact = [
            item
            for item in symbols
            if item.parent == owner and item.name == name
        ]
        if not exact:
            exact = [item for item in symbols if item.name == owner and item.kind == "class"]
    else:
        exact = [item for item in symbols if item.name == name]
        exact.sort(key=lambda item: item.kind != "class")
    parts: list[str] = []
    for symbol in exact[:2]:
        if constructor_owner and symbol.kind == "class":
            contract = _constructor_contract_source(checkout, symbol, symbols)
            if contract:
                parts.append(contract)
                continue
        source = _exact_symbol_source(checkout, symbol) or show_source(
            symbol.file_path,
            checkout=checkout,
            around=symbol.line,
            radius=20,
        )
        if source:
            parts.append(source)
    return "\n\n".join(parts)


def _constructor_contract_source(checkout: Path, symbol, symbols: list) -> str:
    """Show dataclass/inherited fields when Python reports a bad constructor kwarg."""

    target = checkout / symbol.file_path
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        tree = ast.parse("\n".join(lines) + "\n", filename=str(target))
    except (OSError, SyntaxError, ValueError):
        return ""
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == symbol.name
        ),
        None,
    )
    if node is None:
        return ""

    def header_source(class_node: ast.ClassDef, file_lines: list[str], rel: str) -> str:
        first_method = next(
            (
                item
                for item in class_node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        end = int(first_method.lineno - 1) if first_method is not None else int(
            getattr(class_node, "end_lineno", class_node.lineno) or class_node.lineno
        )
        start = max(0, int(class_node.lineno) - 1)
        numbered = [f"{index + 1:5d}|{file_lines[index]}" for index in range(start, min(end, len(file_lines)))]
        return f"# {rel} — constructor fields for {class_node.name}\n" + "\n".join(numbered)

    parts = [header_source(node, lines, symbol.file_path)]
    base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
    for base_name in base_names[:2]:
        base_symbol = next(
            (
                item
                for item in symbols
                if item.kind == "class" and item.name == base_name
            ),
            None,
        )
        if base_symbol is None:
            continue
        base_path = checkout / base_symbol.file_path
        try:
            base_lines = base_path.read_text(encoding="utf-8", errors="replace").splitlines()
            base_tree = ast.parse("\n".join(base_lines) + "\n", filename=str(base_path))
        except (OSError, SyntaxError, ValueError):
            continue
        base_node = next(
            (
                item
                for item in base_tree.body
                if isinstance(item, ast.ClassDef) and item.name == base_name
            ),
            None,
        )
        if base_node is not None:
            parts.append(header_source(base_node, base_lines, base_symbol.file_path))
    return "\n\n".join(parts)


def snippet_failure_context(
    checkout: Path,
    result: SnippetResult,
    *,
    max_chars: int = 5_000,
) -> str:
    """Recover grounded FEDOT API context from a failed model-written snippet."""

    if result.status == "ok":
        return ""
    error = "\n".join(part for part in (result.stderr, result.detail) if part)
    parts: list[str] = [
        "Automatic FEDOT API recovery (ground truth from this checkout; "
        "repair the probe before drawing a conclusion):"
    ]

    missing_module = _MISSING_MODULE.search(error)
    if missing_module:
        basename = missing_module.group(1).rsplit(".", 1)[-1] + ".py"
        candidates: list[str] = []
        for path in sorted((checkout.resolve() / "fedot").rglob(basename)):
            try:
                rel = path.resolve().relative_to(checkout.resolve()).as_posix()
            except (OSError, ValueError):
                continue
            if path.is_file() and not deny_write(path, checkout=checkout):
                candidates.append(rel)
        if len(candidates) == 1:
            parts.append(
                f"Requested module path does not exist; unique matching runtime file: {candidates[0]}"
            )

    for frame in inspect_trace(result.stderr, checkout=checkout)[-2:]:
        source = str(frame.get("source") or "").strip()
        if source:
            parts.append(
                f"Failing FEDOT frame {frame['file']}:{frame['line']} "
                f"in {frame['func']}:\n{source}"
            )

    seen_sources: set[str] = set()
    for query in _failure_queries(result.code, error):
        source = _exact_api_source(checkout, query).strip()
        if source and source not in seen_sources:
            seen_sources.add(source)
            parts.append(f"Actual API for {query}:\n{source}")

    if len(parts) == 1:
        return ""
    output = "\n\n".join(parts)
    output = output[: max(0, max_chars)].rsplit("\n", 1)[0]
    _trace(
        {
            "tool": "api_recovery",
            "status": result.status,
            "queries": _failure_queries(result.code, error),
            "output": output,
        }
    )
    return output


def format_snippet_feedback(
    checkout: Path,
    result: SnippetResult,
    *,
    max_chars: int = MAX_TOOL_OUTPUT,
) -> str:
    """Keep the exception tail and recovered API visible in the next LLM turn."""

    raw = result.output
    if result.status == "ok":
        return raw[:max_chars]
    recovery = snippet_failure_context(
        checkout,
        result,
        max_chars=max(1_000, int(max_chars * 0.65)),
    )
    if not recovery:
        return raw[-max_chars:]
    remaining = max(500, max_chars - len(recovery) - 2)
    return raw[-remaining:] + "\n\n" + recovery


def _trace(event: dict) -> None:
    import os

    raw = os.environ.get("EVOLVE_AGENT_TRACE", "").strip()
    if raw:
        append_journal(Path(raw), {"event": "verifier_tool", **event})


def search_runtime(checkout: Path, query: str, *, limit: int = 20) -> str:
    """Literal search in FEDOT Python and frozen repository JSON metadata."""

    needle = (query or "").strip().lower()
    if not needle:
        return "<empty search query>"
    rows: list[str] = []
    root = checkout.resolve()
    paths = sorted(
        [
            *(root / "fedot").rglob("*.py"),
            *(root / "fedot" / "core" / "repository" / "data").glob("*.json"),
        ]
    )
    for path in paths:
        if deny_write(path, checkout=root):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if needle not in line.lower():
                continue
            rows.append(f"{rel}:{line_no}: {line.strip()[:300]}")
            if len(rows) >= max(1, limit):
                break
        if len(rows) >= max(1, limit):
            break
    output = "\n".join(rows) if rows else "<no runtime matches>"
    _trace({"tool": "search", "query": query, "output": output})
    return output


def symbol_runtime(checkout: Path, query: str, *, limit: int = 4) -> str:
    normalized = _symbol_query(query)
    if not normalized:
        return "<empty symbol query; provide a class, function, or method name>"
    symbols = repo_map(checkout, normalized, limit=max(1, limit))
    if not symbols:
        return "<no runtime matches>"
    parts: list[str] = ["Matches:\n" + format_map(symbols, limit=limit)]
    for symbol in symbols[:limit]:
        source = _exact_symbol_source(checkout, symbol) or show_source(
            symbol.file_path, checkout=checkout, around=symbol.line, radius=20
        )
        if source:
            parts.append(source)
    output = "\n\n".join(parts)[:MAX_TOOL_OUTPUT]
    _trace(
        {
            "tool": "symbol",
            "query": query,
            "normalized_query": normalized,
            "output": output,
        }
    )
    return output


def callers_runtime(checkout: Path, symbol: str, *, limit: int = 12) -> str:
    output = format_map(search_callers(checkout, symbol, limit=limit), limit=limit)
    _trace({"tool": "callers", "query": symbol, "output": output})
    return output


def docs_runtime(checkout: Path, query: str, *, limit: int = 4) -> str:
    """Retrieve docs, docstrings, and operation metadata shipped with FEDOT."""

    from fedotllm.agents.evolve.discovery.knowledge import retrieve_knowledge

    output = retrieve_knowledge(checkout, query, limit=limit, max_chars=MAX_TOOL_OUTPUT)
    _trace({"tool": "docs", "query": query, "output": output})
    return output


def focused_test(checkout: Path, node_id: str, *, timeout_s: float = 90) -> TestResult:
    """Run one existing FEDOT test node without allowing arbitrary commands."""

    node = (node_id or "").strip()
    path_part = node.split("::", 1)[0]
    target = (checkout.resolve() / path_part).resolve()
    root = checkout.resolve()
    if (
        not node
        or not path_part.startswith(("test/", "tests/"))
        or root not in target.parents
        or not target.is_file()
    ):
        result = TestResult(
            status="missing_tests",
            exit_code=None,
            output="focused_test requires an existing test/ or tests/ node id",
            cmd=node,
        )
        _trace({"tool": "focused_test", "node_id": node, "result": result.__dict__})
        return result
    cmd = [fedot_python(checkout), "-m", "pytest", node, "-q"]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=checkout,
            env=_clean_env(checkout),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = TestResult(
            status="timeout",
            exit_code=None,
            output=f"{exc.stdout or ''}\n{exc.stderr or ''}"[-MAX_TOOL_OUTPUT:],
            duration_s=time.perf_counter() - started,
            cmd=" ".join(cmd),
        )
    else:
        status = "passed" if proc.returncode == 0 else (
            "test_failures" if proc.returncode == 1 else "execution_error"
        )
        result = TestResult(
            status=status,
            exit_code=proc.returncode,
            output=f"{proc.stdout or ''}\n{proc.stderr or ''}"[-MAX_TOOL_OUTPUT:],
            duration_s=time.perf_counter() - started,
            cmd=" ".join(cmd),
        )
    _trace(
        {
            "tool": "focused_test",
            "node_id": node,
            "result": {
                "status": result.status,
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "cmd": result.cmd,
                "output": result.output,
            },
        }
    )
    return result
