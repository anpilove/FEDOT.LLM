"""Shared policy for trusted EvolveAgent snippet execution."""

from __future__ import annotations

import ast
import builtins
import re
import sys
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from pathlib import Path

BLOCKED_SOURCE_TOKENS = (
    "cases.json",
    "evolve.evaluation.scorer",
    "evolve.evaluation.tasks",
    "evolve.commands.recall",
    "benchmark_manifest",
    "quality_suite",
    "hidden_exam",
    "final_exam",
)

BLOCKED_IMPORT_PREFIXES = (
    "fedotllm.agents.evolve.evaluation",
    "fedotllm.agents.evolve.commands.recall",
    "fedotllm.agents.evolve.benchmark",
)

_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-[REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA[REDACTED]"),
    (re.compile(r"(?i)Bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (
        re.compile(
            r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+"
        ),
        r"\1=[REDACTED]",
    ),
)


def source_token_blocked(code: str) -> str | None:
    lowered = (code or "").lower()
    for token in BLOCKED_SOURCE_TOKENS:
        if token.lower() in lowered:
            return token
    return None


def _module_blocked(name: str) -> bool:
    normalized = str(name or "").strip()
    if not normalized:
        return False
    return any(
        normalized == prefix or normalized.startswith(f"{prefix}.")
        for prefix in BLOCKED_IMPORT_PREFIXES
    )


def ast_import_blocked(code: str) -> str | None:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and _module_blocked(node.module):
                return node.module
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_blocked(alias.name):
                    return alias.name
            continue
        if not isinstance(node, ast.Call):
            continue
        module_name: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str):
                    module_name = value
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            value = node.args[0].value
            if isinstance(value, str):
                module_name = value
        if module_name and _module_blocked(module_name):
            return module_name
    return None


def snippet_blocked(code: str) -> str | None:
    return source_token_blocked(code) or ast_import_blocked(code)


def redact_snippet_output(text: str) -> str:
    redacted = text or ""
    for pattern, replacement in _REDACT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class _SnippetImportBlocker(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if _module_blocked(fullname):
            raise ImportError(f"snippet import blocked: {fullname}")
        return None


@contextmanager
def snippet_runtime_guards(checkout: Path):
    """Block evaluator imports and reads outside the experiment checkout."""

    checkout = checkout.resolve()
    blocker = _SnippetImportBlocker()
    sys.meta_path.insert(0, blocker)
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):  # noqa: ANN001
        path = Path(file)
        resolved = path.resolve() if path.is_absolute() else (checkout / path).resolve()
        if checkout not in resolved.parents and resolved != checkout:
            raise PermissionError(f"snippet file access outside checkout: {resolved}")
        return real_open(resolved, *args, **kwargs)

    builtins.open = guarded_open
    try:
        yield
    finally:
        builtins.open = real_open
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass
