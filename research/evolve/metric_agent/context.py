from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from research.evolve.metric_agent.guard import deny_write
from research.evolve.metric_agent.types import Lead, ScoreResult

_FILE = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?$', re.MULTILINE)
_FALLBACK_RADIUS = 12
_MAX_FUNC_LINES = 80
_MAX_FRAMES = 3


def show_source(
    path: str | Path,
    *,
    checkout: Path,
    around: int = 1,
    radius: int = _FALLBACK_RADIUS,
) -> str:
    """Enclosing function/class if AST allows, else a small window around the line.

    Not ±40 of the file: that pulled in unrelated docstrings (Agentless is
    file → function → lines).
    """

    checkout = checkout.resolve()
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (checkout / raw).resolve()
    if deny_write(target, checkout=checkout):
        return ""
    if not target.is_file():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = _window(lines, around, radius=radius)
    numbered = [f"{i + 1:5d}|{lines[i]}" for i in range(start, end)]
    rel = target.relative_to(checkout).as_posix()
    return f"# {rel}\n" + "\n".join(numbered)


def inspect_trace(
    traceback: str,
    *,
    checkout: Path,
    radius: int = _FALLBACK_RADIUS,
) -> list[dict]:
    """Frames inside the FEDOT checkout only. Site-packages and scorer paths dropped."""

    checkout = checkout.resolve()
    frames: list[dict] = []
    for match in _FILE.finditer(traceback or ""):
        raw, line, func = match.group(1), int(match.group(2)), match.group(3) or ""
        try:
            raw_path = Path(raw)
            resolved = raw_path.resolve() if raw_path.is_absolute() else (checkout / raw_path).resolve()
            rel = resolved.relative_to(checkout).as_posix()
        except (OSError, ValueError):
            continue
        if deny_write(resolved, checkout=checkout):
            continue
        frames.append(
            {
                "file": rel,
                "line": line,
                "func": func,
                "source": show_source(resolved, checkout=checkout, around=line, radius=radius),
            }
        )
    return frames


def context_from_traceback(
    stock: ScoreResult,
    checkout: Path,
    *,
    max_chars: int = 24_000,
) -> str:
    frames = inspect_trace(stock.traceback, checkout=checkout)
    parts: list[str] = []
    if stock.detail:
        parts.append(f"Error: {stock.detail}")
    for frame in frames[-_MAX_FRAMES:]:
        header = f"{frame['file']}:{frame['line']} in {frame['func']}"
        parts.append(header + "\n" + frame["source"])
    return "\n\n".join(parts)[:max_chars]


def context_from_lead(lead: Lead, checkout: Path, *, max_chars: int = 24_000) -> str:
    source = show_source(lead.file_path, checkout=checkout, around=lead.line)
    parts = [
        f"Site: {lead.file_path}:{lead.line}",
    ]
    if lead.why:
        parts.append(f"Note: {lead.why}")
    if source:
        parts.append(source)
    siblings = _runtime_siblings(checkout, lead)
    if siblings:
        parts.append("Same class:\n" + siblings)
    from research.evolve.metric_agent.repo_map import (
        format_map,
        search_callers,
        search_field_usage,
    )

    func = lead.why.rsplit(" in ", 1)[-1].strip() if " in " in lead.why else ""
    if func:
        callers = search_callers(checkout, func, limit=6)
        if callers:
            parts.append("Callers:\n" + format_map(callers))
    for field in _attrs_in_source(source):
        usages = search_field_usage(checkout, field, limit=8)
        if usages:
            parts.append(f"Field {field}:\n" + format_map(usages))
    return "\n\n".join(parts)[:max_chars]


_SKIP_ATTRS = frozenset(
    {
        "shape",
        "dtype",
        "size",
        "ndim",
        "T",
        "name",
        "value",
        "real",
        "imag",
        "log",
        "logger",
        "cache",
        "nodes",
        "content",
        "parameters",
        "metadata",
        "tags",
        "parent",
        "copy",
        "update",
        "append",
    }
)


def _attrs_in_source(source: str, *, limit: int = 4) -> list[str]:
    body = "\n".join(line.split("|", 1)[-1] for line in (source or "").splitlines() if "|" in line)
    body = textwrap.dedent(body)
    if not body.strip():
        return []
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = node.attr
        if not name or name.startswith("_") or name in _SKIP_ATTRS or name in seen:
            continue
        seen.append(name)
        if len(seen) >= limit:
            break
    return seen


def _runtime_siblings(checkout: Path, lead: Lead, *, limit: int = 2) -> str:
    """Other fit/transform/predict methods on the same class — not the whole file."""

    from research.evolve.metric_agent.repo_map import _FIT_NAMES

    target = checkout / lead.file_path
    if deny_write(target, checkout=checkout) or not target.is_file():
        return ""
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    owner: ast.ClassDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= lead.line <= end:
            owner = node
            break
    if owner is None:
        return ""
    parts: list[str] = []
    for item in owner.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in _FIT_NAMES:
            continue
        end = getattr(item, "end_lineno", item.lineno) or item.lineno
        if item.lineno <= lead.line <= end:
            continue
        snippet = show_source(target, checkout=checkout, around=item.lineno)
        if snippet:
            parts.append(snippet)
        if len(parts) >= limit:
            break
    return "\n\n".join(parts)


def _window(lines: list[str], around: int, *, radius: int) -> tuple[int, int]:
    around = min(max(1, around), max(1, len(lines)))
    span = _enclosing_span(lines, around)
    if span is None:
        start = max(0, around - 1 - radius)
        end = min(len(lines), around + radius)
        return start, end
    start, end = span
    if end - start > _MAX_FUNC_LINES:
        mid = around - 1
        start = max(start, mid - radius)
        end = min(end, mid + radius + 1)
    return start, end


def _enclosing_span(lines: list[str], around: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return None
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        if start <= around <= end:
            width = end - start
            if best is None or width < (best[1] - best[0]):
                best = (start - 1, end)
    return best
