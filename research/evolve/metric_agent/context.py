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
    clip: bool = True,
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
    start, end = _window(lines, around, radius=radius, clip=clip)
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


def context_from_lead(
    lead: Lead,
    checkout: Path,
    *,
    max_chars: int = 24_000,
    mode: str = "auto",
) -> str:
    kind = (mode or "auto").lower()
    if kind == "slice":
        return _context_slice(lead, checkout, max_chars=max_chars)
    if kind == "whole":
        return _context_whole(lead, checkout, max_chars=max_chars)
    if kind == "dep":
        return _context_dep(lead, checkout, max_chars=max_chars)
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


def _context_slice(lead: Lead, checkout: Path, *, max_chars: int) -> str:
    source = show_source(lead.file_path, checkout=checkout, around=lead.line)
    parts = [f"Site: {lead.file_path}:{lead.line}"]
    if lead.why:
        parts.append(f"Note: {lead.why}")
    if source:
        parts.append(source)
    return "\n\n".join(parts)[:max_chars]


def _context_whole(lead: Lead, checkout: Path, *, max_chars: int) -> str:
    target = checkout / lead.file_path
    parts = [f"Site: {lead.file_path}:{lead.line}"]
    if lead.why:
        parts.append(f"Note: {lead.why}")
    if deny_write(target, checkout=checkout) or not target.is_file():
        return "\n\n".join(parts)[:max_chars]
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = [f"{i + 1:5d}|{line}" for i, line in enumerate(lines)]
    parts.append(f"# {lead.file_path}\n" + "\n".join(numbered))
    return "\n\n".join(parts)[:max_chars]


def _context_dep(lead: Lead, checkout: Path, *, max_chars: int) -> str:
    """Enclosing method, same-class __init__, sibling fit/transform/predict, callers, callees, base."""

    from research.evolve.metric_agent.repo_map import format_map, search_callers, search_field_usage

    source = show_source(lead.file_path, checkout=checkout, around=lead.line, clip=False)
    parts = [f"Site: {lead.file_path}:{lead.line}"]
    if lead.why:
        parts.append(f"Note: {lead.why}")
    if source:
        parts.append(source)
    target = checkout / lead.file_path
    owner = _class_at_line(target, checkout, lead.line)
    if owner is not None:
        init = _method_on_class(owner, "__init__")
        if init is not None and not (init.lineno <= lead.line <= (getattr(init, "end_lineno", init.lineno) or init.lineno)):
            snippet = show_source(target, checkout=checkout, around=init.lineno, clip=False)
            if snippet:
                parts.append("Same class __init__:\n" + snippet)
        sibs = _runtime_siblings(checkout, lead, limit=8, include_init=False, clip=False)
        if sibs:
            parts.append("Same class:\n" + sibs)
        callees = _self_callees(target, checkout, owner, lead.line)
        if callees:
            parts.append("Callees:\n" + callees)
        base_src = _base_class_in_file(target, checkout, owner)
        if base_src:
            parts.append("Base class (same file):\n" + base_src)
    names = _lead_call_names(lead)
    caller_bits: list[str] = []
    for name in names:
        found = search_callers(checkout, name, limit=6)
        if found:
            caller_bits.append(format_map(found))
    if caller_bits:
        parts.append("Callers:\n" + "\n".join(caller_bits))
    for field in _attrs_in_source(source):
        usages = search_field_usage(checkout, field, limit=8)
        if usages:
            parts.append(f"Field {field}:\n" + format_map(usages))
    return "\n\n".join(parts)[:max_chars]


def _lead_call_names(lead: Lead) -> list[str]:
    why = (lead.why or "").strip()
    names: list[str] = []
    token = why.rsplit(" ", 1)[-1] if why else ""
    if token:
        names.append(token.split(".")[-1])
        if "." in token:
            names.append(token.split(".")[0])
    return [name for name in names if name and name.isidentifier()]


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


def _runtime_siblings(
    checkout: Path,
    lead: Lead,
    *,
    limit: int = 2,
    include_init: bool = False,
    clip: bool = True,
) -> str:
    """Other fit/transform/predict methods on the same class — not the whole file."""

    from research.evolve.metric_agent.repo_map import _FIT_NAMES

    target = checkout / lead.file_path
    owner = _class_at_line(target, checkout, lead.line)
    if owner is None:
        return ""
    wanted = set(_FIT_NAMES)
    if include_init:
        wanted.add("__init__")
    parts: list[str] = []
    for item in owner.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in wanted:
            continue
        end = getattr(item, "end_lineno", item.lineno) or item.lineno
        if item.lineno <= lead.line <= end:
            continue
        snippet = show_source(target, checkout=checkout, around=item.lineno, clip=clip)
        if snippet:
            parts.append(snippet)
        if len(parts) >= limit:
            break
    return "\n\n".join(parts)


def _class_at_line(target: Path, checkout: Path, line: int) -> ast.ClassDef | None:
    if deny_write(target, checkout=checkout) or not target.is_file():
        return None
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    owner: ast.ClassDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= line <= end:
            owner = node
            break
    return owner


def _method_on_class(owner: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for item in owner.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def _self_callees(target: Path, checkout: Path, owner: ast.ClassDef, line: int) -> str:
    enclosing = None
    for item in owner.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(item, "end_lineno", item.lineno) or item.lineno
        if item.lineno <= line <= end:
            enclosing = item
            break
    root: ast.AST = enclosing if enclosing is not None else owner
    names: list[str] = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
            continue
        name = node.func.attr
        if name and name not in names:
            names.append(name)
    parts: list[str] = []
    for name in names[:6]:
        method = _method_on_class(owner, name)
        if method is None:
            continue
        snippet = show_source(target, checkout=checkout, around=method.lineno, clip=False)
        if snippet:
            parts.append(snippet)
    return "\n\n".join(parts)


def _base_class_in_file(target: Path, checkout: Path, owner: ast.ClassDef) -> str:
    from research.evolve.metric_agent.repo_map import _FIT_NAMES

    bases = []
    for base in owner.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    if not bases:
        return ""
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    parts: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in bases:
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in _FIT_NAMES and item.name != "__init__":
                continue
            snippet = show_source(target, checkout=checkout, around=item.lineno, clip=False)
            if snippet:
                parts.append(snippet)
    return "\n\n".join(parts)


def _window(lines: list[str], around: int, *, radius: int, clip: bool = True) -> tuple[int, int]:
    around = min(max(1, around), max(1, len(lines)))
    span = _enclosing_span(lines, around)
    if span is None:
        start = max(0, around - 1 - radius)
        end = min(len(lines), around + radius)
        return start, end
    start, end = span
    if clip and end - start > _MAX_FUNC_LINES:
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
