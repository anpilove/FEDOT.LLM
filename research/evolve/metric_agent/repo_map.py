"""Aider/AutoCodeRover-style map: AST symbols, callers, field usage in FEDOT checkout."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from research.evolve.metric_agent.guard import deny_write
from research.evolve.metric_agent.types import Lead

EXCLUDED_DIR_PARTS = {".git", "__pycache__", ".pytest_cache", "docs", "examples", "jupyter_notebooks", "caching", "visualisation"}
_MAP_LIMIT = 24
_FIT_NAMES = frozenset(
    {"fit", "transform", "predict", "fit_transform", "inverse_transform", "predict_proba", "fit_predict"}
)


@dataclass(frozen=True)
class Symbol:
    file_path: str
    name: str
    kind: str
    line: int
    parent: str = ""


def iter_symbols(checkout: Path, *, root: str = "fedot") -> list[Symbol]:
    base = checkout / root
    if not base.is_dir():
        return []
    out: list[Symbol] = []
    for path in sorted(base.rglob("*.py")):
        if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        if deny_write(path, checkout=checkout):
            continue
        rel = path.relative_to(checkout).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out.append(Symbol(rel, node.name, "class", node.lineno))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _stub_method(item):
                            continue
                        out.append(Symbol(rel, item.name, "method", item.lineno, parent=node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _stub_method(node):
                    continue
                out.append(Symbol(rel, node.name, "function", node.lineno))
    return out


def repo_map(
    checkout: Path,
    query: str | list[str] | tuple[str, ...] = (),
    *,
    limit: int = _MAP_LIMIT,
) -> list[Symbol]:
    tokens = _tokens(query)
    ranked: list[tuple[int, int, Symbol]] = []
    for symbol in iter_symbols(checkout):
        blob = f"{symbol.file_path} {symbol.parent} {symbol.name}".lower()
        hits = sum(1 for token in tokens if token in blob) if tokens else 0
        core = 0 if symbol.file_path.startswith("fedot/core/") else 1
        ranked.append((-hits, core, symbol))
    ranked.sort(key=lambda row: (row[0], row[1], row[2].file_path, row[2].line))
    if tokens:
        ranked = [row for row in ranked if row[0] < 0]
        return [row[2] for row in ranked[: max(1, limit)]]
    visible = [
        row[2]
        for row in ranked
        if not row[2].name.startswith("_") or row[2].name in _FIT_NAMES
    ]
    return _spread(visible, limit)


def search_callers(checkout: Path, name: str, *, limit: int = 12) -> list[Symbol]:
    if not name:
        return []
    hits: list[Symbol] = []
    seen: set[tuple[str, int]] = set()
    for path in _py_files(checkout):
        rel = path.relative_to(checkout).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _call_name(node)
            if called != name:
                continue
            key = (rel, getattr(node, "lineno", 1))
            if key in seen:
                continue
            seen.add(key)
            hits.append(Symbol(rel, name, "call", node.lineno))
            if len(hits) >= limit:
                return hits
    return hits


def search_field_usage(checkout: Path, field: str, *, limit: int = 16) -> list[Symbol]:
    if not field:
        return []
    writes: list[Symbol] = []
    reads: list[Symbol] = []
    seen: set[tuple[str, int]] = set()
    for path in _py_files(checkout):
        rel = path.relative_to(checkout).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        write_lines = _assign_lines(tree, field)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != field:
                continue
            line = getattr(node, "lineno", 1)
            key = (rel, line)
            if key in seen:
                continue
            seen.add(key)
            kind = "field_write" if line in write_lines else "field"
            symbol = Symbol(rel, field, kind, line)
            (writes if kind == "field_write" else reads).append(symbol)
    ranked = _rank_field_hits(writes) + _rank_field_hits(reads)
    return ranked[: max(1, limit)]


def _assign_lines(tree: ast.AST, field: str) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Attribute) and sub.attr == field:
                    lines.add(getattr(sub, "lineno", 1))
    return lines


def _rank_field_hits(hits: list[Symbol]) -> list[Symbol]:
    def key(symbol: Symbol) -> tuple[int, str, int]:
        path = symbol.file_path
        if "/operations/" in path:
            bucket = 0
        elif path.startswith("fedot/core/data"):
            bucket = 2
        elif path.startswith("fedot/core/"):
            bucket = 1
        else:
            bucket = 3
        return (bucket, path, symbol.line)

    return sorted(hits, key=key)


def leads_from_map(symbols: list[Symbol]) -> list[Lead]:
    leads: list[Lead] = []
    seen: set[tuple[str, int]] = set()
    for symbol in symbols:
        key = (symbol.file_path, symbol.line)
        if key in seen:
            continue
        seen.add(key)
        leads.append(
            Lead(
                channel="repo_map",
                file_path=symbol.file_path,
                line=symbol.line,
                why=f"{symbol.kind} {symbol.parent + '.' if symbol.parent else ''}{symbol.name}",
            )
        )
    return leads


def format_map(symbols: list[Symbol], *, limit: int = 20) -> str:
    if not symbols:
        return "(empty repo map)"
    lines = [f"{item.file_path}:{item.line} {item.kind} {item.parent + '.' if item.parent else ''}{item.name}" for item in symbols[:limit]]
    return "\n".join(lines)


def _spread(symbols: list[Symbol], limit: int) -> list[Symbol]:
    """Round-robin across fedot/core packages so the map is not one subdirectory."""

    buckets: dict[str, list[Symbol]] = {}
    for symbol in symbols:
        buckets.setdefault(_area(symbol.file_path), []).append(symbol)
    for group in buckets.values():
        group.sort(
            key=lambda item: (
                0 if item.name in _FIT_NAMES else 1,
                1 if item.name.startswith("_") else 0,
                item.file_path,
                item.line,
            )
        )
    out: list[Symbol] = []
    seen: set[tuple[str, int]] = set()
    while len(out) < max(1, limit):
        progressed = False
        for key in list(buckets):
            group = buckets[key]
            if not group:
                continue
            item = group.pop(0)
            loc = (item.file_path, item.line)
            if loc in seen:
                continue
            seen.add(loc)
            out.append(item)
            progressed = True
            if len(out) >= max(1, limit):
                return out
        if not progressed:
            break
    return out


def _stub_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Skip abstract/empty methods so hunt does not spend leads on interfaces."""

    for dec in node.decorator_list:
        name = dec.id if isinstance(dec, ast.Name) else getattr(dec, "attr", "")
        if name in {"abstractmethod", "abstractclassmethod", "abstractproperty"}:
            return True
    stmts = [
        item
        for item in node.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(getattr(item, "value", None), ast.Constant)
            and isinstance(item.value.value, str)
        )
    ]
    if not stmts:
        return True
    if len(stmts) == 1 and isinstance(stmts[0], (ast.Pass, ast.Raise)):
        return True
    return False


def _area(path: str) -> str:
    """Leaf package of the file so implementations spread (models vs data_operations), not one `operations/` bucket."""

    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return path
    parent = parts[-2]
    if parent in {"fedot", "core"}:
        return parts[-1].removesuffix(".py")
    return parent


def _py_files(checkout: Path) -> list[Path]:
    root = checkout / "fedot"
    if not root.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        if deny_write(path, checkout=checkout):
            continue
        out.append(path)
    return out


def _tokens(query: str | list[str] | tuple[str, ...] = ()) -> list[str]:
    if isinstance(query, str):
        raw = query.replace("/", " ").replace(".", " ").split()
    else:
        raw = []
        for item in query:
            raw.extend(str(item).replace("/", " ").replace(".", " ").split())
    tokens: list[str] = []
    for word in raw:
        token = word.strip().lower()
        if len(token) < 4:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""
