"""Aider/AutoCodeRover-style map: AST symbols, callers, field usage in FEDOT checkout."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import deny_write
EXCLUDED_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "docs",
    "examples",
    "jupyter_notebooks",
    "caching",
    "visualisation",
    "visualization",
    "explainability",
    "remote",
    "structural_analysis",
}
_SKIP_FILES = frozenset({"visualisation.py", "visualization.py"})
_MAP_LIMIT = 24
_FIT_NAMES = frozenset(
    {"fit", "transform", "predict", "fit_transform", "inverse_transform", "predict_proba", "fit_predict"}
)
_TREE_CAP = 400
_NOISE_IN_NAME = ("timer", "plot", "visual", "logger")
_TRAIN_ROOT_PREFIXES = (
    "fedot/core/pipelines/",
    "fedot/core/operations/",
    "fedot/core/composer/",
    "fedot/core/optimisers/",
    "fedot/preprocessing/",
    "fedot/api/",
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
        if not in_metric_scan(rel):
            continue
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


def reachable_from_fit(checkout: Path, *, limit: int = _TREE_CAP) -> list[Symbol]:
    """Static call tree from pipeline and operator fit/transform — both sides of if.

    Plots/cache/remote are not indexed. Plot helpers drop out unless fit them.
    `self.fit()` does not fan out to every model in the library.
    """

    symbols = [
        item
        for item in iter_symbols(checkout)
        if item.kind in {"method", "function"} and not skip_metric_noise(item.file_path)
    ]
    if not symbols:
        return []
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    for item in symbols:
        by_name[item.name].append(item)
    edges = _call_edges(checkout)
    start = _fit_roots(symbols)
    seen: dict[tuple[str, int], Symbol] = {}
    queue: deque[Symbol] = deque()

    def push(item: Symbol) -> None:
        loc = (item.file_path, item.line)
        if loc in seen or skip_metric_noise(item.file_path):
            return
        seen[loc] = item
        queue.append(item)

    for item in start:
        push(item)
    while queue and len(seen) < max(1, limit):
        current = queue.popleft()
        for called in edges.get((current.file_path, current.line), ()):
            for nxt in _resolve_call(called, current, by_name):
                push(nxt)
                if len(seen) >= max(1, limit):
                    return list(seen.values())
    return list(seen.values())


def fit_neighborhood(checkout: Path, *, limit: int = _TREE_CAP) -> list[tuple[Symbol, tuple[str, ...]]]:
    """Reachable training pipeline, plus the rest of fedot/ AST may miss (dynamic dispatch)."""

    tree = reachable_from_fit(checkout, limit=limit)
    tagged: list[tuple[Symbol, tuple[str, ...]]] = [(item, ("reachable",)) for item in tree]
    seen = {(item.file_path, item.line) for item in tree}
    classes = {(item.file_path, item.parent) for item in tree if item.parent}
    files = {item.file_path for item in tree}
    for item in iter_symbols(checkout):
        if item.kind not in {"method", "function"}:
            continue
        key = (item.file_path, item.line)
        if key in seen or skip_metric_noise(item.file_path) or not in_metric_scan(item.file_path):
            continue
        if any(bit in item.name.lower() for bit in _NOISE_IN_NAME):
            continue
        sigs: list[str] = []
        if "/operation_implementations/" in item.file_path:
            sigs.append("impl_class")
        if item.parent and (item.file_path, item.parent) in classes:
            sigs.append("same_class")
        elif item.file_path in files:
            sigs.append("same_module")
        else:
            sigs.append("core_scan")
        seen.add(key)
        tagged.append((item, tuple(sigs)))
    return tagged


def looks_metric(checkout: Path, symbol: Symbol) -> bool:
    """False only for timer/plot/log noise. Anything else on the fit path can change quality."""

    _ = checkout
    name = symbol.name.lower()
    return not any(bit in name for bit in _NOISE_IN_NAME)


def _fit_roots(symbols: list[Symbol]) -> list[Symbol]:
    """Training pipeline: any method in pipelines/operations/composer/optimisers/preprocessing/api."""

    roots: list[Symbol] = []
    for item in symbols:
        path = item.file_path.replace("\\", "/")
        if not in_metric_scan(path):
            continue
        if any(bit in item.name.lower() for bit in _NOISE_IN_NAME):
            continue
        if any(path.startswith(prefix) for prefix in _TRAIN_ROOT_PREFIXES):
            roots.append(item)
    if roots:
        return roots
    return [item for item in symbols if item.name in _FIT_NAMES and in_metric_scan(item.file_path)]


def _resolve_call(called: str, current: Symbol, by_name: dict[str, list[Symbol]]) -> list[Symbol]:
    hits = by_name.get(called) or []
    if not hits:
        return []
    same_file = [item for item in hits if item.file_path == current.file_path]
    if same_file:
        return same_file
    if called in _FIT_NAMES:
        return []
    if len(hits) == 1:
        return hits
    core = [
        item
        for item in hits
        if in_metric_scan(item.file_path) or "/operation_implementations/" in item.file_path
    ]
    return core[:6] if core else hits[:3]


def in_metric_scan(path: str) -> bool:
    """False only for plots, cache, remote, explainability — not training/metric."""

    norm = path.replace("\\", "/")
    if skip_metric_noise(norm):
        return False
    return norm.startswith("fedot/") and not norm.startswith("fedot/test")


def _call_edges(checkout: Path) -> dict[tuple[str, int], set[str]]:
    edges: dict[tuple[str, int], set[str]] = {}
    for path in _py_files(checkout):
        rel = path.relative_to(checkout).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _call_name(child)
                if name:
                    names.add(name)
            edges[(rel, node.lineno)] = names
    return edges


def skip_metric_noise(path: str) -> bool:
    """Plots, cache, docs — not model quality. Pipeline graphs are not plots."""

    lower = path.replace("\\", "/").lower()
    parts = lower.split("/")
    if any(part in EXCLUDED_DIR_PARTS for part in parts):
        return True
    name = parts[-1] if parts else ""
    if name in _SKIP_FILES:
        return True
    stem = name.removesuffix(".py")
    if "plot" in stem or stem.endswith("_visual") or stem.startswith("visual"):
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
        rel = path.relative_to(checkout).as_posix()
        if not in_metric_scan(rel):
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
