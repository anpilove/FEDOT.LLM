"""Deterministic structural context cards for FEDOT source.

The card is derived only from the inspected checkout: Python AST, FEDOT's
operation registries, default parameters, search-space declarations and tests.
It contains no benchmark or oracle knowledge and performs no LLM calls.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.discovery.repo_map import Symbol, iter_symbols, search_callers
from fedotllm.agents.evolve.execution.guard import deny_write

_LIFECYCLE_METHODS = frozenset(
    {
        "fit",
        "fit_transform",
        "transform",
        "predict",
        "predict_for_fit",
        "predict_proba",
        "inverse_transform",
    }
)
_REPOSITORIES = (
    "model_repository.json",
    "data_operation_repository.json",
    "automl_repository.json",
    "gpu_models_repository.json",
)
_DEFAULTS = "fedot/core/repository/data/default_operation_params.json"
_SEARCH_SPACE = "fedot/core/pipelines/tuning/search_space.py"


@dataclass(frozen=True)
class ContextReference:
    file_path: str
    line: int
    symbol: str
    kind: str


@dataclass(frozen=True)
class OperationContext:
    operation_id: str
    implementation: str
    strategy: str
    defaults: tuple[tuple[str, Any], ...] = ()
    declared_params: tuple[str, ...] = ()


@dataclass(frozen=True)
class FedotContextCard:
    file_path: str
    symbol: str
    kind: str
    line: int
    lifecycle: tuple[str, ...]
    bases: tuple[str, ...]
    operations: tuple[OperationContext, ...]
    callers: tuple[ContextReference, ...]
    tests: tuple[ContextReference, ...]
    source: str

    def render(self, *, max_chars: int = 6_000) -> str:
        """Render a compact, stable prompt fragment."""

        target = self.symbol or "<file>"
        parts = [
            "FEDOT structural context (derived from the current checkout)",
            f"Target: {self.file_path}:{self.line} {self.kind} {target}",
            "Pipeline lifecycle: " + (", ".join(self.lifecycle) or "unknown"),
            "Base classes: " + (" -> ".join(self.bases) or "none found"),
        ]
        if self.operations:
            rows = []
            for operation in self.operations:
                defaults = json.dumps(dict(operation.defaults), sort_keys=True, default=str)
                declared = ", ".join(operation.declared_params) or "none found"
                rows.append(
                    f"- {operation.operation_id}: implementation={operation.implementation}; "
                    f"strategy={operation.strategy or 'unknown'}; defaults={defaults}; "
                    f"declared params=[{declared}]"
                )
            parts.append("Related public operation IDs:\n" + "\n".join(rows))
        if self.callers:
            parts.append("Static callers/references:\n" + _render_refs(self.callers))
        if self.tests:
            parts.append("Related frozen tests:\n" + _render_refs(self.tests))
        if self.source:
            parts.append("Compact source:\n" + self.source)
        text = "\n\n".join(parts)
        limit = max(200, int(max_chars))
        if len(text) <= limit:
            return text
        marker = "\n...[context card truncated]"
        return text[: limit - len(marker)].rsplit("\n", 1)[0] + marker


def build_fedot_context(
    checkout: Path,
    file_path: str | Path,
    *,
    symbol: str = "",
    caller_limit: int = 6,
    test_limit: int = 6,
) -> FedotContextCard | None:
    """Build a source-grounded card for one FEDOT file or symbol.

    ``file_path`` must resolve to a Python file under ``checkout/fedot``.
    A missing symbol falls back to a file-level outline rather than guessing a
    similarly named definition from another module.
    """

    checkout = checkout.resolve()
    target = _safe_target(checkout, file_path)
    if target is None:
        return None
    rel = target.relative_to(checkout).as_posix()
    tree = _parse(target)
    if tree is None:
        return None
    source_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = _select_symbol(rel, tree, symbol)
    all_symbols = iter_symbols(checkout)
    class_index, bases_by_class = _class_graph(checkout, all_symbols)
    target_classes = _target_classes(tree, selected, bases_by_class)
    lifecycle = _lifecycle_roles(rel, tree, selected)
    bases = _base_chain(selected, bases_by_class, class_index)
    operations = _operation_contexts(checkout, target_classes)
    callers = _caller_context(
        checkout,
        selected,
        operations,
        limit=max(0, caller_limit),
    )
    tests = _related_tests(
        checkout,
        rel,
        selected,
        operations,
        limit=max(0, test_limit),
    )
    return FedotContextCard(
        file_path=rel,
        symbol=_qualified_name(selected),
        kind=selected.kind if selected is not None else "file",
        line=selected.line if selected is not None else 1,
        lifecycle=lifecycle,
        bases=bases,
        operations=operations,
        callers=callers,
        tests=tests,
        source=_compact_source(rel, tree, source_lines, selected),
    )


def _safe_target(checkout: Path, file_path: str | Path) -> Path | None:
    raw = Path(file_path)
    target = raw.resolve() if raw.is_absolute() else (checkout / raw).resolve()
    try:
        rel = target.relative_to(checkout).as_posix()
    except ValueError:
        return None
    if not rel.startswith("fedot/") or target.suffix != ".py":
        return None
    if deny_write(target, checkout=checkout) or not target.is_file():
        return None
    return target


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None


def _symbols_in_tree(rel: str, tree: ast.Module) -> list[tuple[Symbol, ast.AST]]:
    rows: list[tuple[Symbol, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append((Symbol(rel, node.name, "class", node.lineno), node))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows.append((Symbol(rel, item.name, "method", item.lineno, node.name), item))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append((Symbol(rel, node.name, "function", node.lineno), node))
    return rows


def _select_symbol(rel: str, tree: ast.Module, query: str) -> Symbol | None:
    rows = [item for item, _ in _symbols_in_tree(rel, tree)]
    query = query.strip()
    if not query:
        return None
    exact = [item for item in rows if _qualified_name(item) == query]
    if not exact:
        exact = [item for item in rows if item.name == query]
    return sorted(exact, key=lambda item: (item.line, item.kind, item.parent))[0] if exact else None


def _qualified_name(symbol: Symbol | None) -> str:
    if symbol is None:
        return ""
    return f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name


def _class_graph(
    checkout: Path,
    symbols: list[Symbol],
) -> tuple[dict[str, tuple[Symbol, ...]], dict[str, tuple[str, ...]]]:
    class_index: dict[str, list[Symbol]] = {}
    bases: dict[str, tuple[str, ...]] = {}
    by_file: dict[str, ast.Module] = {}
    for symbol in symbols:
        if symbol.kind == "class":
            class_index.setdefault(symbol.name, []).append(symbol)
    for rel in sorted({item.file_path for item in symbols if item.kind == "class"}):
        tree = _parse(checkout / rel)
        if tree is None:
            continue
        by_file[rel] = tree
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            key = f"{rel}:{node.name}"
            bases[key] = tuple(filter(None, (_ast_name(base) for base in node.bases)))
    frozen_index = {
        name: tuple(sorted(rows, key=lambda item: (item.file_path, item.line)))
        for name, rows in class_index.items()
    }
    return frozen_index, bases


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _ast_name(node.value)
    return ""


def _target_classes(
    tree: ast.Module,
    selected: Symbol | None,
    bases_by_class: dict[str, tuple[str, ...]],
) -> set[str]:
    declared = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    owner = selected.parent if selected and selected.parent else (
        selected.name if selected and selected.kind == "class" else ""
    )
    roots = {owner} if owner else declared
    related = set(roots)
    changed = True
    while changed:
        changed = False
        for key, bases in bases_by_class.items():
            name = key.rsplit(":", 1)[-1]
            if name not in related and any(base in related for base in bases):
                related.add(name)
                changed = True
    return related


def _base_chain(
    selected: Symbol | None,
    bases_by_class: dict[str, tuple[str, ...]],
    class_index: dict[str, tuple[Symbol, ...]],
    *,
    depth: int = 4,
) -> tuple[str, ...]:
    owner = selected.parent if selected and selected.parent else (
        selected.name if selected and selected.kind == "class" else ""
    )
    if not owner:
        return ()
    start = next(
        (item for item in class_index.get(owner, ()) if item.file_path == selected.file_path),
        None,
    )
    if start is None:
        return ()
    out = [owner]
    current = start
    for _ in range(max(0, depth)):
        names = bases_by_class.get(f"{current.file_path}:{current.name}", ())
        if not names:
            break
        base = names[0]
        out.append(base)
        choices = class_index.get(base, ())
        if not choices:
            break
        current = next(
            (item for item in choices if item.file_path == current.file_path),
            choices[0],
        )
    return tuple(out)


def _lifecycle_roles(
    rel: str,
    tree: ast.Module,
    selected: Symbol | None,
) -> tuple[str, ...]:
    path_role = "FEDOT runtime"
    if "/operation_implementations/data_operations/" in rel:
        path_role = "pipeline data transformation implementation"
    elif "/operation_implementations/models/" in rel:
        path_role = "pipeline model implementation"
    elif "/operations/evaluation/" in rel:
        path_role = "public operation dispatch strategy"
    elif "/core/pipelines/" in rel:
        path_role = "pipeline orchestration"
    elif "/preprocessing/" in rel or "/core/data/" in rel:
        path_role = "data preparation and contract"
    methods: set[str] = set()
    for item, _ in _symbols_in_tree(rel, tree):
        if item.name not in _LIFECYCLE_METHODS:
            continue
        if selected is not None and selected.parent and item.parent != selected.parent:
            continue
        methods.add(item.name)
    return (path_role, *(f"{name} stage" for name in sorted(methods)))


def _strategy_mappings(checkout: Path) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []
    root = checkout / "fedot/core/operations/evaluation"
    if not root.is_dir():
        return rows
    for path in sorted(root.rglob("*.py")):
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(checkout).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "_operations_by_types"
                for target in node.targets
            ):
                continue
            owner = _enclosing_class(tree, node.lineno)
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                implementation = _ast_name(value)
                if implementation:
                    rows.append((key.value, implementation, f"{rel}:{owner}", node.lineno))
    return rows


def _enclosing_class(tree: ast.Module, line: int) -> str:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    return matches[0].name if matches else ""


def _registry_operations(checkout: Path) -> dict[str, dict]:
    root = checkout / "fedot/core/repository/data"
    out: dict[str, dict] = {}
    for name in _REPOSITORIES:
        path = root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for operation_id, spec in (payload.get("operations") or {}).items():
            if isinstance(spec, dict):
                out[str(operation_id)] = spec
    return out


def _default_params(checkout: Path) -> dict[str, dict]:
    try:
        payload = json.loads((checkout / _DEFAULTS).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _declared_params(checkout: Path, operation_ids: set[str]) -> dict[str, tuple[str, ...]]:
    tree = _parse(checkout / _SEARCH_SPACE)
    if tree is None:
        return {}
    out: dict[str, set[str]] = {operation_id: set() for operation_id in operation_ids}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value in out
                and isinstance(value, ast.Dict)
            ):
                out[key.value].update(
                    str(param.value)
                    for param in value.keys
                    if isinstance(param, ast.Constant) and isinstance(param.value, str)
                )
    return {key: tuple(sorted(values)) for key, values in out.items()}


def _operation_contexts(
    checkout: Path,
    target_classes: set[str],
) -> tuple[OperationContext, ...]:
    mappings = [row for row in _strategy_mappings(checkout) if row[1] in target_classes]
    registry = _registry_operations(checkout)
    defaults = _default_params(checkout)
    operation_ids = {row[0] for row in mappings}
    declared = _declared_params(checkout, operation_ids)
    rows = []
    for operation_id, implementation, strategy, _ in sorted(mappings):
        if operation_id not in registry:
            continue
        values = tuple(sorted(defaults.get(operation_id, {}).items()))
        rows.append(
            OperationContext(
                operation_id=operation_id,
                implementation=implementation,
                strategy=strategy,
                defaults=values,
                declared_params=declared.get(operation_id, ()),
            )
        )
    return tuple(rows)


def _caller_context(
    checkout: Path,
    selected: Symbol | None,
    operations: tuple[OperationContext, ...],
    *,
    limit: int,
) -> tuple[ContextReference, ...]:
    if limit <= 0:
        return ()
    refs: list[ContextReference] = []
    if selected is not None and selected.name not in _LIFECYCLE_METHODS | {"__init__"}:
        refs.extend(
            ContextReference(item.file_path, item.line, item.name, item.kind)
            for item in search_callers(checkout, selected.name, limit=limit)
        )
    for operation in operations:
        rel, _, owner = operation.strategy.partition(":")
        refs.append(ContextReference(rel, 1, owner, "operation_dispatch"))
    unique = {
        (item.file_path, item.line, item.symbol, item.kind): item
        for item in refs
    }
    return tuple(unique[key] for key in sorted(unique))[:limit]


def _related_tests(
    checkout: Path,
    rel: str,
    selected: Symbol | None,
    operations: tuple[OperationContext, ...],
    *,
    limit: int,
) -> tuple[ContextReference, ...]:
    if limit <= 0:
        return ()
    strong = {
        *(_qualified_name(selected).split(".") if selected else ()),
        *(operation.implementation for operation in operations),
    }
    public_ids = {operation.operation_id for operation in operations}
    module_name = Path(rel).stem
    patterns = [
        (0, re.compile(rf"\b{re.escape(needle)}\b"))
        for needle in sorted(strong)
        if len(needle) >= 3
    ]
    patterns.extend(
        (1, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])"))
        for needle in sorted(public_ids)
        if len(needle) >= 3
    )
    if len(module_name) >= 4:
        patterns.append((2, re.compile(rf"\b{re.escape(module_name)}\b")))
    hits: list[tuple[int, ContextReference]] = []
    roots = [checkout / "test", checkout / "tests"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            matches = [
                (priority, index)
                for index, line in enumerate(lines, start=1)
                for priority, pattern in patterns
                if pattern.search(line)
            ]
            if not matches:
                continue
            priority, index = min(matches)
            hits.append(
                (
                    priority,
                    ContextReference(
                        path.relative_to(checkout).as_posix(),
                        index,
                        _test_owner(lines, index),
                        "test",
                    ),
                )
            )
    hits.sort(key=lambda row: (row[0], row[1].file_path, row[1].line))
    return tuple(item for _, item in hits[:limit])


def _test_owner(lines: list[str], line: int) -> str:
    for text in reversed(lines[:line]):
        stripped = text.strip()
        if stripped.startswith("def test_"):
            return stripped.removeprefix("def ").split("(", 1)[0]
    return "test module"


def _compact_source(
    rel: str,
    tree: ast.Module,
    lines: list[str],
    selected: Symbol | None,
    *,
    max_lines: int = 100,
) -> str:
    rows = _symbols_in_tree(rel, tree)
    if selected is None:
        return "\n".join(
            f"{item.line:5d}|{item.kind} {_qualified_name(item)}"
            for item, _ in rows[:max_lines]
        )
    node = next((node for item, node in rows if item == selected), None)
    if node is None:
        return ""
    start = max(0, node.lineno - 1)
    end = min(len(lines), int(getattr(node, "end_lineno", node.lineno)), start + max_lines)
    return "\n".join(f"{index + 1:5d}|{lines[index]}" for index in range(start, end))


def _render_refs(refs: tuple[ContextReference, ...]) -> str:
    return "\n".join(
        f"- {item.file_path}:{item.line} {item.kind} {item.symbol}"
        for item in refs
    )
