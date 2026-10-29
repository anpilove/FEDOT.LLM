"""Static dataflow hints for contracts that ordinary tests often miss.

The checks use source structure rather than FEDOT class or method names.  They
cover stale column metadata and positional joins that can lose row identity.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from fedotllm.agents.evolve.discovery.repo_map import Symbol, in_metric_scan, iter_symbols
from fedotllm.agents.evolve.types import MatchSite

_META = re.compile(r".+(_idx|_index|_indices|_indexes|_types)$", re.I)
_MATRIX = frozenset({"features", "predict", "predict_for_fit"})
_WHY = "assigns features/predict but does not rewrite column metadata it reads"
_INDEX_SUFFIXES = ("idx", "index", "indices", "indexes")
_ROW_IDENTITY_WHY = (
    "filters parent rows by index membership before concatenation without "
    "reordering them to one canonical index order"
)


def invariant_leads(checkout: Path, symbols: list[Symbol] | None = None) -> list[MatchSite]:
    items = symbols if symbols is not None else [
        item for item in iter_symbols(checkout) if item.kind in {"method", "function"}
    ]
    out: list[MatchSite] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        if not in_metric_scan(item.file_path):
            continue
        stale = _stale_metadata(checkout, item)
        if not stale:
            continue
        key = (item.file_path, item.line)
        if key in seen:
            continue
        seen.add(key)
        name = f"{item.parent + '.' if item.parent else ''}{item.name}"
        out.append(
            MatchSite(
                channel="invariant",
                file_path=item.file_path,
                line=item.line,
                why=f"{_WHY} in {name}",
                evidence=("features or predict assigned", "column metadata read and not assigned"),
            )
        )
    return [*row_identity_leads(checkout), *out]


def row_identity_leads(checkout: Path) -> list[MatchSite]:
    """Find positional multi-parent joins that do not align rows by identity.

    Membership masks preserve the local order of every input.  Applying such a
    selector independently to several parents and concatenating the results is
    only safe when all parents already use the same order.  A dictionary-based
    position map is treated as evidence that the implementation explicitly
    restores a canonical order.
    """

    leads: list[MatchSite] = []
    fedot_root = checkout / "fedot"
    if not fedot_root.is_dir():
        return leads
    for path in fedot_root.rglob("*.py"):
        rel = path.relative_to(checkout).as_posix()
        if not in_metric_scan(rel):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            lead = _row_identity_lead(rel, cls)
            if lead is not None:
                leads.append(lead)
    return leads


def _row_identity_lead(file_path: str, cls: ast.ClassDef) -> MatchSite | None:
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    membership_selectors = {
        method.name
        for method in methods
        if _calls_numpy(method, "isin") and _has_boolean_style_subscript(method)
    }
    if not membership_selectors or not any(
        _calls_numpy(method, "concatenate") for method in methods
    ):
        return None
    for method in methods:
        if not _calls_self_method_with_index(method, membership_selectors):
            continue
        # A position map is the usual linear-time proof that values are put in
        # canonical identity order instead of merely filtered in local order.
        if any(isinstance(node, ast.DictComp) for node in ast.walk(method)):
            continue
        return MatchSite(
            channel="invariant",
            file_path=file_path,
            line=method.lineno,
            why=f"{_ROW_IDENTITY_WHY} in {cls.name}.{method.name}",
            evidence=(
                "index membership mask preserves each parent's local row order",
                "multiple selected parent arrays are concatenated positionally",
            ),
            signals=("data_plane", "row_identity_contract"),
        )
    return None


def _calls_numpy(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == name
        for child in ast.walk(node)
    )


def _has_boolean_style_subscript(node: ast.AST) -> bool:
    assigned_calls = {
        target.id
        for child in ast.walk(node)
        if isinstance(child, ast.Assign)
        and isinstance(child.value, ast.Call)
        and isinstance(child.value.func, ast.Attribute)
        and child.value.func.attr == "isin"
        for target in child.targets
        if isinstance(target, ast.Name)
    }
    return bool(assigned_calls) and any(
        isinstance(child, ast.Subscript)
        and isinstance(child.slice, ast.Name)
        and child.slice.id in assigned_calls
        for child in ast.walk(node)
    )


def _calls_self_method_with_index(
    node: ast.AST, method_names: set[str]
) -> bool:
    comprehensions = (
        child
        for child in ast.walk(node)
        if isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp))
    )
    for comprehension in comprehensions:
        for child in ast.walk(comprehension.elt):
            if not isinstance(child, ast.Call) or not isinstance(
                child.func, ast.Attribute
            ):
                continue
            if child.func.attr not in method_names:
                continue
            if not isinstance(child.func.value, ast.Name) or child.func.value.id != "self":
                continue
            # A one-argument selector can legitimately obtain the canonical
            # index. The risky comprehension independently filters identity
            # and values for every parent.
            if len(child.args) >= 2 and any(
                _is_index_expression(argument) for argument in child.args
            ):
                return True
    return False


def _is_index_expression(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr.lower().endswith(
        _INDEX_SUFFIXES
    )


def _stale_metadata(checkout: Path, symbol: Symbol) -> frozenset[str]:
    path = checkout / symbol.file_path
    if not path.is_file():
        return frozenset()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    node = _fn_at(tree, symbol)
    if node is None:
        return frozenset()
    reads: set[str] = set()
    writes: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                _collect_store(target, writes)
        elif isinstance(child, ast.AnnAssign) and child.target is not None:
            _collect_store(child.target, writes)
        elif isinstance(child, ast.AugAssign):
            _collect_store(child.target, writes)
        elif isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
            reads.add(child.attr)
    if writes.isdisjoint(_MATRIX):
        return frozenset()
    stale = {name for name in reads if _META.match(name) and name not in writes}
    return frozenset(stale)


def _collect_store(target: ast.AST, writes: set[str]) -> None:
    if isinstance(target, ast.Attribute):
        writes.add(target.attr)
    elif isinstance(target, ast.Tuple):
        for elt in target.elts:
            _collect_store(elt, writes)


def _fn_at(tree: ast.AST, symbol: Symbol) -> ast.AST | None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.lineno == symbol.line and node.name == symbol.name:
            return node
    return None
