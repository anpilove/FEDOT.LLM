"""FEDOT operation registry → source files. JSON dispatch, not AST call graph."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import deny_write
from fedotllm.agents.evolve.discovery.repo_map import in_metric_scan, skip_metric_noise
from fedotllm.agents.evolve.types import PatchSite

_REPO_JSON = (
    "model_repository.json",
    "data_operation_repository.json",
    "automl_repository.json",
)
_ALWAYS_FILES = (
    "fedot/core/pipelines/pipeline.py",
    "fedot/core/pipelines/node.py",
    "fedot/core/operations/operation.py",
    "fedot/core/operations/operation_parameters.py",
    "fedot/core/operations/hyperparameters_preprocessing.py",
    "fedot/core/operations/evaluation/evaluation_interfaces.py",
    "fedot/core/repository/data/default_operation_params.json",
)
_ALWAYS_DIRS = (
    "fedot/core/data/",
    "fedot/preprocessing/",
)


def registry_dir(checkout: Path) -> Path:
    return checkout / "fedot" / "core" / "repository" / "data"


def registry_files(checkout: Path) -> list[str]:
    """Strategy modules from JSON, their implementation imports, plus data/preproc/pipeline."""

    root = checkout.resolve()
    found: list[str] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        norm = rel.replace("\\", "/")
        if norm in seen or skip_metric_noise(norm):
            return
        if norm.endswith(".py") and not in_metric_scan(norm):
            return
        if "/gpu/" in norm:
            return
        target = root / norm
        if deny_write(target, checkout=root) or not target.is_file():
            return
        seen.add(norm)
        found.append(norm)

    json_dir = registry_dir(root)
    if not json_dir.is_dir():
        return []

    for name in _REPO_JSON:
        path = json_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for spec in (payload.get("metadata") or {}).values():
            strategies = spec.get("strategies") or []
            if isinstance(strategies, str):
                strategies = [strategies]
            if not strategies:
                continue
            module = strategies[0]
            if not isinstance(module, str) or module.startswith("["):
                continue
            add(module.replace(".", "/") + ".py")

    for rel in list(found):
        for imported in _impl_imports(root / rel, root):
            add(imported)

    for rel in _ALWAYS_FILES:
        add(rel)
    for prefix in _ALWAYS_DIRS:
        folder = root / prefix
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.py")):
            add(path.relative_to(root).as_posix())
    return found


def _impl_imports(path: Path, checkout: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        if "operation_implementations" not in module:
            continue
        rel = module.replace(".", "/") + ".py"
        if (checkout / rel).is_file():
            out.append(rel)
    return out


def registry_leads(checkout: Path) -> list[PatchSite]:
    files = registry_files(checkout)
    if not files:
        return []
    from fedotllm.agents.evolve.discovery.repo_map import iter_symbols, looks_metric

    allowed = set(files)
    leads: list[PatchSite] = []
    for symbol in iter_symbols(checkout):
        if symbol.file_path not in allowed or symbol.kind not in {"method", "function"}:
            continue
        if not looks_metric(checkout, symbol):
            continue
        leads.append(
            PatchSite(
                channel="registry",
                file_path=symbol.file_path,
                line=symbol.line,
                why=f"{symbol.kind} {symbol.parent + '.' if symbol.parent else ''}{symbol.name}",
                signals=_file_signals(symbol.file_path),
            )
        )
    params = "fedot/core/repository/data/default_operation_params.json"
    if params in allowed:
        leads.append(
            PatchSite(
                channel="registry",
                file_path=params,
                line=1,
                why="defaults default_operation_params.json",
                signals=("defaults", "registry"),
            )
        )
    return leads


def _file_signals(path: str) -> tuple[str, ...]:
    if path.endswith("default_operation_params.json"):
        return ("defaults", "registry")
    if path.startswith(("fedot/core/data/", "fedot/preprocessing/")):
        return ("data_plane",)
    if path.endswith(
        (
            "/pipelines/pipeline.py",
            "/pipelines/node.py",
            "/operations/operation.py",
        )
    ):
        return ("pipeline",)
    return ("registry",)
