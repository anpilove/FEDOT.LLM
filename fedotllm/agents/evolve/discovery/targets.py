"""Validated target refinement; never resolve names globally or from patch scores."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

from fedotllm.agents.evolve.types import PatchSite, VerificationResult


def resolve_verification_target(checkout: Path, lead: PatchSite, *, file_path: str = "", line: int | None = None, symbol: str = ""):
    root = checkout.resolve()
    original = (root / lead.file_path).resolve()
    path = (root / (file_path or lead.file_path)).resolve()
    if not path.is_relative_to(root / "fedot") or path.suffix != ".py":
        raise ValueError("verification target must be a FEDOT runtime Python file")
    if path != original:
        raise ValueError("cross-file target refinement requires a separate source lead")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    nominated = int(line if line is not None else lead.line)
    matches = []

    def walk(node, parents=()):
        for child in ast.iter_child_nodes(node):
            named = isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            names = (*parents, child.name) if named else parents
            if named and child.lineno <= nominated <= (child.end_lineno or child.lineno):
                matches.append((len(names), child.lineno, ".".join(names)))
            walk(child, names)

    walk(tree)
    if not matches:
        raise ValueError("verification target line is outside any source class/function")
    _, _, qualified = max(matches)
    if symbol.strip() and symbol.strip() != qualified:
        raise ValueError("verification target symbol and source line disagree")
    relative = path.relative_to(root).as_posix()
    target = {
        "file_path": relative, "line": nominated, "symbol": qualified,
        "source_file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "original_file_path": lead.file_path, "original_line": lead.line,
    }
    return replace(lead, file_path=relative, line=nominated), target


def apply_verified_target(checkout: Path, lead: PatchSite, verification: VerificationResult) -> PatchSite:
    """Revalidate the persisted identity before Fixer, coverage or bookkeeping."""
    target = verification.resolved_target
    if verification.status != "verified_bug" or not target:
        return lead
    refined, current = resolve_verification_target(
        checkout, lead, file_path=str(target["file_path"]),
        line=int(target["line"]), symbol=str(target["symbol"]),
    )
    if target.get("source_file_sha256") != current["source_file_sha256"]:
        raise ValueError("verified target source changed before candidate planning")
    return refined
