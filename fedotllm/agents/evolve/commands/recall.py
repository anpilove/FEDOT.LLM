"""Phase 1: offline file-level localization. Not imported by scout/fixer.

Gold is historical FEDOT sites (unique files). Used only here. Do not retune
`_impact` from one run.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.discovery.discover import discover_leads, static_leads
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import MatchSite

# Frozen unique files from local historical replacements. Eval-only.
_GOLD_FILES: tuple[str, ...] = (
    "fedot/api/api_utils/api_composer.py",
    "fedot/api/api_utils/assumptions/task_assumptions.py",
    "fedot/api/time.py",
    "fedot/core/data/data.py",
    "fedot/core/data/merge/data_merger.py",
    "fedot/core/operations/evaluation/classification.py",
    "fedot/core/operations/evaluation/evaluation_interfaces.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/__init__.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/decompose.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_imbalanced_class.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_selectors.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py",
    "fedot/core/operations/evaluation/operation_implementations/implementation_interfaces.py",
    "fedot/core/operations/evaluation/operation_implementations/models/__init__.py",
    "fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py",
    "fedot/core/operations/evaluation/operation_implementations/models/discriminant_analysis.py",
    "fedot/core/operations/evaluation/operation_implementations/models/knn.py",
    "fedot/core/operations/model.py",
    "fedot/preprocessing/preprocessing.py",
)

_KS = (1, 3, 5, 15)
_RANDOM_DRAWS = 20
_RANDOM_SEED = 42


def split_gold(files: tuple[str, ...] = _GOLD_FILES) -> dict[str, list[str]]:
    """Stable disjoint DEV/TEST. Hash of path, not hand-picked favorites."""

    dev: list[str] = []
    test: list[str] = []
    for path in files:
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        (dev if digest[0] % 2 == 0 else test).append(path)
    return {"dev": dev, "test": test}


def unique_files(leads: list[MatchSite]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lead in leads:
        if lead.file_path in seen:
            continue
        seen.add(lead.file_path)
        out.append(lead.file_path)
    return out


def file_rank(ranked: list[str], gold: str) -> int | None:
    try:
        return ranked.index(gold) + 1
    except ValueError:
        return None


def recall_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def metrics(ranks: list[int | None], *, ks: tuple[int, ...] = _KS) -> dict[str, float]:
    row = {f"recall@{k}": recall_at(ranks, k) for k in ks}
    row["mrr"] = mean_reciprocal_rank(ranks)
    row["covered"] = sum(rank is not None for rank in ranks) / len(ranks) if ranks else 0.0
    return row


def _score_ranked(ranked: list[str], gold: list[str]) -> dict[str, Any]:
    ranks = [file_rank(ranked, path) for path in gold]
    return {
        **metrics(ranks),
        "n": len(gold),
        "ranks": {path: rank for path, rank in zip(gold, ranks)},
    }


def _random_metrics(pool: list[str], gold: list[str], *, draws: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    acc = {f"recall@{k}": 0.0 for k in _KS}
    acc["mrr"] = 0.0
    acc["covered"] = 0.0
    for _ in range(draws):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        part = metrics([file_rank(shuffled, path) for path in gold])
        for key, value in part.items():
            acc[key] += value
    return {key: value / draws for key, value in acc.items()} | {"n": len(gold), "draws": draws}


def measure_localization(
    checkout: Path,
    *,
    seed: int = _RANDOM_SEED,
    draws: int = _RANDOM_DRAWS,
    inference=None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    present = [path for path in _GOLD_FILES if (checkout / path).is_file()]
    missing = [path for path in _GOLD_FILES if path not in present]
    splits = split_gold(tuple(present))
    structural = unique_files(static_leads(checkout))
    plus_inv = unique_files(static_leads(checkout, rank_metadata_stale=True))
    variants: dict[str, list[str] | None] = {
        "random_reachable": None,
        "structural": structural,
        "structural_invariant": plus_inv,
    }
    llm_pick = None
    if inference is not None:
        trace: dict[str, Any] = {}
        plus_llm = unique_files(discover_leads(checkout, inference=inference, limit=200, trace=trace))
        variants["structural_invariant_llm"] = plus_llm
        llm_pick = trace.get("llm_pick")
    out: dict[str, Any] = {
        "gold_present": present,
        "gold_missing": missing,
        "pool_files_structural": len(structural),
        "pool_files_invariant": len(plus_inv),
        "llm_pick": llm_pick,
        "splits": {name: list(paths) for name, paths in splits.items()},
        "by_split": {},
    }
    for split_name, gold in splits.items():
        row: dict[str, Any] = {
            "random_reachable": _random_metrics(structural, gold, draws=draws, seed=seed),
        }
        for name, ranked in variants.items():
            if ranked is None:
                continue
            row[name] = _score_ranked(ranked, gold)
        out["by_split"][split_name] = row
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        append_journal(workspace / "recall.jsonl", {"event": "recall", **out})
    return out
