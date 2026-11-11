"""Offline screen of archived quality pairs against the frozen 40-task registry.

The three screen datasets are the first classification task, the first
regression task, and the first forecasting task in registry order. That
choice is fixed before any pair status is read. A valid regression on one
of those three tasks may reject the proposal early. A timeout or a missing
metric is counted on its own and is not a rejection. KEEP is available only
after every registered dataset has an improved pair.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

SCREEN_PROBLEMS = ("classification", "regression", "ts_forecasting")
FULL_DATASET_COUNT = 40
_IMPROVED = frozenset({"improved", "ok"})
_REGRESSION = frozenset({"regression"})
_TIMEOUT = frozenset({"timeout"})
_MISSING = frozenset({"missing", "missing_metric", "not_measured", "invalid"})
_REQUIRED_FIELDS = ("proposal", "task_id", "problem", "status")


def frozen_screen_tasks(datasets: Iterable[object]) -> dict[str, str]:
    """Return one task id per problem from registry order, ignoring outcomes."""

    chosen: dict[str, str] = {}
    for dataset in datasets:
        problem = str(getattr(dataset, "problem"))
        if problem in SCREEN_PROBLEMS and problem not in chosen:
            chosen[problem] = str(getattr(dataset, "task_id"))
        if len(chosen) == len(SCREEN_PROBLEMS):
            break
    if set(chosen) != set(SCREEN_PROBLEMS):
        missing = [problem for problem in SCREEN_PROBLEMS if problem not in chosen]
        raise ValueError(f"registry is missing a screen problem: {', '.join(missing)}")
    return {problem: chosen[problem] for problem in SCREEN_PROBLEMS}


def load_pair_archive(path: Path | str) -> list[dict]:
    """Read ``{"pairs": [...]}``. Each pair names a proposal, task, problem, and status."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("archive must be a JSON object with a pairs list")
    pairs: list[dict] = []
    for index, row in enumerate(payload["pairs"]):
        if not isinstance(row, dict):
            raise ValueError(f"pairs[{index}] must be an object")
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"pairs[{index}] missing {', '.join(missing)}")
        proposal = row["proposal"]
        task_id = row["task_id"]
        problem = row["problem"]
        status = row["status"]
        if not all(isinstance(value, str) and value for value in (proposal, task_id, problem, status)):
            raise ValueError(f"pairs[{index}] fields must be non-empty strings")
        if problem not in SCREEN_PROBLEMS:
            raise ValueError(f"pairs[{index}] problem must be one of {', '.join(SCREEN_PROBLEMS)}")
        pairs.append(
            {
                "proposal": proposal,
                "task_id": task_id,
                "problem": problem,
                "status": status,
            }
        )
    return pairs


def _status_kind(status: str) -> str:
    if status in _REGRESSION:
        return "regression"
    if status in _TIMEOUT:
        return "timeout"
    if status in _MISSING:
        return "missing_metric"
    if status in _IMPROVED:
        return "improved"
    return "missing_metric"


def _merge_kind(current: str | None, new: str) -> str:
    """A later improved pair must not hide a regression, timeout, or missing metric."""

    rank = {"regression": 4, "timeout": 3, "missing_metric": 2, "improved": 1}
    if current is None or rank[new] > rank[current]:
        return new
    return current


def _full_decision(kinds: Mapping[str, str], task_ids: tuple[str, ...]) -> str:
    observed = [kinds.get(task_id, "missing_metric") for task_id in task_ids]
    if any(kind == "regression" for kind in observed):
        return "regression"
    if any(kind == "timeout" for kind in observed):
        return "timeout"
    if any(kind == "missing_metric" for kind in observed):
        return "missing_metric"
    if len(observed) == FULL_DATASET_COUNT and all(kind == "improved" for kind in observed):
        return "keep"
    return "missing_metric"


def _early_decision(kinds: Mapping[str, str], screen_ids: tuple[str, ...]) -> str:
    observed = [kinds.get(task_id, "missing_metric") for task_id in screen_ids]
    if any(kind == "regression" for kind in observed):
        return "regression"
    if any(kind == "timeout" for kind in observed):
        return "timeout"
    if any(kind == "missing_metric" for kind in observed):
        return "missing_metric"
    return "continue"


def screen_pair_archive(pairs: list[dict], datasets: Iterable[object]) -> dict:
    """Compare a full 40-task reading with the frozen three-task screen."""

    dataset_rows = tuple(datasets)
    if len(dataset_rows) != FULL_DATASET_COUNT:
        raise ValueError(f"full check requires {FULL_DATASET_COUNT} datasets, found {len(dataset_rows)}")
    screen = frozen_screen_tasks(dataset_rows)
    screen_ids = tuple(screen.values())
    registry_problem = {str(getattr(row, "task_id")): str(getattr(row, "problem")) for row in dataset_rows}
    task_ids = tuple(registry_problem)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("registry task ids must be unique")

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    timeout_pairs = 0
    missing_metric_pairs = 0
    problem_mismatches = 0
    unknown_task_pairs = 0
    for pair in pairs:
        kind = _status_kind(pair["status"])
        if kind == "timeout":
            timeout_pairs += 1
        elif kind == "missing_metric":
            missing_metric_pairs += 1
        task_id = pair["task_id"]
        expected_problem = registry_problem.get(task_id)
        if expected_problem is None:
            unknown_task_pairs += 1
            continue
        if pair["problem"] != expected_problem:
            problem_mismatches += 1
            kind = "missing_metric"
        proposal_tasks = grouped[pair["proposal"]]
        proposal_tasks[task_id] = _merge_kind(proposal_tasks.get(task_id), kind)

    proposals = []
    disagreements = []
    pairs_saved = 0
    archived_pairs_not_required = 0
    screen_id_set = set(screen_ids)
    for proposal in sorted(grouped):
        kinds = grouped[proposal]
        early = _early_decision(kinds, screen_ids)
        full = _full_decision(kinds, task_ids)
        saved = FULL_DATASET_COUNT - len(screen_ids) if early == "regression" else 0
        pairs_saved += saved
        if saved:
            archived_pairs_not_required += sum(
                1 for task_id in kinds if task_id not in screen_id_set
            )
        record = {
            "proposal": proposal,
            "early": early,
            "full": full,
            "pairs_saved": saved,
            "early_reject": early == "regression",
            "keep": full == "keep",
        }
        proposals.append(record)
        if early == "regression" and full != "regression":
            disagreements.append({**record, "kind": "false_early_reject"})
        elif early != "regression" and full == "regression":
            disagreements.append({**record, "kind": "screen_missed_regression"})

    return {
        "screen_task_ids": screen,
        "screen_fixed_before_results": True,
        "full_dataset_count": FULL_DATASET_COUNT,
        "screen_dataset_count": len(screen_ids),
        "proposals": len(proposals),
        "pairs_total": len(pairs),
        "pairs_saved": pairs_saved,
        "archived_pairs_not_required": archived_pairs_not_required,
        "early_rejections": sum(1 for row in proposals if row["early_reject"]),
        "full_keeps": [row["proposal"] for row in proposals if row["keep"]],
        "early_keeps": [row["proposal"] for row in proposals if row["early"] == "keep"],
        "timeout_pairs": timeout_pairs,
        "missing_metric_pairs": missing_metric_pairs,
        "problem_mismatches": problem_mismatches,
        "unknown_task_pairs": unknown_task_pairs,
        "disagreements": disagreements,
        "keep_requires_full_check": True,
        "proposal_decisions": proposals,
    }
