"""Mandatory DEV safety gate for candidates found on a focused workload."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import Decision, ScoreResult


def enforce_global_dev_protect(
    experiment: Path,
    local_stock: dict[str, ScoreResult],
    local_patched: dict[str, ScoreResult],
    *,
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    source: Path,
    seed: int,
    journal: Path,
    measure_stock_fn: Callable,
    measure_patched_fn: Callable,
    verdict_fn: Callable,
) -> tuple[Decision, dict[str, ScoreResult], dict[str, ScoreResult]]:
    """Extend a focused DEV result to the immutable global safety suite."""

    safety_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    safety_stock = dict(local_stock)
    safety_patched = dict(local_patched)
    missing_stock = tuple(task_id for task_id in safety_ids if task_id not in safety_stock)
    missing_patched = tuple(
        task_id for task_id in safety_ids if task_id not in safety_patched
    )
    if missing_stock:
        safety_stock.update(
            measure_stock_fn(
                missing_stock,
                checkout=source,
                split="dev",
                seed=seed,
            )
        )
    if missing_patched:
        safety_patched.update(
            measure_patched_fn(
                missing_patched,
                checkout=experiment,
                split="dev",
                seed=seed,
            )
        )

    missing_results = [
        task_id
        for task_id in safety_ids
        if task_id not in safety_stock or task_id not in safety_patched
    ]
    if missing_results:
        decision = Decision(
            keep=False,
            reason="global_protect_missing_results: " + ",".join(missing_results),
            target_delta=None,
            stage="infrastructure",
            infrastructure_error=True,
            experiment_id=experiment.name,
        )
    else:
        decision = verdict_fn(
            safety_stock,
            safety_patched,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
        )
        decision.experiment_id = experiment.name
        if not decision.keep and decision.reason.startswith("regression"):
            decision.reason = (
                "regression global protect: "
                + decision.reason[len("regression ") :]
            )

    append_journal(
        journal,
        {
            "event": "global_dev_protect",
            "candidate_experiment": experiment.name,
            "tasks": list(safety_ids),
            "decision": asdict(decision),
        },
    )
    return decision, safety_stock, safety_patched
