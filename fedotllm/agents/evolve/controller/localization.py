"""Campaign hypothesis revisions and runtime parameter hydration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.storage.hypothesis import Hypothesis, as_row
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import MatchSite, ScoreResult

ScoreRunner = Callable[..., dict[str, ScoreResult]]

def start_revision(
    workspace: Path,
    parent: Hypothesis,
    lead: MatchSite,
    revision: int,
    reason: str,
) -> Hypothesis:
    child = Hypothesis(
        id=f"{parent.id}-r{revision}",
        parent_id=parent.id,
        lead=asdict(lead),
        claim=f"Refine after experiment result: {reason}",
        expected_effect="address measured feedback without resampling",
    )
    append_journal(
        workspace / "hypotheses.jsonl",
        {"event": "hypothesis", **as_row(child)},
    )
    return child


def hydrate_configuration_surfaces(
    stock: dict[str, ScoreResult],
    operation_hints: dict[str, tuple[str, ...]],
    *,
    source: Path,
    seed: int,
    max_operations: int,
    measure_stock_fn: ScoreRunner,
) -> int:
    """Collect one real estimator parameter surface per high-impact operation."""

    uses: dict[str, list[str]] = {}
    for task_id, operations in operation_hints.items():
        for operation in operations:
            uses.setdefault(operation, []).append(task_id)

    def observed(operation: str) -> bool:
        return any(
            str(row.get("operation") or "") == operation
            and bool(row.get("estimator_defaults"))
            for result in stock.values()
            for row in result.dataflow
        )

    # A task already instrumented for crash/dataflow localization must not be
    # rerun merely because its failing estimator could not expose get_params.
    refreshed_tasks: set[str] = {
        task_id
        for task_id, result in stock.items()
        if result.coverage or result.dataflow
    }
    refreshes = 0
    for operation, task_ids in sorted(
        uses.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if observed(operation):
            continue
        candidates = sorted(
            (
                task_id
                for task_id in task_ids
                if task_id not in refreshed_tasks
                and stock.get(task_id) is not None
                and stock[task_id].status in {"ok", "crash"}
            ),
            key=lambda task_id: len(operation_hints.get(task_id, ())),
        )
        if not candidates or refreshes >= max(0, max_operations):
            continue
        task_id = candidates[0]
        refreshed = measure_stock_fn(
            (task_id,),
            checkout=source,
            seed=seed,
            collect_coverage=True,
        ).get(task_id)
        refreshed_tasks.add(task_id)
        refreshes += 1
        if refreshed is not None and refreshed.dataflow:
            stock[task_id] = refreshed
    return refreshes
