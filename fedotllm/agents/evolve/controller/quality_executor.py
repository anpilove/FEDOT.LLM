"""Executor for paired hour-long Fedot quality jobs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.evaluation.fedot_quality import (
    decision_from_quality_jobs,
    run_quality_jobs,
)
from fedotllm.agents.evolve.evaluation.quality_registry import (
    list_quality_task_ids,
    load_quality_registry,
)
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import Decision, PatchCandidate


def measure_fedot_quality(
    source: Path,
    experiment: Path,
    *,
    journal: Path | None = None,
    candidate: PatchCandidate | None = None,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
) -> Decision:
    """Hour-long Fedot is the only quality verdict. Hunt must enqueue, not inline-run."""

    registry = load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    rows = run_quality_jobs(
        stock_checkout=source,
        patch_checkout=experiment,
        task_ids=ids,
        n_jobs=n_jobs,
        cpu_quota=cpu_quota,
        registry=registry,
    )
    decision = decision_from_quality_jobs(rows)
    if journal is not None:
        append_journal(
            journal,
            {
                "event": "fedot_quality",
                "candidate": None if candidate is None else candidate.candidate_id,
                "registry_identity": registry.identity,
                "task_ids": list(ids),
                "decision": asdict(decision),
                "jobs": [
                    {
                        "spec": row["spec"],
                        "search_ran": row["search_ran"],
                        "stock_status": row["stock"]["status"],
                        "patched_status": row["patched"]["status"],
                        "stock_score": row["stock"]["score"],
                        "patched_score": row["patched"]["score"],
                    }
                    for row in rows
                ],
            },
        )
    return decision
