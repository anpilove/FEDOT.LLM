"""Queue technically valid patches for hour-long Fedot jobs.

Hunt never runs those jobs inline. A frozen/toy score only sets priority.
"""

from __future__ import annotations

import json
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import list_quality_task_ids
from fedotllm.agents.evolve.types import PatchCandidate


def queue_dir(workspace: Path) -> Path:
    path = workspace / "quality_queue"
    path.mkdir(parents=True, exist_ok=True)
    return path


def enqueue_quality_job(
    workspace: Path,
    *,
    candidate: PatchCandidate,
    patch_text: str,
    hint: str,
    priority: str,
    probe_status: str = "",
    toy_metric: str = "",
    task_ids: tuple[str, ...] | None = None,
) -> Path:
    """Record a pair job. Does not fit Fedot and does not judge quality."""

    ids = task_ids or list_quality_task_ids()
    directory = queue_dir(workspace)
    job_path = directory / f"{candidate.candidate_id}.json"
    patch_path = directory / f"{candidate.candidate_id}.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    payload = {
        "candidate_id": candidate.candidate_id,
        "file_path": candidate.file_path,
        "priority": priority,
        "hint": hint,
        "probe_status": probe_status,
        "toy_metric": toy_metric,
        "task_ids": list(ids),
        "patch": str(patch_path),
        "status": "queued",
    }
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return job_path


def queue_priority(*, probe_status: str, toy_metric_moved: bool) -> str:
    """Toy/frozen scores never veto. They only rank the hour-long queue."""

    if probe_status == "changed" or toy_metric_moved:
        return "high"
    return "normal"
