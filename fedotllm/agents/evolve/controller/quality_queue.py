"""Queue technically valid patches for hour-long Fedot jobs.

Hunt never runs those jobs inline. A frozen/toy score only sets priority.
"""

from __future__ import annotations

import json
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import list_quality_task_ids
from fedotllm.agents.evolve.types import PatchCandidate


def campaign_workspace(workspace: Path) -> Path:
    """CLI ``run`` evaluates inside ``<campaign>/runs/<id>``; the queue is not."""

    workspace = workspace.resolve()
    if workspace.parent.name == "runs":
        return workspace.parent.parent
    return workspace


def queue_dir(workspace: Path) -> Path:
    path = campaign_workspace(workspace) / "quality_queue"
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
        "patch": patch_path.name,
        "status": "queued",
    }
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return job_path


def cheap_screen_not_quality_verdict(
    *,
    probe_status: str = "",
    reason: str = "",
    target_delta: object = None,
    stage: str = "",
) -> bool:
    """Toy 100-tree / bit-identical δ=0 / probe ``no_change`` never judge hour Fedot.

    Hour-long ``stage="quality"`` scores remain the only quality verdict.
    """

    if str(stage or "") == "quality":
        return False
    status = str(probe_status or "")
    why = str(reason or "")
    if status == "no_change" or why in {
        "behavior_probe_no_change",
        "no_affected_metric_signal",
        "technically_valid",
    }:
        return True
    if why.startswith("target_delta"):
        try:
            return target_delta is not None and abs(float(target_delta)) <= 1e-12
        except (TypeError, ValueError):
            return "0.0000" in why
    return False


def queue_priority(*, probe_status: str, toy_metric_moved: bool) -> str:
    """Toy/frozen scores never veto. They only rank the hour-long queue."""

    if probe_status == "changed" or toy_metric_moved:
        return "high"
    return "normal"


def list_queued_jobs(workspace: Path) -> list[dict]:
    """Queued hour jobs. Toy δ=0 / probe no_change stay in the list."""

    jobs: list[dict] = []
    directory = queue_dir(workspace)
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["_path"] = str(path)
        jobs.append(payload)
    rank = {"high": 0, "normal": 1, "low": 2}
    jobs.sort(
        key=lambda item: (
            rank.get(str(item.get("priority") or "normal"), 9),
            str(item.get("candidate_id") or ""),
        )
    )
    return jobs


def write_job_status(job_path: Path, status: str, extra: dict | None = None) -> None:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["status"] = status
    if extra:
        payload.update(extra)
    job_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
