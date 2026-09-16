"""Executor for paired hour-long Fedot quality jobs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.controller.quality_queue import (
    list_queued_jobs,
    write_job_status,
)
from fedotllm.agents.evolve.evaluation.fedot_quality import (
    bind_stock_cache,
    decision_from_quality_jobs,
    run_quality_jobs,
    run_quality_stock_jobs,
)
from fedotllm.agents.evolve.evaluation.quality_registry import (
    list_quality_task_ids,
    load_quality_registry,
    select_task_ids_for_job,
)
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import Decision, PatchCandidate, PatchEdit


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


def _normalize_patch_path(raw: str) -> str:
    path = raw.strip()
    for prefix in ("a/", "b/", "stock/", "patched/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path


def _parse_unified_hunks(patch_text: str) -> list[PatchEdit]:
    edits: list[PatchEdit] = []
    path = ""
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False

    def flush() -> None:
        nonlocal old_lines, new_lines, in_hunk
        if in_hunk and path and old_lines:
            edits.append(
                PatchEdit(
                    path,
                    "\n".join(old_lines) + "\n",
                    "\n".join(new_lines) + "\n",
                )
            )
        old_lines, new_lines, in_hunk = [], [], False

    for raw in patch_text.splitlines():
        if raw.startswith("--- "):
            flush()
            path = _normalize_patch_path(raw[4:])
            continue
        if raw.startswith("+++ "):
            continue
        if raw.startswith("@@"):
            flush()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("-"):
            old_lines.append(raw[1:])
        elif raw.startswith("+"):
            new_lines.append(raw[1:])
        elif raw.startswith("\\"):
            continue
        else:
            body = raw[1:] if raw.startswith(" ") else raw
            old_lines.append(body)
            new_lines.append(body)
    flush()
    return edits


def apply_queue_patch(checkout: Path, *, candidate_id: str, file_path: str, patch_text: str) -> bool:
    edits = _parse_unified_hunks(patch_text)
    if not edits:
        return False
    candidate = PatchCandidate(
        candidate_id=candidate_id,
        file_path=file_path or edits[0].file_path,
        edits=edits,
    )
    return apply_patch(checkout, candidate)


def _job_quality_task_ids(
    job: dict,
    pool: tuple[str, ...],
    registry,
) -> tuple[str, ...]:
    """Family-applicable tasks. Empty means skip, not drop."""

    recorded = tuple(str(item) for item in (job.get("task_ids") or ()) if str(item))
    selected = select_task_ids_for_job(job, requested=recorded or pool, registry=registry)
    if selected:
        return tuple(task_id for task_id in selected if task_id in pool) or selected
    return select_task_ids_for_job(job, requested=pool, registry=registry)


def drain_quality_queue(
    source: Path,
    workspace: Path,
    *,
    journal: Path | None = None,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    stock_cache: Path | None = None,
) -> dict:
    """Reuse matching stock cache, warm only missing tasks, then score patches.

    Toy δ=0 / probe ``no_change`` never skip a job. Hunt does not run this.
    Default: tabular OpenML for tabular patches, public TS for TS patches.
    Inapplicable tasks are omitted and are not a quality verdict.
    Re-scan the queue after each batch so later rsync arrivals are scored.
    An empty applicable set does not fall back to warming the whole pool.
    """

    if stock_cache is not None:
        bind_stock_cache(stock_cache)
    registry = load_quality_registry()
    pool = task_ids or list_quality_task_ids(registry)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    journal = journal or (workspace / "quality_jobs.jsonl")
    processed: set[str] = set()
    results: list[dict] = []
    stock_by_task: dict[str, dict] = {}

    while True:
        queued = [
            job
            for job in list_queued_jobs(workspace)
            if str(job.get("status") or "queued") in {"queued", ""}
            and str(job.get("candidate_id") or job.get("_path") or "") not in processed
        ]
        if not queued:
            break
        needed: list[str] = []
        for job in queued:
            for task_id in _job_quality_task_ids(job, pool, registry):
                if task_id not in needed:
                    needed.append(task_id)
        missing = tuple(task_id for task_id in needed if task_id not in stock_by_task)
        if missing:
            stock_rows = run_quality_stock_jobs(
                stock_checkout=source,
                task_ids=missing,
                n_jobs=n_jobs,
                cpu_quota=cpu_quota,
                registry=registry,
            )
            append_journal(
                journal,
                {
                    "event": "fedot_quality_stock",
                    "task_ids": list(missing),
                    "jobs": [
                        {
                            "spec": row["spec"],
                            "search_ran": row["search_ran"],
                            "stock_status": row["stock"]["status"],
                            "stock_score": row["stock"]["score"],
                            "cache_hit": bool(row.get("cache_hit")),
                        }
                        for row in stock_rows
                    ],
                },
            )
            for row in stock_rows:
                stock_by_task[row["spec"]["source_dataset"]] = row
        for job in queued:
            candidate_id = str(job.get("candidate_id") or "")
            processed.add(candidate_id or str(job.get("_path") or ""))
            job_path = Path(str(job["_path"]))
            recorded = Path(str(job.get("patch") or ""))
            sibling = job_path.with_suffix(".patch")
            named = job_path.parent / recorded.name if recorded.name else sibling
            if recorded.is_file():
                patch_path = recorded
            elif named.is_file():
                patch_path = named
            else:
                patch_path = sibling
            job_ids = _job_quality_task_ids(job, pool, registry)
            if not job_ids:
                write_job_status(
                    job_path,
                    "skipped",
                    {"reason": "inapplicable_tasks", "keep": None},
                )
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "skipped",
                        "reason": "inapplicable_tasks",
                        "keep": None,
                        "task_ids": [],
                        "priority": job.get("priority"),
                        "probe_status": job.get("probe_status"),
                        "toy_metric": job.get("toy_metric"),
                    }
                )
                continue
            write_job_status(job_path, "running")
            if not patch_path.is_file():
                write_job_status(job_path, "apply_failed", {"reason": "missing_patch"})
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "apply_failed",
                        "reason": "missing_patch",
                        "keep": False,
                    }
                )
                continue
            tree = create_experiment_checkout(
                source,
                workspace,
                run_id="quality-drain",
                candidate_id=candidate_id,
            )
            try:
                applied = apply_queue_patch(
                    tree,
                    candidate_id=candidate_id,
                    file_path=str(job.get("file_path") or ""),
                    patch_text=patch_path.read_text(encoding="utf-8"),
                )
                if not applied:
                    write_job_status(job_path, "apply_failed", {"reason": "patch_rejected"})
                    results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "apply_failed",
                            "reason": "patch_rejected",
                            "keep": False,
                        }
                    )
                    continue
                decision = measure_fedot_quality(
                    source,
                    tree,
                    journal=journal,
                    task_ids=job_ids,
                    n_jobs=n_jobs,
                    cpu_quota=cpu_quota,
                )
                status = "keep" if decision.keep else "drop"
                if decision.infrastructure_error:
                    status = "infrastructure_error"
                write_job_status(
                    job_path,
                    status,
                    {
                        "reason": decision.reason,
                        "keep": decision.keep,
                        "target_delta": decision.target_delta,
                        "regression_deltas": decision.regression_deltas,
                        "infrastructure_error": decision.infrastructure_error,
                    },
                )
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": status,
                        "reason": decision.reason,
                        "keep": decision.keep,
                        "target_delta": decision.target_delta,
                        "infrastructure_error": decision.infrastructure_error,
                        "task_ids": list(job_ids),
                        "priority": job.get("priority"),
                        "probe_status": job.get("probe_status"),
                        "toy_metric": job.get("toy_metric"),
                    }
                )
            finally:
                discard_experiment_checkout(tree, workspace=workspace, source=source)

    stock_rows = list(stock_by_task.values())
    return {
        "mode": "drain",
        "registry_identity": registry.identity,
        "task_ids": list(stock_by_task),
        "stock_ok": all(row["search_ran"] for row in stock_rows),
        "stock": [
            {
                "task_id": row["spec"]["source_dataset"],
                "status": row["stock"]["status"],
                "score": row["stock"]["score"],
                "search_ran": row["search_ran"],
                "cache_hit": bool(row.get("cache_hit")),
            }
            for row in stock_rows
        ],
        "stock_cache_hits": [
            row["spec"]["source_dataset"]
            for row in stock_rows
            if row.get("cache_hit")
        ],
        "stock_warmed": [
            row["spec"]["source_dataset"]
            for row in stock_rows
            if not row.get("cache_hit")
        ],
        "jobs": results,
    }
