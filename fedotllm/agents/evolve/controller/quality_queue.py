"""Queue technically valid patches for hour-long Fedot jobs.

Hunt never runs those jobs inline. A frozen/toy score only sets priority.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.evaluation.quality_registry import (
    QualityRegistry,
    list_quality_task_ids,
    load_quality_registry,
    select_task_ids_for_job,
)
from fedotllm.agents.evolve.evaluation.quality_scope import (
    analyze_quality_scope,
    extract_change_tags,
    extract_changed_symbols,
)
from fedotllm.agents.evolve.types import PatchCandidate
from fedotllm.agents.evolve.storage.journal import write_json_atomic


def campaign_workspace(workspace: Path) -> Path:
    """CLI ``run`` evaluates inside ``<campaign>/runs/<id>``; the queue is not."""

    workspace = workspace.resolve()
    # A server service itself lives under evolve/runs/<campaign>; only hunt
    # children use the legacy <campaign>/runs/<hunt-id> layout.
    if any(
        (workspace / name).is_file()
        for name in ("service_manifest.json", "launch_audit.json")
    ):
        return workspace
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
    hypothesis: dict | None = None,
    source_root: Path | None = None,
    registry: QualityRegistry | None = None,
) -> Path:
    """Record a pair job. Does not fit Fedot and does not judge quality."""

    registry = registry or load_quality_registry()
    changed_files = list(
        dict.fromkeys(edit.file_path for edit in candidate.edits if edit.file_path)
    ) or ([candidate.file_path] if candidate.file_path else [])
    changed_symbols: list[str] = []
    ast_errors: list[str] = []
    if source_root is not None and candidate.edits:
        changed_symbols, ast_errors = extract_changed_symbols(
            source_root, candidate.edits
        )
    changed_code_tags = extract_change_tags(edit.new_code for edit in candidate.edits)
    impact = {
        "changed_files": changed_files,
        "changed_symbols": changed_symbols,
        "changed_code_tags": changed_code_tags,
        "ast_errors": ast_errors,
        "hypothesis": hypothesis or {},
    }
    scope = analyze_quality_scope(impact, registry)
    requested = (
        tuple(task_ids) if task_ids is not None else list_quality_task_ids(registry)
    )
    required_ids = select_task_ids_for_job(
        {**impact, "quality_scope": scope}, registry=registry
    )
    ids = select_task_ids_for_job(
        {**impact, "quality_scope": scope}, requested=requested, registry=registry
    )
    if not ids and task_ids is None:
        # An empty filtered subset is not evidence that a patch is harmless.
        ids = list_quality_task_ids(registry)
        scope = {
            **scope,
            "problem_families": ["classification", "regression", "ts_forecasting"],
            "reasons": [*scope["reasons"], "empty_scope_fallback"],
            "confidence": "fallback",
            "dataset_traits": [],
        }
        required_ids = ids
    directory = queue_dir(workspace)
    patch_sha256 = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    source_revision = ""
    source_sha256 = ""
    if source_root is not None:
        from fedotllm.agents.evolve.execution.checkout import (
            source_commit,
            source_fingerprint,
        )

        source_revision = source_commit(source_root)
        source_sha256 = source_fingerprint(source_root)
    check_identity = {
        "registry_identity": registry.identity,
        "registry_sha256": hashlib.sha256(
            json.dumps(asdict(registry), sort_keys=True).encode()
        ).hexdigest(),
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "required_task_ids": list(required_ids),
        "seed": registry.seed,
        "watchdog_seconds": registry.watchdog_seconds,
        "scope_policy_version": 2,
    }
    revision_id = candidate.candidate_id
    job_path = directory / f"{revision_id}.json"
    if job_path.is_file():
        try:
            existing = json.loads(job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"unreadable proposed improvement: {job_path}") from exc
        if not isinstance(existing, dict):
            raise ValueError(
                f"invalid proposed improvement: {job_path}: expected object"
            )
        old_patch = directory / str(existing.get("patch") or f"{revision_id}.patch")
        old_hash = str(existing.get("patch_sha256") or "")
        if not old_hash and old_patch.is_file():
            old_hash = hashlib.sha256(old_patch.read_bytes()).hexdigest()
        if (
            old_hash == patch_sha256
            and existing.get("check_identity") == check_identity
        ):
            # Idempotent enqueue must not reset a completed or running evaluation.
            return job_path
        if old_hash == patch_sha256:
            check_hash = hashlib.sha256(
                json.dumps(check_identity, sort_keys=True).encode()
            ).hexdigest()
            revision_id = f"{candidate.candidate_id}-q{check_hash[:12]}"
        else:
            revision_id = f"{candidate.candidate_id}-r{patch_sha256[:12]}"
        job_path = directory / f"{revision_id}.json"
        if job_path.is_file():
            try:
                existing_revision = json.loads(job_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"unreadable proposed improvement: {job_path}"
                ) from exc
            if (
                existing_revision.get("patch_sha256") == patch_sha256
                and existing_revision.get("check_identity") == check_identity
            ):
                return job_path
            raise FileExistsError(f"candidate revision id collision: {revision_id}")
    patch_path = directory / f"{revision_id}.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    readable_title = str(
        (hypothesis or {}).get("title")
        or (hypothesis or {}).get("claim")
        or hint
        or candidate.candidate_id
    ).strip()
    frozen_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "candidate_id": revision_id,
        "candidate_revision_id": revision_id,
        "proposal_id": candidate.candidate_id,
        "patch_revision": patch_sha256[:12],
        "patch_sha256": patch_sha256,
        "title": readable_title,
        "issue_url": str((hypothesis or {}).get("issue_url") or ""),
        "source_revision": source_revision,
        "source_sha256": source_sha256,
        "check_identity": check_identity,
        "scope_policy_version": 2,
        "registry_identity": registry.identity,
        "seed": int(registry.seed),
        "timeout_seconds": int(registry.timeout_seconds),
        "fedot_search_budget_seconds": int(registry.fedot_search_budget_seconds),
        "quality_scope_frozen": True,
        "scope_frozen_at": frozen_at,
        "created_at": frozen_at,
        "file_path": candidate.file_path,
        "changed_files": changed_files,
        "changed_symbols": changed_symbols,
        "changed_code_tags": changed_code_tags,
        "ast_errors": ast_errors,
        "hypothesis": hypothesis or {},
        "quality_scope": scope,
        "priority": priority,
        "hint": hint,
        "probe_status": probe_status,
        "toy_metric": toy_metric,
        "task_ids": list(ids),
        "required_task_ids": list(required_ids),
        "watchdog_seconds": registry.watchdog_seconds,
        "metric_contract": {
            task_id: {
                "metric": item.metric,
                "higher_is_better": item.higher_is_better,
                "delta_mode": item.delta_mode,
                "min_delta": item.min_delta,
            }
            for task_id in required_ids
            for item in registry.datasets
            if item.task_id == task_id
        },
        "patch": patch_path.name,
        "status": "queued",
    }
    write_json_atomic(job_path, payload)
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
        if path.name.startswith("._"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"unreadable proposed improvement: {path}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid proposed improvement: {path}: expected object")
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


def summarize_quality_queue(workspace: Path) -> dict:
    """Summarize terminal and pending quality proposals for run reports."""

    directory = campaign_workspace(workspace) / "quality_queue"
    status_counts: Counter[str] = Counter()
    pending: list[dict] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("._"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status_counts["unreadable"] += 1
                continue
            if not isinstance(payload, dict):
                status_counts["invalid"] += 1
                continue
            status = str(payload.get("status") or "unknown")
            status_counts[status] += 1
            if status in {"queued", "running", "partial", "skipped"}:
                pending.append(
                    {
                        "candidate_id": str(payload.get("candidate_id") or path.stem),
                        "status": status,
                        "hypothesis_id": str(
                            (payload.get("hypothesis") or {}).get("id") or ""
                        ),
                        "task_count": len(payload.get("task_ids") or ()),
                    }
                )
    return {
        "total_proposals": sum(status_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "pending": pending,
    }


def write_job_status(job_path: Path, status: str, extra: dict | None = None) -> None:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["status"] = status
    if extra:
        payload.update(extra)
    write_json_atomic(job_path, payload)


def quality_result_message(result: dict) -> str:
    """Explain a terminal quality outcome without conflating drops and failures."""

    candidate = str(result.get("candidate_id") or "unknown")
    hypothesis = result.get("hypothesis") or {}
    claim = str(hypothesis.get("claim") or "").strip()
    subject = f"Кандидат {candidate}"
    if claim:
        subject += f", гипотеза «{claim}»"
    task_count = len(result.get("task_ids") or ())
    reason = str(result.get("reason") or "без указанной причины")
    status = str(result.get("status") or "")
    outcome = str(result.get("quality_outcome") or "")
    improved_count = int(result.get("improved_task_count") or 0)

    if status in {"infrastructure_error", "failed"}:
        return (
            f"Оценка качества кандидата {candidate} не завершена ({reason}); "
            "не считать это подтверждённой регрессией или основанием принять патч. "
            f"Запланировано задач: {task_count}."
        )
    if status == "candidate_error":
        return (
            f"Качество кандидата {candidate} не подтверждено: ошибка на patched-версии "
            f"({reason}); автоматически повторять этот детерминированный сбой не следует."
        )
    if status == "evaluation_error":
        return (
            f"Качество кандидата {candidate} не оценено: не прошёл stock/baseline "
            f"({reason}); это не подтверждённая регрессия и не повод принимать патч."
        )
    if status == "apply_failed":
        return (
            f"Патч кандидата {candidate} не применился ({reason}); качество не оценено."
        )
    if status == "skipped":
        return f"Кандидат {candidate} пропущен: {reason}; качество не оценено."
    if status in {"partial", "blocked", "fedot_budget_timeout", "unverified_metric"}:
        return (
            f"Проверка улучшения {candidate} не завершена: {reason}; "
            f"нужны результаты всех {task_count} датасетов. Итогового вердикта качества нет."
        )
    if status == "keep":
        if outcome == "local_positive":
            return (
                f"Сохраняем {subject} как локальный положительный результат: "
                f"существенный прирост на {improved_count} из {task_count} задач; "
                "на остальных нет измеримого снижения."
            )
        return (
            f"Принимаем {subject}: существенный прирост на большинстве задач "
            f"({improved_count} из {task_count}); регрессий нет. "
            "Подробные метрики записаны в журнал."
        )
    if status == "drop" and reason == "no_quality_gain":
        return (
            f"Не принимаем {subject}: существенный прирост только на "
            f"{improved_count} из {task_count} задач; критерий локального или "
            "широкого эффекта не выполнен. Подробные метрики записаны в журнал."
        )
    if status == "drop" and reason.startswith("regression "):
        return (
            f"Не принимаем {subject}: защитная проверка обнаружила регрессию "
            f"({reason}). Подробные метрики записаны в журнал."
        )
    return (
        f"Не принимаем {subject}: критерии качества не пройдены ({reason}). "
        f"Оценено задач: {task_count}. Подробные метрики записаны в журнал."
    )
