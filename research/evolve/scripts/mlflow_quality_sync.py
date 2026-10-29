"""Idempotently mirror FEDOT quality journal events into local MLflow runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _find_or_create(
    client, experiment_id: str, key: str, *, name: str, tags: dict[str, str]
):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.controller_key = '{key}'",
        max_results=1,
    )
    if runs:
        return runs[0]
    return client.create_run(
        experiment_id=experiment_id,
        run_name=name[:250],
        tags={"controller_key": key, **tags},
    )


def _log_metric_if_changed(
    client, run_id: str, name: str, value: float, *, step: int
) -> None:
    history = client.get_metric_history(run_id, name)
    if history and history[-1].value == value and history[-1].step == step:
        return
    client.log_metric(run_id, name, value, step=step)


def sync_quality_to_mlflow(
    journal_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    tracking_uri: str,
    experiment_name: str = "FEDOT quality controller",
) -> dict[str, int]:
    """Sync stable parent and dataset-attempt runs; never combine metric families."""

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:  # local Markdown/JSON reporting remains independent
        raise RuntimeError(
            "MLflow is not installed; install it locally to enable this sync"
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    candidates = summary.get("candidates") or {}
    side_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in journal_rows:
        if str(row.get("event") or "").startswith("quality_task_side_"):
            candidate_id = str(row.get("candidate") or "")
            if candidate_id:
                side_events[candidate_id].append(row)

    counts = {"parents": 0, "children": 0}
    for candidate_id, meta in candidates.items():
        revision = str(meta.get("candidate_revision_id") or candidate_id)
        parent_key = _key("quality-parent", revision)
        tags = {
            "candidate_id": candidate_id,
            "proposal_id": str(meta.get("proposal_id") or candidate_id),
            "candidate_revision_id": revision,
            "patch_revision": str(meta.get("patch_revision") or ""),
            "patch_sha256": str(meta.get("patch_sha256") or ""),
            "source_revision": str(meta.get("source_revision") or ""),
            "registry_identity": str(meta.get("registry_identity") or ""),
            "quality_outcome": str(
                meta.get("derived_quality_outcome")
                or meta.get("quality_outcome")
                or "pending"
            ),
            "queue_status": str(
                meta.get("queue_status") or meta.get("status") or "unknown"
            ),
            "issue_url": str(meta.get("issue_url") or ""),
            "hypothesis_json": json.dumps(
                meta.get("hypothesis") or {}, ensure_ascii=False
            ),
            "quality_scope_json": json.dumps(
                meta.get("quality_scope") or {}, ensure_ascii=False
            ),
            "task_ids_json": json.dumps(meta.get("task_ids") or [], ensure_ascii=False),
        }
        parent = _find_or_create(
            client,
            experiment_id,
            parent_key,
            name=str(meta.get("title") or candidate_id),
            tags={"mlflow.runName": str(meta.get("title") or candidate_id), **tags},
        )
        parent_id = parent.info.run_id
        for tag, value in tags.items():
            client.set_tag(parent_id, tag, value)
        planned = int(meta.get("planned_task_count") or len(meta.get("task_ids") or []))
        completed = int(meta.get("completed_task_pair_count") or 0)
        measured = int(meta.get("measured_task_pair_count") or 0)
        _log_metric_if_changed(
            client, parent_id, "datasets_planned", float(planned), step=0
        )
        _log_metric_if_changed(
            client, parent_id, "datasets_completed", float(completed), step=completed
        )
        _log_metric_if_changed(
            client, parent_id, "datasets_measured", float(measured), step=completed
        )
        for metric_name, field in (
            ("quality_wins", "derived_improved_task_count"),
            ("quality_losses", "derived_regressed_task_count"),
        ):
            count = meta.get(field)
            if isinstance(count, (int, float)):
                _log_metric_if_changed(
                    client, parent_id, metric_name, float(count), step=completed
                )
        if meta.get("reason"):
            client.set_tag(parent_id, "decision_reason", str(meta["reason"]))
        if meta.get("quality_message"):
            client.log_text(
                parent_id, str(meta["quality_message"]), "quality_message.txt"
            )
        client.log_text(
            parent_id,
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            "proposal_report.json",
        )
        status = str(meta.get("queue_status") or meta.get("status") or "")
        if status in {"queued", "running", "unknown", ""}:
            if parent.info.status != "RUNNING":
                client.update_run(parent_id, status="RUNNING")
        else:
            client.set_terminated(parent_id, status="FINISHED")
        counts["parents"] += 1

        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        specs: dict[tuple[str, int], dict[str, Any]] = {}
        for event in side_events.get(candidate_id, []):
            task_id = str(event.get("task_id") or "")
            side = str(event.get("side") or "")
            if not task_id or side not in {"stock", "patch"}:
                continue
            attempt = int(event.get("attempt") or 1)
            pair_key = (task_id, attempt)
            specs[pair_key] = event.get("spec") or specs.get(pair_key) or {}
            phase = str(event.get("event") or "")
            if phase.endswith(("finished", "reused")):
                grouped[pair_key][side] = event.get("result") or {}
            elif side not in grouped[pair_key]:
                grouped[pair_key][side] = {"status": "running"}

        for (task_id, attempt), sides in grouped.items():
            child_key = _key("quality-child", revision, task_id, str(attempt))
            spec = specs.get((task_id, attempt), {})
            child_tags = {
                "mlflow.parentRunId": parent_id,
                "candidate_revision_id": revision,
                "task_id": task_id,
                "attempt": str(attempt),
                "metric": str(spec.get("metric") or "unknown"),
                "registry_identity": str(meta.get("registry_identity") or ""),
            }
            child = _find_or_create(
                client,
                experiment_id,
                child_key,
                name=f"{candidate_id} / {task_id} / attempt {attempt}",
                tags=child_tags,
            )
            child_id = child.info.run_id
            for tag, value in child_tags.items():
                client.set_tag(child_id, tag, value)
            for side in ("stock", "patch"):
                result = sides.get(side) or {}
                client.set_tag(
                    child_id, f"{side}_status", str(result.get("status") or "pending")
                )
                observations = result.get("metric_observations") or {}
                if result.get("worker_log_path") or observations.get("worker_log_path"):
                    client.set_tag(
                        child_id,
                        f"{side}_worker_log_path",
                        str(
                            result.get("worker_log_path")
                            or observations["worker_log_path"]
                        ),
                    )
                score = result.get("score")
                if isinstance(score, (int, float)) and math.isfinite(float(score)):
                    _log_metric_if_changed(
                        client, child_id, f"{side}_score", float(score), step=attempt
                    )
            stock = sides.get("stock") or {}
            patch = sides.get("patch") or {}
            stock_score, patch_score = stock.get("score"), patch.get("score")
            if (
                stock.get("status") == "ok"
                and patch.get("status") == "ok"
                and isinstance(stock_score, (int, float))
                and math.isfinite(float(stock_score))
                and isinstance(patch_score, (int, float))
                and math.isfinite(float(patch_score))
            ):
                delta = float(patch_score) - float(stock_score)
                if spec.get("higher_is_better") is False:
                    delta = -delta
                if spec.get("delta_mode") == "relative":
                    delta /= max(abs(float(stock_score)), 1e-12)
                _log_metric_if_changed(
                    client, child_id, "quality_gain", delta, step=attempt
                )
            child_terminal = all(
                (sides.get(side) or {}).get("status")
                not in {None, "running", "pending"}
                for side in ("stock", "patch")
            )
            if child_terminal:
                client.log_text(
                    child_id,
                    json.dumps(
                        {"spec": spec, "sides": sides},
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    "dataset_attempt.json",
                )
                client.set_terminated(child_id, status="FINISHED")
            elif child.info.status != "RUNNING":
                client.update_run(child_id, status="RUNNING")
            counts["children"] += 1
    return counts
