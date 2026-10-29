#!/usr/bin/env python3
"""Pull server quality-drain journal and build a thesis-friendly summary.

Usage (after ``tsh login``):
  python research/evolve/scripts/sync_hour_verdicts.py
  python research/evolve/scripts/sync_hour_verdicts.py --local-only
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HOST = "anpilov.kirill2@research-cluster-anpilov-2ekirill2"
REMOTE_WS = "~/fedotllm-controller-runs/2026-09-16-quality-drain"
LOCAL_OUT = REPO / "research/evolve/campaigns/2026-09-16/server_mirror"
LOCAL_QUEUE = REPO / "research/evolve/campaigns/2026-09-15/quality_queue"


def tsh_cert_expired() -> bool:
    try:
        result = subprocess.run(
            ["/usr/local/bin/tsh", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}"
        return "EXPIRED" in output
    except OSError:
        return True


def tsh_rsync(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/local/bin/tsh",
        "--skip-version-check",
        "scp",
        f"{HOST}:{remote_path}",
        str(local_path),
    ]
    subprocess.run(cmd, check=True)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _queue_meta(queue_dir: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not queue_dir.is_dir():
        return meta
    for path in queue_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cid = str(payload.get("candidate_id") or path.stem)
        meta[cid] = {
            "proposal_id": payload.get("proposal_id") or cid,
            "candidate_revision_id": payload.get("candidate_revision_id") or cid,
            "patch_revision": payload.get("patch_revision"),
            "patch_sha256": payload.get("patch_sha256"),
            "title": payload.get("title") or "",
            "issue_url": payload.get("issue_url") or "",
            "source_revision": payload.get("source_revision") or "",
            "registry_identity": payload.get("registry_identity") or "",
            "scope_frozen_at": payload.get("scope_frozen_at"),
            "file_path": payload.get("file_path"),
            "status": payload.get("status"),
            "keep": payload.get("keep"),
            "reason": payload.get("reason"),
            "target_delta": payload.get("target_delta"),
            "regression_deltas": payload.get("regression_deltas"),
            "infrastructure_error": payload.get("infrastructure_error"),
            "quality_failure": payload.get("quality_failure"),
            "hypothesis": payload.get("hypothesis") or {},
            "quality_scope": payload.get("quality_scope") or {},
            "task_ids": payload.get("task_ids") or [],
            "quality_message": payload.get("quality_message") or "",
        }
    return meta


def _dataset_rows(jobs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for job in jobs:
        spec = job.get("spec") or {}
        task_id = str(spec.get("source_dataset") or "")
        stock = job.get("stock_score")
        patched = job.get("patched_score")
        stock_observations = job.get("stock_observations") or {}
        patched_observations = job.get("patched_observations") or {}
        delta = None
        if isinstance(stock, (int, float)) and isinstance(patched, (int, float)):
            higher = bool(spec.get("higher_is_better"))
            delta = (patched - stock) if higher else (stock - patched)
        rows.append(
            {
                "task_id": task_id,
                "metric": spec.get("metric"),
                "higher_is_better": spec.get("higher_is_better"),
                "min_delta": spec.get("min_delta", 0.0),
                "delta_mode": spec.get("delta_mode", "absolute"),
                "stock_status": job.get("stock_status"),
                "patched_status": job.get("patched_status"),
                "stock_score": stock,
                "patched_score": patched,
                "delta": delta,
                "stock_detail": job.get("stock_detail"),
                "patched_detail": job.get("patched_detail"),
                "stock_traceback": job.get("stock_traceback"),
                "patched_traceback": job.get("patched_traceback"),
                "stock_search_ran": stock_observations.get("search_ran"),
                "patched_search_ran": patched_observations.get("search_ran"),
                "stock_search_detail": stock_observations.get("search_detail"),
                "patched_search_detail": patched_observations.get("search_detail"),
                "stock_was_optimised": stock_observations.get("was_optimised"),
                "patched_was_optimised": patched_observations.get("was_optimised"),
                "stock_worker_log_path": stock_observations.get("worker_log_path"),
                "patched_worker_log_path": patched_observations.get("worker_log_path"),
            }
        )
    return rows


def derive_quality_outcome(datasets: list[dict], task_ids: list[str]) -> dict:
    """Re-evaluate complete rows with frozen per-task thresholds, never averaging metrics."""

    ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    if not ids:
        return {"outcome": "not_evaluated", "keep": None, "wins": 0, "losses": 0}
    by_task = {str(row.get("task_id") or ""): row for row in datasets}
    rows = [by_task.get(task_id) for task_id in ids]
    if any(row is None for row in rows):
        return {
            "outcome": "not_evaluated",
            "keep": None,
            "wins": 0,
            "losses": 0,
            "unresolved_task_ids": [
                task_id for task_id, row in zip(ids, rows) if row is None
            ],
        }
    unresolved = []
    wins = 0
    losses = 0
    for task_id, row in zip(ids, rows):
        before = row.get("stock_score")
        after = row.get("patched_score")
        if (
            row.get("stock_status") != "ok"
            or row.get("patched_status") != "ok"
            or not isinstance(before, (int, float))
            or not isinstance(after, (int, float))
            or not math.isfinite(float(before))
            or not math.isfinite(float(after))
        ):
            unresolved.append(task_id)
            continue
        delta = float(row.get("delta") or 0.0)
        tolerance = 1e-12 * max(1.0, abs(float(before)), abs(float(after)))
        if delta < -tolerance:
            losses += 1
        elif delta >= float(row.get("min_delta") or 0.0):
            wins += 1
    if unresolved:
        return {
            "outcome": "not_evaluated",
            "keep": None,
            "wins": wins,
            "losses": losses,
            "unresolved_task_ids": unresolved,
        }
    if losses:
        outcome, keep = "regression", False
    elif wins > len(ids) / 2:
        outcome, keep = "broad_positive", True
    elif wins >= 2:
        outcome, keep = "local_positive", True
    else:
        outcome, keep = "insufficient_effect", False
    return {"outcome": outcome, "keep": keep, "wins": wins, "losses": losses}


def summarize(journal: list[dict], queue_dir: Path) -> dict:
    by_candidate: dict[str, dict] = defaultdict(
        lambda: {
            "events": [],
            "keep": None,
            "status": "unknown",
            "jobs": [],
            "side_events": [],
        }
    )
    stock_tasks: set[str] = set()
    unattributed_journal_rows: list[dict] = []
    for row in journal:
        event = str(row.get("event") or "")
        if event == "fedot_quality_stock":
            for job in row.get("jobs") or []:
                spec = job.get("spec") or {}
                tid = spec.get("source_dataset")
                if tid:
                    stock_tasks.add(str(tid))
            continue
        if event not in {
            "fedot_quality",
            "quality_disposition",
            "quality_candidate_started",
        } and not event.startswith("quality_task_side_"):
            continue
        cid = str(row.get("candidate") or row.get("candidate_id") or "")
        if not cid or cid == "None":
            unattributed_journal_rows.append(row)
            continue
        if event.startswith("quality_task_side_"):
            by_candidate[cid]["side_events"].append(row)
            by_candidate[cid]["events"].append(row)
            by_candidate[cid]["registry_identity"] = row.get(
                "registry_identity"
            ) or by_candidate[cid].get("registry_identity")
            continue
        if event == "quality_candidate_started":
            by_candidate[cid]["events"].append(row)
            by_candidate[cid]["status"] = "running"
            by_candidate[cid]["task_ids"] = row.get("task_ids") or []
            by_candidate[cid]["hypothesis"] = row.get("hypothesis") or {}
            by_candidate[cid]["quality_scope"] = row.get("quality_scope") or {}
            by_candidate[cid]["registry_identity"] = row.get("registry_identity")
            for field in (
                "proposal_id",
                "candidate_revision_id",
                "patch_revision",
                "patch_sha256",
                "title",
                "issue_url",
                "source_revision",
            ):
                if row.get(field) is not None:
                    by_candidate[cid][field] = row[field]
            continue
        if event == "quality_disposition":
            by_candidate[cid]["events"].append(row)
            by_candidate[cid]["status"] = (
                row.get("status") or by_candidate[cid]["status"]
            )
            by_candidate[cid]["reason"] = row.get("reason") or by_candidate[cid].get(
                "reason"
            )
            by_candidate[cid]["quality_outcome"] = row.get("quality_outcome")
            by_candidate[cid]["quality_message"] = row.get("quality_message") or ""
            by_candidate[cid]["keep"] = row.get("status") == "keep"
            continue
        if event != "fedot_quality":
            continue
        decision = row.get("decision") or {}
        by_candidate[cid]["events"].append(row)
        by_candidate[cid]["jobs"] = row.get("jobs") or []
        if "keep" in decision:
            by_candidate[cid]["keep"] = decision.get("keep")
        if decision.get("reason"):
            by_candidate[cid]["reason"] = decision.get("reason")
        if decision.get("quality_outcome"):
            by_candidate[cid]["quality_outcome"] = decision.get("quality_outcome")
        if decision.get("target_delta") is not None:
            by_candidate[cid]["target_delta"] = decision.get("target_delta")
        if decision.get("regression_deltas"):
            by_candidate[cid]["regression_deltas"] = decision.get("regression_deltas")
        if decision.get("infrastructure_error"):
            by_candidate[cid]["infrastructure_error"] = decision.get(
                "infrastructure_error"
            )
        if decision.get("infrastructure_error"):
            by_candidate[cid]["status"] = "infrastructure_error"
        elif decision.get("quality_failure"):
            by_candidate[cid]["status"] = decision["quality_failure"]
        else:
            by_candidate[cid]["status"] = "keep" if decision.get("keep") else "drop"

    queue_meta = _queue_meta(queue_dir)
    for cid, payload in queue_meta.items():
        slot = by_candidate[cid]
        slot.setdefault("file_path", payload.get("file_path"))
        for field in (
            "proposal_id",
            "candidate_revision_id",
            "patch_revision",
            "patch_sha256",
            "title",
            "issue_url",
            "source_revision",
            "registry_identity",
            "scope_frozen_at",
        ):
            if payload.get(field) is not None:
                slot[field] = payload[field]
        if payload.get("status"):
            slot["queue_status"] = payload.get("status")
        if slot.get("keep") is None and payload.get("keep") is not None:
            slot["keep"] = payload.get("keep")
        if not slot.get("reason") and payload.get("reason"):
            slot["reason"] = payload.get("reason")
        if slot.get("target_delta") is None and payload.get("target_delta") is not None:
            slot["target_delta"] = payload.get("target_delta")
        if payload.get("regression_deltas") and not slot.get("regression_deltas"):
            slot["regression_deltas"] = payload.get("regression_deltas")
        if payload.get("infrastructure_error") is not None:
            slot["infrastructure_error"] = payload.get("infrastructure_error")
        if payload.get("quality_failure"):
            slot["quality_failure"] = payload["quality_failure"]
        if payload.get("quality_outcome"):
            slot["quality_outcome"] = payload["quality_outcome"]
        slot["hypothesis"] = payload.get("hypothesis") or {}
        slot["quality_scope"] = payload.get("quality_scope") or {}
        slot["task_ids"] = payload.get("task_ids") or []
        slot["quality_message"] = payload.get("quality_message") or ""

    for cid, slot in by_candidate.items():
        if slot.get("side_events"):
            latest: dict[tuple[str, str], dict] = {}
            attempts: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for event in slot["side_events"]:
                key = (str(event.get("task_id") or ""), str(event.get("side") or ""))
                if not all(key):
                    continue
                state = {
                    "status": "running"
                    if event.get("event", "").endswith("started")
                    else "unknown",
                    "finished": event.get("event", "").endswith(("finished", "reused")),
                    "attempt": int(event.get("attempt") or 1),
                    "worker_log_path": None,
                }
                result = event.get("result") or {}
                state.update(result)
                state["event_ts"] = event.get("ts")
                latest[key] = state
                if state["finished"]:
                    attempts[key].append(state)
            task_ids = list(
                dict.fromkeys(
                    [*(slot.get("task_ids") or []), *(task for task, _side in latest)]
                )
            )
            datasets = []
            active_tasks = set()
            completed_pairs = 0
            measured_pairs = 0
            for task_id in task_ids:
                stock = latest.get((task_id, "stock"), {})
                patched = latest.get((task_id, "patch"), {})
                stock_finished = bool(stock.get("finished"))
                patch_finished = bool(patched.get("finished"))
                if stock_finished and patch_finished:
                    completed_pairs += 1
                if (
                    stock.get("status") == "ok"
                    and patched.get("status") == "ok"
                    and isinstance(stock.get("score"), (int, float))
                    and isinstance(patched.get("score"), (int, float))
                ):
                    measured_pairs += 1
                if (stock.get("status") == "running" and not stock_finished) or (
                    patched.get("status") == "running" and not patch_finished
                ):
                    active_tasks.add(task_id)
                spec = next(
                    (
                        event.get("spec") or {}
                        for event in reversed(slot["side_events"])
                        if event.get("task_id") == task_id
                    ),
                    {},
                )
                before, after = stock.get("score"), patched.get("score")
                delta = None
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    delta = (
                        (after - before)
                        if spec.get("higher_is_better")
                        else (before - after)
                    )
                    if spec.get("delta_mode") == "relative":
                        delta /= max(abs(float(before)), 1e-12)
                datasets.append(
                    {
                        "task_id": task_id,
                        "metric": spec.get("metric"),
                        "higher_is_better": spec.get("higher_is_better"),
                        "min_delta": spec.get("min_delta", 0.0),
                        "delta_mode": spec.get("delta_mode", "absolute"),
                        "stock_status": stock.get("status"),
                        "patched_status": patched.get("status"),
                        "stock_score": before,
                        "patched_score": after,
                        "delta": delta,
                        "stock_worker_log_path": stock.get("worker_log_path"),
                        "patched_worker_log_path": patched.get("worker_log_path"),
                        "stock_attempts": attempts.get((task_id, "stock"), []),
                        "patched_attempts": attempts.get((task_id, "patch"), []),
                        "pair_complete": stock_finished and patch_finished,
                    }
                )
            slot["datasets"] = datasets
            slot["planned_task_count"] = len(slot.get("task_ids") or [])
            slot["completed_task_pair_count"] = completed_pairs
            slot["measured_task_pair_count"] = measured_pairs
            slot["active_task_ids"] = sorted(active_tasks)
        elif slot.get("jobs"):
            slot["datasets"] = _dataset_rows(slot["jobs"])
        if slot.get("datasets"):
            derived = derive_quality_outcome(
                slot["datasets"], slot.get("task_ids") or []
            )
            slot["derived_quality_outcome"] = derived["outcome"]
            slot["derived_keep"] = derived["keep"]
            slot["derived_improved_task_count"] = derived["wins"]
            slot["derived_regressed_task_count"] = derived["losses"]
            slot["derived_unresolved_task_ids"] = derived.get("unresolved_task_ids", [])

    queued = (
        sorted(p.stem for p in queue_dir.glob("*.json")) if queue_dir.is_dir() else []
    )
    keep_yes = [cid for cid, meta in by_candidate.items() if meta.get("keep") is True]
    keep_no = [
        cid
        for cid, meta in by_candidate.items()
        if (meta.get("queue_status") or meta.get("status")) == "drop"
    ]
    infrastructure_error_count = sum(
        1
        for meta in by_candidate.values()
        if (meta.get("queue_status") or meta.get("status")) == "infrastructure_error"
    )
    quality_failure_counts = {
        status: sum(
            1
            for meta in by_candidate.values()
            if (meta.get("queue_status") or meta.get("status")) == status
        )
        for status in ("candidate_error", "evaluation_error")
    }
    pending = [
        cid
        for cid in queued
        if queue_meta.get(cid, {}).get("status") in {None, "queued", "running"}
    ]
    for slot in by_candidate.values():
        slot.pop("events", None)
        slot.pop("side_events", None)
    derived_outcomes = [
        meta.get("derived_quality_outcome")
        for meta in by_candidate.values()
        if meta.get("derived_quality_outcome")
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stock_tasks_warmed": sorted(stock_tasks),
        "queue_count": len(queued),
        "pending_count": len(pending),
        "completed_count": len(queued) - len(pending),
        "candidates_with_journal": len(by_candidate),
        "unattributed_journal_rows": unattributed_journal_rows,
        "keep_true": keep_yes,
        "keep_false_count": len(keep_no),
        "infrastructure_error_count": infrastructure_error_count,
        "quality_failure_counts": quality_failure_counts,
        "derived_quality_outcome_counts": {
            outcome: derived_outcomes.count(outcome)
            for outcome in (
                "broad_positive",
                "local_positive",
                "regression",
                "insufficient_effect",
                "not_evaluated",
            )
        },
        "active_tasks": [
            {"candidate_id": cid, "task_id": task_id}
            for cid, meta in sorted(by_candidate.items())
            for task_id in meta.get("active_task_ids", [])
        ],
        "candidates": dict(sorted(by_candidate.items())),
        "queue_only": {
            cid: queue_meta[cid]
            for cid in queued
            if cid not in by_candidate
            and queue_meta.get(cid, {}).get("status") not in {None, "queued"}
        },
    }


def attach_run_manifest(summary: dict, manifest: dict) -> dict:
    """Attach snapshot freshness and non-authoritative worker-log observations."""

    summary["server_snapshot_utc"] = manifest.get("last_confirmed_server_snapshot_utc")
    observed = summary["server_snapshot_utc"]
    if observed:
        stamp = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
        age_s = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
        summary["snapshot_age_seconds"] = age_s
        summary["snapshot_stale"] = age_s >= 55 * 60
    else:
        summary["snapshot_stale"] = True
    summary["latest_artifact_observation"] = (
        manifest.get("latest_artifact_observation") or {}
    )
    summary["current_stage"] = manifest.get("current_stage") or ""
    return summary


def render_md(summary: dict) -> str:
    def metric_value(value: object) -> str:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.6g}"

    snapshot_label = (
        "unknown"
        if not summary.get("server_snapshot_utc")
        else "stale"
        if summary.get("snapshot_stale")
        else "fresh"
    )
    lines = [
        "# Hour Fedot verdicts (server quality-drain)",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"- Last server snapshot: **{summary.get('server_snapshot_utc') or 'unknown'}** "
        f"({snapshot_label})",
        "",
        f"- Stock tasks seen in journal: **{len(summary['stock_tasks_warmed'])}**",
        f"- Queue size: **{summary['queue_count']}**",
        f"- Completed (non-queued status): **{summary.get('completed_count', 0)}**",
        f"- Pending: **{summary.get('pending_count', 0)}**",
        f"- Candidates with journal rows: **{summary['candidates_with_journal']}**",
        f"- **keep=true:** {len(summary['keep_true'])}",
        f"- **Quality drops:** {summary['keep_false_count']}",
        f"- **Infrastructure errors (not quality drops):** {summary.get('infrastructure_error_count', 0)}",
        f"- **Candidate-side errors:** {summary.get('quality_failure_counts', {}).get('candidate_error', 0)}",
        f"- **Evaluation/baseline errors:** {summary.get('quality_failure_counts', {}).get('evaluation_error', 0)}",
        f"- Outcomes recalculated from completed dataset rows: **{summary.get('derived_quality_outcome_counts', {})}**",
        f"- Active candidate/dataset pairs: **{len(summary.get('active_tasks', []))}**",
        "",
    ]
    observation = summary.get("latest_artifact_observation") or {}
    if observation:
        observed_at = observation.get("observed_at_utc") or "unknown"
        stage = observation.get("stage") or summary.get("current_stage") or "unknown"
        worker_tasks = (
            observation.get("observed_worker_tasks")
            or observation.get("active_dataset_workers")
            or []
        )
        latest_worker_log = observation.get("latest_worker_log_update_utc") or "unknown"
        lines.extend(
            [
                "## Последнее наблюдение по worker-логам",
                f"- Снимок: `{observed_at}`.",
                f"- Последний наблюдавшийся этап: `{stage}`.",
                f"- Задачи из worker-логов: {', '.join(worker_tasks) if worker_tasks else 'нет данных'}.",
                f"- Последнее изменение worker-лога: `{latest_worker_log}`.",
            ]
        )
        if observation.get("process_state") not in {"running", "confirmed_live"}:
            lines.append("- Живость процесса этим снимком не подтверждена.")
        lines.append("")
    if summary["keep_true"]:
        lines.append("## Confirmed improvements (keep=true)")
        for cid in summary["keep_true"]:
            meta = summary["candidates"][cid]
            lines.append(
                f"- `{cid}` — {meta.get('reason', '')} δ={meta.get('target_delta')}"
            )
        lines.append("")
    lines.append("## All scored candidates")
    lines.append(
        "| candidate | patch scope | progress | outcome | keep | queue status | reason |"
    )
    lines.append("|---|---|---:|---|---|---|---|")
    for cid, meta in summary["candidates"].items():
        planned = int(meta.get("planned_task_count") or len(meta.get("task_ids") or []))
        completed = int(meta.get("completed_task_pair_count") or 0)
        progress = f"{completed}/{planned}" if planned else "—"
        title = str(meta.get("title") or f"`{cid}`").replace("|", "\\|")
        lines.append(
            f"| {title} | `{meta.get('file_path', '')}` | {progress} | "
            f"{meta.get('derived_quality_outcome') or meta.get('quality_outcome', '—')} | "
            f"{meta.get('derived_keep', meta.get('keep'))} | "
            f"{meta.get('queue_status', meta.get('status'))} | "
            f"{meta.get('reason', '')} |"
        )
    lines.append("")
    lines.append("## Hypothesis, scope, and decision message")
    for cid, meta in summary["candidates"].items():
        hypothesis = meta.get("hypothesis") or {}
        scope = meta.get("quality_scope") or {}
        task_ids = meta.get("task_ids") or []
        message = str(meta.get("quality_message") or "").strip()
        if not (hypothesis or scope or task_ids or message):
            continue
        lines.append(f"### `{cid}`")
        if meta.get("proposal_id") and meta.get("proposal_id") != cid:
            lines.append(
                f"- Proposal: `{meta['proposal_id']}`; revision: `{meta.get('patch_revision') or 'unknown'}`"
            )
        if meta.get("issue_url"):
            lines.append(f"- Issue: {meta['issue_url']}")
        if meta.get("registry_identity"):
            lines.append(f"- Registry: `{meta['registry_identity']}`")
        if hypothesis.get("claim"):
            lines.append(f"- Hypothesis: {hypothesis['claim']}")
        if hypothesis.get("expected_effect"):
            lines.append(f"- Expected effect: {hypothesis['expected_effect']}")
        if scope.get("problem_families"):
            lines.append(f"- Problem families: {', '.join(scope['problem_families'])}")
        if task_ids:
            lines.append(
                f"- Task IDs: {', '.join(str(task_id) for task_id in task_ids)}"
            )
        if meta.get("derived_quality_outcome"):
            lines.append(
                f"- Recalculated outcome: {meta['derived_quality_outcome']} "
                f"({meta.get('derived_improved_task_count', 0)} gains, "
                f"{meta.get('derived_regressed_task_count', 0)} losses)"
            )
        if message:
            lines.append(f"- Decision: {message}")
        if meta.get("active_task_ids"):
            lines.append(f"- Running datasets: {', '.join(meta['active_task_ids'])}")
        datasets = meta.get("datasets") or []
        if datasets:
            lines.extend(
                [
                    "",
                    "| Task | Metric | Stock | Patched | Stock score | Patched score | Quality gain |",
                    "|---|---|---|---|---:|---:|---:|",
                ]
            )
            for row in datasets:
                task_id = str(row.get("task_id") or "—").replace("|", "\\|")
                metric = str(row.get("metric") or "—").replace("|", "\\|")
                lines.append(
                    f"| `{task_id}` | {metric} | {row.get('stock_status') or '—'} | "
                    f"{row.get('patched_status') or '—'} | "
                    f"{metric_value(row.get('stock_score'))} | "
                    f"{metric_value(row.get('patched_score'))} | "
                    f"{metric_value(row.get('delta'))} |"
                )
        lines.append("")
    if summary.get("queue_only"):
        lines.append("")
        lines.append("## Queue status without journal row yet")
        for cid, meta in summary["queue_only"].items():
            lines.append(
                f"- `{cid}` status={meta.get('status')} keep={meta.get('keep')} "
                f"reason={meta.get('reason', '')} file={meta.get('file_path')}"
            )
    unattributed = summary.get("unattributed_journal_rows") or []
    if unattributed:
        lines.extend(["", "## Journal rows without a candidate ID"])
        lines.append(
            f"These {len(unattributed)} row(s) are not assigned to any candidate; "
            "matching failure text alone is insufficient evidence."
        )
        for index, row in enumerate(unattributed, start=1):
            decision = row.get("decision") or {}
            tasks = [str(item) for item in row.get("task_ids") or []]
            lines.append(
                f"- Row {index}: reason `{decision.get('reason', '')}`; "
                f"tasks: {', '.join(tasks) or 'not recorded'}"
            )
    lines.append("")
    lines.append(
        "_Toy/DEV scores and hunt probe `no_change` are not verdicts. "
        "Only hour Fedot vs cached stock on preregistered tasks._"
    )
    return "\n".join(lines) + "\n"


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _verdict_numbers(
    journal: list[dict], queue_meta: dict[str, dict]
) -> dict[str, int]:
    numbers: dict[str, int] = {}
    verdict_idx = 0
    for row in journal:
        if row.get("event") != "fedot_quality":
            continue
        cid = str(row.get("candidate") or row.get("candidate_id") or "")
        if not cid or cid == "None":
            continue
        if cid not in numbers:
            verdict_idx += 1
            numbers[cid] = verdict_idx
    return numbers


def write_verdict_files(
    summary: dict,
    journal: list[dict],
    queue_meta: dict[str, dict],
    out_dir: Path,
) -> list[Path]:
    numbers = _verdict_numbers(journal, queue_meta)
    written: list[Path] = []
    generated_at = summary.get("generated_at")
    for cid, meta in summary["candidates"].items():
        if meta.get("queue_status") in {None, "queued", "running"} and not meta.get(
            "datasets"
        ):
            continue
        payload = _json_safe(
            {
                "candidate_id": cid,
                "file_path": meta.get("file_path"),
                "queue_status": meta.get("queue_status"),
                "keep": meta.get("keep"),
                "reason": meta.get("reason"),
                "target_delta": meta.get("target_delta"),
                "regression_deltas": meta.get("regression_deltas"),
                "infrastructure_error": meta.get("infrastructure_error"),
                "quality_failure": meta.get("quality_failure"),
                "hypothesis": meta.get("hypothesis") or {},
                "quality_scope": meta.get("quality_scope") or {},
                "task_ids": meta.get("task_ids") or [],
                "quality_message": meta.get("quality_message") or "",
                "derived_quality_outcome": meta.get("derived_quality_outcome"),
                "derived_keep": meta.get("derived_keep"),
                "derived_improved_task_count": meta.get("derived_improved_task_count"),
                "derived_regressed_task_count": meta.get(
                    "derived_regressed_task_count"
                ),
                "derived_unresolved_task_ids": meta.get("derived_unresolved_task_ids")
                or [],
                "datasets": meta.get("datasets") or [],
                "verdict_number": numbers.get(cid),
                "generated_at": generated_at,
            }
        )
        path = out_dir / f"verdict_{cid}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


def write_campaign_rollup(summary: dict, path: Path) -> None:
    scored: list[dict] = []
    by_status: dict[str, int] = defaultdict(int)
    for cid, meta in summary["candidates"].items():
        status = meta.get("queue_status") or meta.get("status")
        if status in {None, "queued", "running"}:
            continue
        by_status[str(status)] += 1
        datasets = meta.get("datasets") or []
        improved = [
            row["task_id"]
            for row in datasets
            if isinstance(row.get("delta"), (int, float))
            and not (isinstance(row["delta"], float) and math.isnan(row["delta"]))
            and row["delta"] > 0
        ]
        regressed = [
            row["task_id"]
            for row in datasets
            if isinstance(row.get("delta"), (int, float))
            and not (isinstance(row["delta"], float) and math.isnan(row["delta"]))
            and row["delta"] < 0
        ]
        scored.append(
            {
                "candidate_id": cid,
                "file_path": meta.get("file_path"),
                "queue_status": status,
                "keep": meta.get("keep"),
                "reason": meta.get("reason"),
                "task_count": len(datasets),
                "improved_tasks": improved,
                "regressed_tasks": regressed,
            }
        )
    payload = {
        "generated_at": summary.get("generated_at"),
        "queue_count": summary.get("queue_count"),
        "scored_count": len(scored),
        "pending_count": summary.get("pending_count"),
        "unattributed_journal_row_count": len(
            summary.get("unattributed_journal_rows") or []
        ),
        "keep_true_count": len(summary.get("keep_true") or []),
        "by_queue_status": dict(sorted(by_status.items())),
        "infra_blockers": sorted(
            {
                row["reason"]
                for row in scored
                if row.get("queue_status") == "infrastructure_error"
                and row.get("reason")
            }
        ),
        "scored": sorted(scored, key=lambda row: row["candidate_id"]),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_detail_jsonl(summary: dict, path: Path) -> None:
    lines: list[str] = []
    for cid, meta in summary["candidates"].items():
        if meta.get("queue_status") in {None, "queued", "running"} and not meta.get(
            "datasets"
        ):
            continue
        lines.append(
            json.dumps(
                {
                    "candidate_id": cid,
                    "file_path": meta.get("file_path"),
                    "queue_status": meta.get("queue_status"),
                    "keep": meta.get("keep"),
                    "reason": meta.get("reason"),
                    "target_delta": meta.get("target_delta"),
                    "regression_deltas": meta.get("regression_deltas"),
                    "infrastructure_error": meta.get("infrastructure_error"),
                    "hypothesis": meta.get("hypothesis") or {},
                    "quality_scope": meta.get("quality_scope") or {},
                    "task_ids": meta.get("task_ids") or [],
                    "quality_message": meta.get("quality_message") or "",
                    "derived_quality_outcome": meta.get("derived_quality_outcome"),
                    "derived_keep": meta.get("derived_keep"),
                    "datasets": meta.get("datasets") or [],
                },
                ensure_ascii=False,
            )
        )
    for row in summary.get("unattributed_journal_rows") or []:
        decision = row.get("decision") or {}
        lines.append(
            json.dumps(
                {
                    "candidate_id": None,
                    "unattributed": True,
                    "reason": decision.get("reason"),
                    "keep": decision.get("keep"),
                    "infrastructure_error": decision.get("infrastructure_error"),
                    "task_ids": row.get("task_ids") or [],
                    "datasets": _dataset_rows(row.get("jobs") or []),
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sync_remote(journal_local: Path, queue_remote: Path) -> None:
    tsh_rsync(f"{REMOTE_WS}/quality_jobs.jsonl", journal_local)
    queue_remote.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            "/usr/local/bin/tsh --skip-version-check ssh",
            f"{HOST}:{REMOTE_WS}/quality_queue/",
            f"{queue_remote}/",
        ],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Rebuild summaries from local server_mirror without tsh scp/rsync",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Read one quality-run directory containing quality_queue/ and quality_jobs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report and local MLflow database directory (defaults to <workspace>/reports)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Also sync the local journal to an SQLite MLflow store",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        help="Tracking URI; defaults to sqlite:///<output-dir>/mlflow.db",
    )
    parser.add_argument("--write-md", type=Path)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument(
        "--write-detail-jsonl",
        type=Path,
    )
    parser.add_argument(
        "--write-rollup",
        type=Path,
    )
    args = parser.parse_args()

    run_workspace = args.workspace.resolve() if args.workspace else None
    output_dir = args.output_dir or (
        run_workspace / "reports" if run_workspace else LOCAL_OUT
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_local = (
        run_workspace / "quality_jobs.jsonl"
        if run_workspace
        else LOCAL_OUT / "quality_jobs.jsonl"
    )
    queue_remote = (
        run_workspace / "quality_queue"
        if run_workspace
        else LOCAL_OUT / "quality_queue_remote"
    )

    if args.local_only or run_workspace is not None:
        print("local-only: skipping remote sync")
    elif tsh_cert_expired():
        print(
            "tsh cert EXPIRED — run `tsh login`, then rerun without --local-only. "
            "Use --local-only to rebuild from the last mirrored journal/queue.",
            file=sys.stderr,
        )
        return 2
    else:
        sync_remote(journal_local, queue_remote)

    queue_dir = (
        queue_remote
        if run_workspace is not None or any(queue_remote.glob("*.json"))
        else LOCAL_QUEUE
    )

    journal = load_jsonl(journal_local)
    queue_meta = _queue_meta(queue_dir)
    summary = summarize(journal, queue_dir)
    if run_workspace is not None:
        try:
            manifest = json.loads(
                (run_workspace / "run_manifest.json").read_text(encoding="utf-8")
            )
            attach_run_manifest(summary, manifest)
        except (OSError, json.JSONDecodeError, ValueError):
            summary["server_snapshot_utc"] = None
            summary["snapshot_stale"] = True

    write_json = args.write_json or output_dir / "hour_verdicts_summary.json"
    write_md = args.write_md or output_dir / "hour_verdicts_summary.md"
    write_detail = args.write_detail_jsonl or output_dir / "hour_verdicts_detail.jsonl"
    write_rollup = args.write_rollup or output_dir / "campaign_rollup.json"
    write_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_md.write_text(render_md(summary), encoding="utf-8")
    write_detail_jsonl(summary, write_detail)
    verdict_paths = write_verdict_files(summary, journal, queue_meta, output_dir)
    write_campaign_rollup(summary, write_rollup)
    print(f"wrote {write_json}")
    print(f"wrote {write_md}")
    print(f"wrote {write_detail}")
    print(f"wrote {write_rollup}")
    for path in verdict_paths:
        print(f"wrote {path}")
    print(f"keep_true={len(summary['keep_true'])} queue={summary['queue_count']}")
    if args.mlflow:
        try:
            from research.evolve.scripts.mlflow_quality_sync import (
                sync_quality_to_mlflow,
            )

            uri = args.mlflow_uri or f"sqlite:///{(output_dir / 'mlflow.db').resolve()}"
            counts = sync_quality_to_mlflow(journal, summary, tracking_uri=uri)
            print(f"synced MLflow: {counts} ({uri})")
        except Exception as exc:
            print(
                f"MLflow sync unavailable; local reports were written: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    try:
        from research.evolve.scripts.github_quality_sync import sync_issue_comments

        issue_result = sync_issue_comments(summary)
        if issue_result.get("reason") != "no_issue_urls":
            print(f"GitHub issue sync: {issue_result}")
    except Exception as exc:
        print(
            f"GitHub issue sync unavailable; local reports were written: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
