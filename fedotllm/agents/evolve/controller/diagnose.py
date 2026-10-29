"""Read-only diagnosis of one saved Evolve quality run."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.storage.quality_events import read_quality_events


_TERMINAL = {
    "artifact_error", "storage_error", "completed", "duration_exhausted",
    "restart_exhausted", "hunt_budget_exhausted", "hunt_campaigns_exhausted",
}
_TIMESTAMPED_LOG_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{1,6})?)"
    r"\s+-\s+(?P<message>.*)$"
)
_GENERATION_MARKER = re.compile(r"Generation num:\s*(\d+)\s+size:\s*(\d+)")
_POPULATION_MARKER = re.compile(
    r"(\d+) individuals out of (\d+) in previous population were evaluated successfully",
    re.IGNORECASE,
)
_TUNING_TRIAL_MARKER = re.compile(
    r"TPE using\s+(\d+)/(\d+)\s+trials with best loss", re.IGNORECASE
)
_FIT_OPERATION_MARKER = re.compile(
    r"((?:Sparse)?LaggedTransformationImplementation).*?Window size.*?changed.*?"
    r"from\s+(\d+)\s+to\s+(\d+)",
    re.IGNORECASE,
)
_STACK_DUMP_MARKER = re.compile(r"^--- stack dump at (.+) ---$", re.MULTILINE)
_FAULTHANDLER_TIMEOUT_MARKER = re.compile(r"^Timeout \((.+)\)!$", re.MULTILINE)
_STACK_SNAPSHOT_MARKER = re.compile(
    r"^(?:Timeout \(.+\)!|--- stack dump at .+ ---)$", re.MULTILINE
)
_STACK_THREAD = re.compile(r"^(?:Current thread|Thread) (\S+) \(most recent call first\):$")
_STACK_FRAME = re.compile(r'^\s+File "(.+)", line (\d+) in (.*)$')


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _age_seconds(value, now):
    parsed = _parse_time(value)
    return None if parsed is None else max(0.0, (now - parsed).total_seconds())


def _local_artifact_path(root: Path, raw_path) -> Path | None:
    """Resolve one named artifact, preserving invocation directories after SCP."""
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_file():
        return path
    for snapshot_copy in (
        root / "worker_artifacts" / path.parent.name / path.name,
        root / path.parent.name / path.name,
        root / path.name,
        root / "worker_artifacts" / path.name,
        root / "logs" / path.name,
    ):
        if snapshot_copy.is_file():
            return snapshot_copy
    return None


def _worker_log_evidence(path: Path | None) -> dict:
    """Read timestamps and known FEDOT markers from the exact saved worker log."""
    if path is None:
        return {}
    latest_line = None
    latest_line_at = None
    latest_marker = None
    latest_marker_at = None
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                match = _TIMESTAMPED_LOG_LINE.match(line)
                if not match:
                    continue
                timestamp = _parse_time(match.group("timestamp").replace(",", "."))
                if timestamp is None:
                    continue
                if latest_line_at is None or timestamp >= latest_line_at:
                    latest_line_at = timestamp
                    latest_line = line

                message = match.group("message")
                generation = _GENERATION_MARKER.search(message)
                population = _POPULATION_MARKER.search(message)
                tuning_trial = _TUNING_TRIAL_MARKER.search(message)
                if tuning_trial and int(tuning_trial.group(1)) == 0:
                    tuning_trial = None
                fit_operation = _FIT_OPERATION_MARKER.search(message)
                if not any((generation, population, tuning_trial, fit_operation)):
                    continue
                if latest_marker_at is not None and timestamp < latest_marker_at:
                    continue
                kind = (
                    "generation" if generation else
                    "population_evaluated" if population else
                    "tuning_trial_completed" if tuning_trial else
                    "fit_operation"
                )
                latest_marker = {
                    "at": timestamp.isoformat(),
                    "line": line,
                    "kind": kind,
                    "generation": int(generation.group(1)) if generation else None,
                    "population_size": int(generation.group(2)) if generation else None,
                    "evaluated_count": int(population.group(1)) if population else None,
                    "evaluated_population_size": int(population.group(2)) if population else None,
                    "tuning_trials_completed": int(tuning_trial.group(1)) if tuning_trial else None,
                    "tuning_trials_total": int(tuning_trial.group(2)) if tuning_trial else None,
                    "operation": fit_operation.group(1) if fit_operation else None,
                    "window_size_before": int(fit_operation.group(2)) if fit_operation else None,
                    "window_size_after": int(fit_operation.group(3)) if fit_operation else None,
                }
                latest_marker_at = timestamp
    except OSError as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}

    return {
        "last_timestamped_line_at": latest_line_at.isoformat() if latest_line_at else None,
        "last_timestamped_line": latest_line,
        "last_optimizer_marker": latest_marker,
    }


def _stack_locations(text: str) -> dict:
    """Extract saved Python locations; repeated samples alone do not prove a hang."""
    snapshots = []
    has_markers = bool(_STACK_SNAPSHOT_MARKER.search(text))
    segments = _STACK_SNAPSHOT_MARKER.split(text)[1:] if has_markers else [text]
    for segment in segments:
        threads = []
        current = None
        for line in segment.splitlines():
            thread = _STACK_THREAD.match(line)
            if thread:
                current = {"thread": thread.group(1), "frames": []}
                threads.append(current)
                continue
            frame = _STACK_FRAME.match(line)
            if frame and current is not None:
                current["frames"].append({
                    "file": frame.group(1), "line": int(frame.group(2)),
                    "function": frame.group(3),
                })
        snapshots.append(threads)
    latest = snapshots[-1] if snapshots else []
    repeated = 0
    if not has_markers:
        # Signal dumps have no reliable per-dump delimiter. Keep only each
        # thread's latest observed frames, without inventing snapshot counts.
        latest = list({thread["thread"]: thread for thread in latest}.values())
    if has_markers and latest and any(thread["frames"] for thread in latest):
        for snapshot in reversed(snapshots):
            if snapshot != latest:
                break
            repeated += 1
    return {
        "stack_threads": latest,
        "repeated_stack_snapshots": repeated,
        "stack_repetition_proves_hang": False,
        "stack_frame_scope": "python_frames_in_this_process",
        # Legacy faulthandler has no end marker; a live copy can end mid-dump.
        "stack_snapshot_completeness": (
            "complete" if segments and segments[-1].rstrip().endswith("--- stack dump complete ---")
            else "unknown"
        ),
        "stack_snapshot_boundary": "marker" if has_markers else "unmarked_stream",
    }


def _stack_trace_evidence(path: Path | None) -> dict:
    """Summarize a captured worker stack artifact without scanning other files."""
    if path is None:
        return {"stack_trace_artifact_state": "not_copied"}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        markers = _STACK_DUMP_MARKER.findall(text)
        faulthandler_markers = _FAULTHANDLER_TIMEOUT_MARKER.findall(text)
        return {
            "stack_trace_artifact_state": "present",
            "stack_trace_size_bytes": path.stat().st_size,
            "stack_dump_count": len(markers) + len(faulthandler_markers),
            "last_stack_dump_at": markers[-1] if markers else None,
            "last_faulthandler_timeout": (
                faulthandler_markers[-1] if faulthandler_markers else None
            ),
            "stack_trace_tail": text[-12_000:],
            **_stack_locations(text),
        }
    except OSError as exc:
        return {
            "stack_trace_artifact_state": "unreadable",
            "stack_trace_error": f"{type(exc).__name__}: {exc}",
        }


def _child_trace_evidence(root: Path, reference, attempt_id) -> dict:
    """Read only children named by this attempt's explicit diagnostic manifest."""
    report = {"manifest_state": "not_copied", "coverage": "observed_python_children_only",
              "processes": [], "invalid_manifest_rows": 0}
    if not reference or not attempt_id:
        return report
    remote = Path(str(reference))
    candidates = [remote, root / remote.parent.name / remote.name,
                  root / "worker_artifacts" / remote.parent.name / remote.name,
                  root / "worker_artifacts" / remote.name]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        return report
    report["manifest_path"] = str(manifest)
    try:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("manifest entry must be an object")
                except (ValueError, json.JSONDecodeError):
                    report["invalid_manifest_rows"] += 1
                    continue
                if row.get("attempt_id") != attempt_id:
                    continue
                raw_path = row.get("stack_trace_path")
                # Flattened SCP child files must stay inside the copied attempt
                # directory. Never follow arbitrary absolute paths in a manifest.
                local = manifest.parent / Path(str(raw_path)).name if raw_path else None
                if local is not None and (local.is_symlink() or not local.is_file()):
                    local = None
                report["processes"].append({
                    **{key: row.get(key) for key in (
                        "pid", "ppid", "owner_pid", "attempt_id", "started_at",
                        "capture_method", "coverage", "reason", "stack_trace_path",
                    )},
                    **_stack_trace_evidence(local),
                })
        report["manifest_state"] = "present"
    except (OSError, UnicodeError) as exc:
        report.update(manifest_state="unreadable", error=f"{type(exc).__name__}: {exc}")
    return report


def _sidecar_marker_kind(line) -> str | None:
    line = str(line or "")
    if _GENERATION_MARKER.search(line):
        return "generation"
    if _POPULATION_MARKER.search(line):
        return "population_evaluated"
    if _FIT_OPERATION_MARKER.search(line):
        return "fit_operation"
    return None


def _event_stage_reference(event):
    worker_paths = event.get("worker_paths")
    worker_paths = worker_paths if isinstance(worker_paths, dict) else {}
    result = event.get("result")
    observations = result.get("metric_observations") if isinstance(result, dict) else {}
    observations = observations if isinstance(observations, dict) else {}
    return (worker_paths.get("stage_status_path") or event.get("stage_status_path")
            or observations.get("stage_status_path"))


def _stage_match(root, path, payload, event):
    reference = _event_stage_reference(event)
    if reference and _local_artifact_path(root, reference) != path:
        return False, "stage_status_path"
    worker_paths = event.get("worker_paths")
    worker_paths = worker_paths if isinstance(worker_paths, dict) else {}
    invocation = event.get("worker_invocation_id") or worker_paths.get("worker_invocation_id")
    saved_invocation = payload.get("worker_invocation_id")
    # A logical attempt can be rerun with a different nonce or evaluation.
    # Never let attempt_id alone attach those artifacts to the current worker.
    if invocation is not None or saved_invocation is not None:
        if not invocation or saved_invocation != invocation:
            return False, "worker_invocation_id"
    for key in ("registry_identity", "evaluation_identity"):
        if event.get(key) is not None and payload.get(key) != event[key]:
            return False, key
    attempt_id = str(event.get("attempt_id") or "")
    saved_attempt_id = str(payload.get("attempt_id") or "")
    if attempt_id and saved_attempt_id:
        return saved_attempt_id == attempt_id, "attempt_id"
    if payload.get("proposal_id") is not None:
        return (
            str(payload.get("proposal_id") or "") == str(
                event.get("candidate_revision_id") or event.get("candidate") or ""
            )
            and str(payload.get("task_id") or payload.get("source_dataset") or "") == str(event.get("task_id") or "")
            and str(payload.get("side") or "") == str(event.get("side") or "")
            and int(payload.get("attempt") or 1) == int(event.get("attempt") or 1)
        ), "worker_identity"
    # Legacy stage records had no proposal/attempt metadata. Match only the
    # exact dataset+side filename and a stage start within one minute of the
    # durable quality-side started event; label this as an inference.
    task_id, side = str(event.get("task_id") or ""), str(event.get("side") or "")
    if not path.name.startswith(f"quality-out-{task_id}-{side}-"):
        return False, "legacy_filename_and_start_time"
    active_stage = payload.get("active_stage") or payload.get("last_stage")
    stage = (payload.get("stage_timings") or {}).get(active_stage, {})
    stage_started = _parse_time(payload.get("stage_started_at") or stage.get("started_at"))
    event_started = _parse_time(event.get("ts"))
    if stage_started is None or event_started is None:
        return False, "legacy_filename_and_start_time"
    return abs((stage_started - event_started).total_seconds()) <= 60, "legacy_filename_and_start_time"


def _stage_artifact_paths(root: Path, quality_events: list[dict]) -> list[Path]:
    """Read named sidecars and only named invocation folders, plus legacy flats."""
    paths = list((root / "worker_artifacts").glob("quality-out-*.stages.json"))
    paths.extend(root.glob("quality-out-*.stages.json"))
    for event in quality_events:
        worker_paths = event.get("worker_paths")
        worker_paths = worker_paths if isinstance(worker_paths, dict) else {}
        reference = _event_stage_reference(event)
        if reference:
            local = _local_artifact_path(root, reference)
            if local is not None:
                paths.append(local)
        else:
            invocation = event.get("worker_invocation_id") or worker_paths.get("worker_invocation_id")
            # Do not turn malformed journal data into a path traversal or a
            # recursive scan of unrelated workers.
            if isinstance(invocation, str) and re.fullmatch(r"[0-9a-f]{32}", invocation):
                for folder in (root / "worker_artifacts" / invocation, root / invocation):
                    paths.extend(folder.glob("quality-out-*.stages.json"))
    return sorted(set(paths))


def _side_identity(event):
    return (
        str(event.get("candidate") or ""),
        str(event.get("task_id") or ""),
        str(event.get("side") or ""),
    )


def _terminal_dataset_checks(root: Path, quality_events: list[dict], stages: list[tuple[Path, dict]]):
    """List measured and missing stock/patch sides without inferring a verdict."""
    queue_dir = root / "quality_queue"
    queue_rows = []
    if queue_dir.is_dir():
        for path in sorted(queue_dir.glob("*.json")):
            if path.name.startswith("._"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            candidate_id = str(payload.get("candidate_id") or path.stem)
            task_ids = payload.get("required_task_ids") or payload.get("task_ids") or ()
            queue_rows.append({
                "candidate": candidate_id,
                "proposal_id": str(payload.get("proposal_id") or candidate_id),
                "patch_revision": payload.get("patch_revision"),
                "title": payload.get("title"),
                "task_ids": tuple(dict.fromkeys(str(item) for item in task_ids if str(item))),
                "scope_source": "quality_queue",
            })

    candidates = {row["candidate"] for row in queue_rows}
    # A flat SCP snapshot may intentionally omit the queue directory. Preserve
    # what the durable side events prove and report the narrower evidence scope.
    observed_candidates = {}
    for event in quality_events:
        candidate = str(event.get("candidate") or "")
        if not candidate:
            continue
        observed_candidates.setdefault(candidate, set()).add(str(event.get("task_id") or ""))
    for candidate, task_ids in sorted(observed_candidates.items()):
        if candidate not in candidates:
            queue_rows.append({
                "candidate": candidate,
                "proposal_id": candidate,
                "patch_revision": None,
                "title": None,
                "task_ids": tuple(sorted(item for item in task_ids if item)),
                "scope_source": "observed_events_only",
            })

    latest_events = {}
    for event in quality_events:
        if event.get("event") not in {
            "quality_task_side_finished", "quality_task_side_reused",
            "quality_task_side_interrupted",
        }:
            continue
        latest_events[_side_identity(event)] = event

    checks = []
    for proposal in queue_rows:
        for task_id in proposal["task_ids"]:
            for side in ("stock", "patch"):
                event = latest_events.get((proposal["candidate"], task_id, side))
                result = event.get("result") if event and isinstance(event.get("result"), dict) else {}
                observations = result.get("metric_observations")
                observations = observations if isinstance(observations, dict) else {}
                raw_score = result.get("score")
                score = None
                if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                    if math.isfinite(float(raw_score)):
                        score = float(raw_score)
                status = str(result.get("status") or "no_terminal_side_event")
                metric_valid = observations.get("metric_valid") is True
                measured = status == "ok" and score is not None and metric_valid
                metric_state = (
                    "measured" if measured
                    else "unverified" if status == "ok" and score is not None
                    else "missing"
                )
                stage_matches = []
                if event:
                    stage_matches = [
                        (path, payload, method)
                        for path, payload in stages
                        for matched, method in [_stage_match(root, path, payload, event)]
                        if matched
                    ]
                stage_path, stage_payload, _match_method = (
                    max(stage_matches, key=lambda item: str(item[1].get("updated_at") or ""))
                    if stage_matches else (None, {}, None)
                )
                worker_paths = event.get("worker_paths") if event and isinstance(
                    event.get("worker_paths"), dict
                ) else {}
                worker_log_ref = (
                    observations.get("worker_log_path")
                    or stage_payload.get("worker_log_path")
                    or (worker_paths.get("worker_log_path") if worker_paths else None)
                    or (event.get("worker_log_path") if event else None)
                )
                stage_ref = (
                    observations.get("stage_status_path")
                    or (str(stage_path) if stage_path else None)
                    or (worker_paths.get("stage_status_path") if worker_paths else None)
                    or (event.get("stage_status_path") if event else None)
                )
                local_log = _local_artifact_path(root, worker_log_ref)
                local_stage = _local_artifact_path(root, stage_ref)
                checks.append({
                    "candidate_revision_id": proposal["candidate"],
                    "proposal_id": proposal["proposal_id"],
                    "patch_revision": proposal["patch_revision"],
                    "proposal_title": proposal["title"],
                    "task_id": task_id,
                    "side": side,
                    "scope_source": proposal["scope_source"],
                    "metric_state": metric_state,
                    "metric_valid": metric_valid,
                    "result_status": status,
                    "score": score,
                    "duration_s": result.get("duration_s"),
                    "failure_kind": observations.get("failure_kind") or result.get("failure_kind"),
                    "timeout_kind": observations.get("timeout_kind"),
                    "wait_kind": observations.get("wait_kind") or (
                        event.get("wait_kind") if event else None
                    ),
                    "worker_spawned": (
                        observations.get("worker_spawned")
                        if "worker_spawned" in observations
                        else event.get("worker_spawned") if event else None
                    ),
                    "side_elapsed_seconds": observations.get("side_elapsed_seconds"),
                    "timeout_limit_seconds": observations.get("timeout_limit_seconds"),
                    "stage": (
                        observations.get("failed_stage")
                        or observations.get("last_stage")
                        or observations.get("active_stage")
                        or stage_payload.get("active_stage")
                        or stage_payload.get("last_stage")
                    ),
                    "detail": result.get("detail"),
                    "attempt": event.get("attempt") if event else None,
                    "attempt_id": event.get("attempt_id") if event else None,
                    "event": event.get("event") if event else None,
                    "event_at": event.get("ts") if event else None,
                    "worker_log": str(local_log or worker_log_ref) if worker_log_ref else None,
                    "stage_file": str(local_stage or stage_ref) if stage_ref else None,
                    "verdict": None,
                })
    return checks, bool(queue_rows and all(row["scope_source"] == "quality_queue" for row in queue_rows))


def diagnose_run(
    workspace: Path | str,
    *,
    now: datetime | None = None,
    stale_supervisor_seconds: float = 900,
    stale_worker_seconds: float = 3600,
) -> dict:
    """Summarize only this run's saved service, side and worker artifacts."""
    root = Path(workspace).expanduser().resolve()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    status_path = root / "logs" / "service_status.json"
    if not status_path.is_file():
        # Timestamped SCP snapshots place exact requested files at their root.
        status_path = root / "service_status.json"
    status_error = None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status_error = f"unreadable service status: {type(exc).__name__}: {exc}"
        status = {}
    if not isinstance(status, dict):
        status_error = "service status is not a JSON object"
        status = {}
    terminal = (
        status.get("state") in _TERMINAL
        and status.get("child_running") is False
        and status.get("active_child") is None
    )

    events_path = root / "quality_jobs.jsonl"
    try:
        quality_events = read_quality_events(events_path, ignore_incomplete_tail=True)
        journal_error = None
    except (OSError, ValueError) as exc:
        quality_events = []
        journal_error = f"{type(exc).__name__}: {exc}"

    latest_side_events = {}
    for event in quality_events:
        if event.get("event") in {
            "quality_task_side_started", "quality_task_worker_spawned",
            "quality_task_side_waiting_stock_cache",
            "quality_task_side_finished", "quality_task_side_reused",
            "quality_task_side_interrupted",
        }:
            identity = _side_identity(event)
            previous = latest_side_events.get(identity, {})
            # Older controllers periodically labelled the actual stock owner
            # as a waiter after spawning it. Preserve its launch evidence and
            # inspect its sidecar/stack; spawn alone does not prove activity.
            if (event.get("event") == "quality_task_side_waiting_stock_cache"
                    and previous.get("event") == "quality_task_worker_spawned"
                    and (event.get("attempt") or 1) == (previous.get("attempt") or 1)):
                continue
            latest_side_events[identity] = event
    active = [event for event in latest_side_events.values()
              if event.get("event") in {
                  "quality_task_side_started", "quality_task_worker_spawned",
                  "quality_task_side_waiting_stock_cache",
              }]
    stage_paths = _stage_artifact_paths(root, quality_events)
    stages = []
    for path in stage_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            stages.append((path, payload))

    worker_reports = []
    for event in active:
        if event.get("event") == "quality_task_side_waiting_stock_cache":
            try:
                wait_elapsed = float(event.get("wait_elapsed_seconds") or 0.0)
            except (TypeError, ValueError):
                wait_elapsed = None
            wait_event_age = _age_seconds(event.get("ts"), now)
            wait_elapsed_at_snapshot = (
                wait_elapsed + wait_event_age
                if wait_elapsed is not None and wait_event_age is not None
                else wait_elapsed
                if wait_elapsed is not None and wait_elapsed > stale_worker_seconds
                else None
            )
            wait_is_stale = (
                wait_elapsed_at_snapshot is not None
                and wait_elapsed_at_snapshot > stale_worker_seconds
            )
            waiting_for_scheduler = event.get("wait_kind") == "scheduler_queue"
            wait_state = (
                "incomplete_after_service_exit"
                if terminal
                else (
                    "stale_scheduler_queue" if waiting_for_scheduler
                    else "stale_shared_stock_wait"
                ) if wait_is_stale
                else "waiting_in_scheduler_queue" if waiting_for_scheduler
                else "waiting_on_shared_stock"
            )
            wait_evaluation_state = (
                "interrupted_wait_after_service_exit"
                if terminal
                else (
                    "stale_scheduler_queue" if waiting_for_scheduler
                    else "stale_shared_stock_wait"
                ) if wait_is_stale
                else "waiting_for_scheduler_slot" if waiting_for_scheduler
                else "waiting_on_stock_cache"
            )
            worker_reports.append({
                "state": wait_state,
                "progress_state": (
                    "last_observed_wait_before_service_exit"
                    if terminal else "waiting_for_scheduler_slot"
                    if waiting_for_scheduler else "waiting_on_stock_cache"
                ),
                "evaluation_state": wait_evaluation_state,
                "service_terminal": terminal,
                "wait_was_stale_at_snapshot": wait_is_stale,
                "progress_evidence_source": "quality_journal_wait_event",
                "proposal_id": event.get("proposal_id") or event.get("candidate"),
                "candidate_revision_id": event.get("candidate_revision_id") or event.get("candidate"),
                "patch_revision": event.get("patch_revision"),
                "task_id": event.get("task_id"),
                "side": event.get("side"),
                "attempt": event.get("attempt") or 1,
                "worker_launch_state": "not_spawned_waiting",
                "wait_kind": event.get("wait_kind"),
                "pair_slot_held_during_wait": event.get("pair_slot_held_during_wait"),
                "campaign_pair_cap": event.get("campaign_pair_cap"),
                "scheduler_active_pair_slots": event.get("scheduler_active_pair_slots"),
                "scheduler_active_pair_callbacks": event.get("scheduler_active_pair_slots"),
                "scheduler_queued_pair_tasks": event.get("scheduler_queued_pair_tasks"),
                "scheduler_active_compute_workers": event.get("scheduler_active_compute_workers"),
                "scheduler_peak_compute_workers": event.get("scheduler_peak_compute_workers"),
                "wait_elapsed_seconds": wait_elapsed,
                "wait_event_age_seconds": wait_event_age,
                "wait_elapsed_seconds_at_snapshot": wait_elapsed_at_snapshot,
                "quality_started_at": event.get("ts"),
                "worker_artifacts": event.get("worker_paths"),
                "artifact_state": event.get("artifact_state"),
            })
            continue
        matches = [(path, payload, match_kind) for path, payload in stages
                   for matched, match_kind in [_stage_match(root, path, payload, event)] if matched]
        selected = max(matches, key=lambda item: str(item[1].get("updated_at") or "")) if matches else (None, {}, None)
        path, stage, match_method = selected
        heartbeat = stage.get("updated_at")
        worker_age = _age_seconds(heartbeat, now)
        expected_paths = event.get("worker_paths") if isinstance(event.get("worker_paths"), dict) else {}
        worker_log_reference = stage.get("worker_log_path") or (
            str(path.with_name(path.name.replace("quality-out-", "quality-log-", 1)
                               .replace(".stages.json", ".log"))) if path else None
        ) or event.get("worker_log_path") or expected_paths.get("worker_log_path")
        worker_log_path = _local_artifact_path(root, worker_log_reference)
        worker_log = str(worker_log_path or worker_log_reference) if worker_log_reference else None
        log_evidence = _worker_log_evidence(worker_log_path)
        stack_trace_reference = stage.get("stack_trace_path") or expected_paths.get(
            "stack_trace_path"
        )
        stack_trace_path = _local_artifact_path(root, stack_trace_reference)
        stack_trace_evidence = _stack_trace_evidence(stack_trace_path)
        child_trace_evidence = _child_trace_evidence(
            root, stage.get("process_trace_manifest_path")
            or expected_paths.get("process_trace_manifest_path"), event.get("attempt_id"),
        )
        raw_marker = log_evidence.get("last_optimizer_marker") or {}
        sidecar_fedot_log_at = stage.get("last_fedot_log_at")
        sidecar_optimizer_event_at = stage.get("last_optimizer_event_at")
        sidecar_fedot_log_age = _age_seconds(sidecar_fedot_log_at, now)
        sidecar_optimizer_event_age = _age_seconds(sidecar_optimizer_event_at, now)
        raw_marker_at = raw_marker.get("at")
        raw_marker_age = _age_seconds(raw_marker_at, now)
        # Prefer the newest timestamped optimizer marker, comparing raw worker
        # output with the sidecar's explicit optimizer-event claim. A generic
        # timestamped warning is exposed separately and never counts as progress.
        selected_marker = raw_marker
        progress_evidence_source = "raw_worker_log_marker" if raw_marker else None
        selected_marker_at = raw_marker_at
        sidecar_marker_line = stage.get("last_fedot_log_line")
        sidecar_marker_kind = _sidecar_marker_kind(sidecar_marker_line)
        if sidecar_optimizer_event_at and (
            not selected_marker_at
            or (_parse_time(sidecar_optimizer_event_at) or datetime.min.replace(tzinfo=timezone.utc))
            > (_parse_time(selected_marker_at) or datetime.min.replace(tzinfo=timezone.utc))
        ):
            selected_marker = {
                "at": sidecar_optimizer_event_at,
                "line": sidecar_marker_line,
                "kind": sidecar_marker_kind or "sidecar_optimizer_event",
            }
            selected_marker_at = sidecar_optimizer_event_at
            progress_evidence_source = "sidecar_optimizer_event_claim"
        selected_marker_age = _age_seconds(selected_marker_at, now)
        # Legacy last_fedot_log_at sometimes represented only the log mtime.
        # Keep it as a conservative stale signal, but never treat a fresh legacy
        # timestamp as proof of FEDOT optimizer progress.
        fallback_log_age = sidecar_fedot_log_age
        worker_log_exists = worker_log_path is not None
        if not matches:
            worker_state = "unknown"
        elif worker_age is not None and worker_age <= stale_worker_seconds:
            worker_state = "heartbeat_fresh"
        elif worker_age is not None:
            worker_state = "stale_worker"
        else:
            worker_state = "unknown"
        if not worker_log_exists:
            progress_state = "stage_heartbeat_only" if worker_state == "heartbeat_fresh" else "not_observed"
            progress_evidence_source = "worker_log_missing"
        elif selected_marker_at:
            if selected_marker_age is not None and selected_marker_age <= stale_worker_seconds:
                progress_state = "recent_fedot_output"
            else:
                progress_state = "log_quiet"
        elif fallback_log_age is not None and fallback_log_age > stale_worker_seconds:
            progress_state = "log_quiet"
            progress_evidence_source = "legacy_sidecar_log_timestamp"
        else:
            # Legacy sidecars used the worker-log mtime as last_fedot_log_at.
            # A fresh value from that schema cannot distinguish FEDOT output
            # from other writes to the shared log file.
            progress_state = "stage_heartbeat_only" if worker_state == "heartbeat_fresh" else "not_observed"
            progress_evidence_source = "legacy_unverified_log_timestamp"
        fit_stage = (stage.get("active_stage") or stage.get("last_stage")) == "fit"
        fit_elapsed = stage.get("stage_elapsed_seconds")
        try:
            fit_elapsed = float(fit_elapsed)
        except (TypeError, ValueError):
            fit_elapsed = None
        evaluation_state = (
            "stale_evaluation"
            if fit_stage and progress_state == "log_quiet"
            and (selected_marker_age if selected_marker_at else fallback_log_age) is not None
            and (selected_marker_age if selected_marker_at else fallback_log_age) > stale_worker_seconds
            else "fit_without_optimizer_evidence"
            if fit_stage and not selected_marker_at and fit_elapsed is not None
            and fit_elapsed > stale_worker_seconds
            else "fit_activity_observed"
            if fit_stage and progress_state == "recent_fedot_output"
            and selected_marker.get("kind") == "fit_operation"
            else "progressing"
            if progress_state == "recent_fedot_output"
            else "unconfirmed"
        )
        worker_reports.append({
            "state": worker_state,
            "progress_state": progress_state,
            "evaluation_state": evaluation_state,
            "progress_evidence_source": progress_evidence_source,
            "proposal_id": event.get("proposal_id") or event.get("candidate"),
            "candidate_revision_id": event.get("candidate_revision_id") or event.get("candidate"),
            "patch_revision": event.get("patch_revision"),
            "task_id": event.get("task_id"),
            "side": event.get("side"),
            "attempt": event.get("attempt") or 1,
            "worker_launch_state": (
                "spawned" if event.get("event") == "quality_task_worker_spawned"
                else "expected"
            ),
            "stage": stage.get("active_stage") or stage.get("last_stage"),
            "fedot_phase": stage.get("fedot_phase"),
            "fedot_phase_at": stage.get("fedot_phase_at"),
            "fedot_phase_line": stage.get("fedot_phase_line"),
            "match_method": match_method,
            "last_stage_event": stage.get("event"),
            "worker_heartbeat_at": heartbeat,
            "worker_heartbeat_age_seconds": worker_age,
            "fedot_log_age_seconds": selected_marker_age if selected_marker_at else fallback_log_age,
            "raw_fedot_log_age_seconds": raw_marker_age,
            "sidecar_fedot_log_age_seconds": sidecar_fedot_log_age,
            "sidecar_optimizer_event_age_seconds": sidecar_optimizer_event_age,
            "raw_last_timestamped_log_at": log_evidence.get("last_timestamped_line_at"),
            "raw_last_timestamped_log_line": log_evidence.get("last_timestamped_line"),
            "raw_log_read_error": log_evidence.get("read_error"),
            "raw_optimizer_event_at": raw_marker_at,
            "raw_optimizer_event_kind": raw_marker.get("kind"),
            "raw_optimizer_generation": raw_marker.get("generation"),
            "raw_optimizer_population_size": raw_marker.get("population_size"),
            "raw_evaluated_count": raw_marker.get("evaluated_count"),
            "raw_evaluated_population_size": raw_marker.get("evaluated_population_size"),
            "raw_tuning_trials_completed": raw_marker.get("tuning_trials_completed"),
            "raw_tuning_trials_total": raw_marker.get("tuning_trials_total"),
            "raw_fit_operation": raw_marker.get("operation"),
            "raw_fit_window_size_before": raw_marker.get("window_size_before"),
            "raw_fit_window_size_after": raw_marker.get("window_size_after"),
            "selected_optimizer_event_at": selected_marker_at,
            "selected_optimizer_event_kind": selected_marker.get("kind"),
            "sidecar_last_fedot_log_at": sidecar_fedot_log_at,
            "sidecar_last_fedot_log_line": sidecar_marker_line,
            "last_fedot_log_line": selected_marker.get("line") or sidecar_marker_line,
            "raw_last_fedot_log_line": raw_marker.get("line"),
            "stage_elapsed_seconds": stage.get("stage_elapsed_seconds"),
            "cpu_accounting": stage.get("cpu_accounting"),
            "campaign_pair_cap": event.get("campaign_pair_cap"),
            "scheduler_active_pair_slots": event.get("scheduler_active_pair_slots"),
            "scheduler_active_pair_callbacks": event.get("scheduler_active_pair_slots"),
            "scheduler_queued_pair_tasks": event.get("scheduler_queued_pair_tasks"),
            "scheduler_active_compute_workers": event.get("scheduler_active_compute_workers"),
            "scheduler_peak_compute_workers": event.get("scheduler_peak_compute_workers"),
            "last_fedot_log_at": selected_marker_at or sidecar_fedot_log_at,
            "quality_started_at": event.get("ts"),
            "worker_log": worker_log,
            "worker_log_remote_path": worker_log_reference,
            "stack_trace_path": stack_trace_reference,
            **stack_trace_evidence,
            "child_trace_evidence": child_trace_evidence,
            "worker_artifacts": expected_paths,
            "artifact_state": event.get("artifact_state"),
            "stage_file": str(path) if path else str(
                _local_artifact_path(
                    root, event.get("stage_status_path") or expected_paths.get("stage_status_path")
                ) or event.get("stage_status_path") or expected_paths.get("stage_status_path")
            ),
        })

    heartbeat_age = _age_seconds(status.get("heartbeat_at"), now)
    supervisor_is_running = status.get("state") in {"running", "starting"}
    supervisor_state = (
        "unknown" if status_error else
        "stale_supervisor" if supervisor_is_running and heartbeat_age is not None
        and heartbeat_age > stale_supervisor_seconds else
        "heartbeat_fresh" if supervisor_is_running and heartbeat_age is not None else
        "unknown" if supervisor_is_running else "not_running"
    )
    dataset_checks, dataset_scope_complete = _terminal_dataset_checks(root, quality_events, stages)
    if journal_error or status_error:
        state = "unknown"
    elif terminal:
        state = "terminal"
    elif supervisor_state == "stale_supervisor":
        state = "stale_supervisor"
    elif any(row["state"] == "stale_worker" for row in worker_reports):
        state = "stale_worker"
    elif any(row["evaluation_state"] == "stale_evaluation" for row in worker_reports):
        state = "stale_evaluation"
    elif any(row["evaluation_state"] == "stale_scheduler_queue" for row in worker_reports):
        state = "stale_scheduler_queue"
    elif any(row["evaluation_state"] == "stale_shared_stock_wait" for row in worker_reports):
        state = "stale_shared_stock_wait"
    elif any(row["evaluation_state"] == "fit_without_optimizer_evidence" for row in worker_reports):
        state = "stale_evaluation_unconfirmed"
    elif any(row["evaluation_state"] == "waiting_on_stock_cache" for row in worker_reports):
        state = "waiting_on_shared_stock"
    elif any(row["evaluation_state"] == "waiting_for_scheduler_slot" for row in worker_reports):
        state = "waiting_in_scheduler_queue"
    elif supervisor_state == "heartbeat_fresh" and active and any(
        row["evaluation_state"] == "fit_activity_observed" for row in worker_reports
    ):
        state = "fit_activity_observed"
    elif supervisor_state == "heartbeat_fresh" and active and all(
        row["evaluation_state"] == "progressing" for row in worker_reports
    ):
        state = "progressing"
    elif supervisor_state == "heartbeat_fresh" and active:
        state = "progress_unconfirmed"
    else:
        state = "unknown"
    return {
        "state": state,
        "workspace": str(root),
        "service_state": status.get("state"),
        "service_heartbeat_at": status.get("heartbeat_at"),
        "service_heartbeat_age_seconds": heartbeat_age,
        "supervisor_state": supervisor_state,
        "active_child": status.get("active_child"),
        "pending_proposals": status.get("pending_proposals", []),
        "queue_counts": status.get("queue_counts", {}),
        "worker_checks": worker_reports,
        "controller_stack_evidence": _stack_trace_evidence(_local_artifact_path(
            root, root / "logs" / "controller.stacks.txt",
        )),
        "dataset_checks": dataset_checks,
        "dataset_scope_complete": dataset_scope_complete,
        "journal_error": journal_error,
        "service_status_error": status_error,
        "reason": status_error or journal_error,
        "evidence": [str(status_path), str(events_path),
                     *[str(row[0]) for row in stages]],
        "interpretation": (
            "Heartbeat показывает только активность процесса. cpu_accounting показывает CPU-время worker и только уже учтённых завершённых дочерних процессов; это не загрузка сервера, а живые дочерние workers могут ещё отсутствовать в счётчике. Низкое значение не различает I/O, sleep, блокировку и неучтённую работу дочернего процесса. Возраст и последняя строка raw worker log считаются по timestamped строкам самого сохранённого лога; sidecar claim показан отдельно. Маркер Generation/individuals подтверждает событие optimizer, но не конечную метрику; маркер WindowSizeSelector означает лишь активность fit-пути и получает состояние fit_activity_observed, а не progressing. Ни один marker сам по себе не доказывает завершение или корректность результата. stale_evaluation означает, что распознанный marker давно не появлялся; fit_without_optimizer_evidence означает длительный fit без marker. Эти состояния требуют диагностики, но не доказывают deadlock и не останавливают работу автоматически. dataset_checks показывает только факты метрик по сторонам и не выводит keep/drop."
        ),
        "automatic_action": "none; диагностика не перезапускает и не останавливает работу",
    }
