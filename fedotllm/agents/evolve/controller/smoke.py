"""Run isolated controller acceptance; default CLI mode only checks input paths.

Uses the production queue, patch checkout, FEDOT worker, watchdog and recovery.
The intentional import failure changes only an owned experimental copy.
Acceptance results cannot replace measurements on the full quality registry.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
from fedotllm.agents.evolve.controller.quality_queue import (
    enqueue_quality_job, list_queued_jobs, write_job_status,
)
from fedotllm.agents.evolve.controller.release import controller_version
from fedotllm.agents.evolve.evaluation.smoke_data import smoke_registry
from fedotllm.agents.evolve.execution.checkout import source_fingerprint
from fedotllm.agents.evolve.storage.journal import read_jsonl, write_json_atomic
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit


def _control_proposal(source: Path, *, broken: bool):
    path = "fedot/__init__.py"
    old = (source / path).read_text(encoding="utf-8")
    if not old:
        raise ValueError("smoke needs a nonempty FEDOT package initializer")
    # Append after the initializer so module docstrings/future imports stay valid.
    new = old.rstrip() + "\n" + (
        'raise RuntimeError("intentional controller smoke import failure")\n'
        if broken else "__evolve_controller_smoke__ = True\n"
    )
    proposal = PatchCandidate(
        "smoke-01-failure" if broken else "smoke-02-control", path,
        edits=[PatchEdit(path, old, new)],
    )
    patch = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))
    return proposal, patch


def _validate_workspace(source: Path, workspace: Path) -> Path:
    if workspace.is_symlink():
        raise ValueError("smoke workspace must not be a symlink")
    source, workspace = source.resolve(), workspace.resolve()
    if source == workspace or source in workspace.parents or workspace in source.parents:
        raise ValueError("smoke artifacts and FEDOT source must be disjoint")
    return workspace


def validate_worker_artifacts(workspace: Path, events: list[dict]) -> dict:
    """Check retained evidence for each worker actually executed by the smoke run."""
    workspace = Path(workspace).resolve()
    artifact_root = workspace / "worker_artifacts"
    checked = 0
    reused = 0
    waited_without_worker = 0
    issues = []

    def add_issue(event, artifact, reason, path=None):
        issues.append({
            "candidate":event.get("candidate"), "task_id":event.get("task_id"),
            "side":event.get("side"), "attempt":event.get("attempt"),
            "artifact":artifact, "reason":reason,
            "path":str(path) if path is not None else None,
        })

    def read_artifact(event, artifact, raw_path):
        if not isinstance(raw_path, str) or not raw_path:
            add_issue(event, artifact, "path_missing", raw_path)
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            add_issue(event, artifact, "path_not_absolute", raw_path)
            return None
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            add_issue(event, artifact, f"file_unavailable:{type(exc).__name__}", raw_path)
            return None
        if artifact_root not in resolved.parents:
            add_issue(event, artifact, "outside_worker_artifacts", resolved)
            return None
        if not resolved.is_file():
            add_issue(event, artifact, "not_a_regular_file", resolved)
            return None
        try:
            if resolved.stat().st_size <= 0:
                add_issue(event, artifact, "empty_file", resolved)
                return None
            return resolved.read_bytes()
        except OSError as exc:
            add_issue(event, artifact, f"file_unreadable:{type(exc).__name__}", resolved)
            return None

    for event in events:
        if event.get("event") != "quality_task_side_finished":
            continue
        result = event.get("result") or {}
        observations = result.get("metric_observations") or {}
        artifact_state = event.get("artifact_state")
        if (artifact_state in {"stock_cache_hit", "singleflight_shared", "reused"}
                or observations.get("cache_hit") or observations.get("singleflight_shared")):
            reused += 1
            continue

        if observations.get("worker_spawned") is False:
            waited_without_worker += 1
            continue

        checked += 1
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            add_issue(event, "worker_artifacts", "artifact_root_missing_or_symlink", artifact_root)
            continue

        required_identity = ("proposal_id", "task_id", "side", "attempt", "attempt_id")
        for key in required_identity:
            value = event.get(key)
            valid = type(value) is int and value > 0 if key == "attempt" else (
                isinstance(value, str) and bool(value)
            )
            if not valid:
                add_issue(event, "event", f"identity_missing:{key}")

        paths = event.get("worker_paths") or {}
        log_path = paths.get("worker_log_path") or result.get("worker_log_path")
        stage_path = paths.get("stage_status_path") or observations.get("stage_status_path")
        claimed_log = result.get("worker_log_path")
        claimed_stage = observations.get("stage_status_path")
        if claimed_log and log_path and Path(claimed_log) != Path(log_path):
            add_issue(event, "worker_log", "event_result_path_mismatch", claimed_log)
        if claimed_stage and stage_path and Path(claimed_stage) != Path(stage_path):
            add_issue(event, "stage_status", "event_result_path_mismatch", claimed_stage)

        read_artifact(event, "worker_log", log_path)
        stage_bytes = read_artifact(event, "stage_status", stage_path)
        if stage_bytes is None:
            continue
        try:
            stage = json.loads(stage_bytes)
            if not isinstance(stage, dict):
                raise ValueError("stage status must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            add_issue(event, "stage_status", f"invalid_json:{type(exc).__name__}", stage_path)
            continue
        expected_identity = {
            "task_id":event.get("task_id"), "side":event.get("side"),
            "attempt":event.get("attempt"), "attempt_id":event.get("attempt_id"),
            "proposal_id":event.get("proposal_id"),
        }
        for key, expected in expected_identity.items():
            if expected is None:
                continue
            if key not in stage:
                add_issue(event, "stage_status", f"identity_missing:{key}", stage_path)
            elif stage[key] != expected:
                add_issue(event, "stage_status", f"identity_mismatch:{key}", stage_path)

    return {
        "checked_workers":checked,
        "reused_results":reused,
        "waited_without_worker":waited_without_worker,
        "issues":issues,
        "complete":bool(checked) and not issues,
    }


def run_smoke(source: Path, workspace: Path) -> dict:
    workspace = _validate_workspace(source, workspace)
    source = source.resolve()
    registry = smoke_registry()
    fingerprint = source_fingerprint(source)
    proposals = [_control_proposal(source, broken=broken) for broken in (True, False)]
    workspace.mkdir(parents=True, exist_ok=False)
    # Keep the queue out of the legacy campaign/runs/<hunt-id> path heuristic.
    check = workspace / "check"
    check.mkdir()
    logs = workspace / "logs"
    logs.mkdir()
    report = {
        "purpose":"controller_acceptance", "passed":False,
        "time":datetime.now(timezone.utc).isoformat(),
        "controller_sha256":controller_version(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments":{"fedot_source":str(source), "workspace":str(workspace)},
        "affected_paths":[str(workspace)], "source_sha256":fingerprint,
        "registry":asdict(registry), "full_quality_run_started":False,
    }
    write_json_atomic(workspace / "acceptance.json", {**report, "stage":"started"})
    try:
        for proposal, patch in proposals:
            enqueue_quality_job(
                check, candidate=proposal, patch_text=patch,
                hint="Controller acceptance only", priority="high",
                source_root=source, registry=registry,
            )
        drain = drain_quality_queue(
            source, check, registry=registry, n_jobs=4, cpu_quota=12,
            artifact_root=workspace / "worker_artifacts",
            stock_cache_root=workspace / "stock_cache", continue_on_error=True,
        )
        write_json_atomic(logs / "drain.json", drain)
        events = read_jsonl(check / "quality_jobs.jsonl", strict=True)
        control = [row for row in events if row.get("event") == "fedot_quality"
                   and row.get("candidate") == "smoke-02-control"]
        metrics = control[-1]["jobs"] if control else []
        expected_ids = {dataset.task_id for dataset in registry.datasets}
        valid_metrics = (
            len(metrics) == 3 and {row["spec"]["source_dataset"] for row in metrics} == expected_ids
            and all(row[f"{side}_status"] == "ok"
                    and isinstance(row[f"{side}_score"], (int, float))
                    and math.isfinite(row[f"{side}_score"])
                    for row in metrics for side in ("stock", "patched"))
        )
        dispositions = {row["candidate_id"]:row["status"] for row in drain["jobs"]}
        unexpected_failures = []
        expected_failed_sides = set()
        for event in events:
            if event.get("event") != "quality_task_side_finished":
                continue
            result = event.get("result") or {}
            expected_failure = (
                event.get("candidate") == "smoke-01-failure" and event.get("side") == "patch"
                and result.get("status") == "crash"
                and "intentional controller smoke import failure" in result.get("detail", "")
            )
            if expected_failure:
                expected_failed_sides.add(event.get("task_id"))
            elif result.get("status") != "ok":
                unexpected_failures.append({"candidate":event.get("candidate"),
                                           "task_id":event.get("task_id"), "side":event.get("side"),
                                           "status":result.get("status"), "detail":result.get("detail")})
        failure_handled = (dispositions.get("smoke-01-failure") == "candidate_error"
                           and expected_failed_sides == expected_ids and not unexpected_failures)
        control_measured = dispositions.get("smoke-02-control") in {"keep", "drop"}
        recovery_ok = False
        if valid_metrics and control_measured:
            # Simulate interruption between durable results and terminal queue status.
            control_path = check / "quality_queue/smoke-02-control.json"
            write_job_status(control_path, "running")
            offset = len(events)
            recovered = drain_quality_queue(
                source, check, registry=registry, n_jobs=4, cpu_quota=12,
                artifact_root=workspace / "worker_artifacts",
                stock_cache_root=workspace / "stock_cache", continue_on_error=True,
            )
            write_json_atomic(logs / "recovery.json", recovered)
            new_events = read_jsonl(check / "quality_jobs.jsonl", strict=True)[offset:]
            reused = {(row.get("task_id"), row.get("side")) for row in new_events
                      if row.get("event") == "quality_task_side_reused"}
            recovery_ok = (
                reused == {(task, side) for task in expected_ids for side in ("stock", "patch")}
                and not any(row.get("event") == "quality_task_side_started" for row in new_events)
                and len(recovered["jobs"]) == 1
                and recovered["jobs"][0]["status"] in {"keep", "drop"}
            )
        queue_counts = dict(Counter(row["status"] for row in list_queued_jobs(check)))
        source_unchanged = source_fingerprint(source) == fingerprint
        sqlite_files = []
        artifact_bytes = 0
        for path in (workspace / "worker_artifacts").rglob("*"):
            if not path.is_file():
                continue
            artifact_bytes += path.stat().st_size
            with path.open("rb") as stream:
                sqlite_header = stream.read(16) == b"SQLite format 3\x00"
            if sqlite_header or any(part.endswith(".fedot-cache") for part in path.parts):
                sqlite_files.append(str(path))
        successful_sides = [row.get("result") or {} for row in events
                            if row.get("event") == "quality_task_side_finished"
                            and (row.get("result") or {}).get("status") == "ok"]
        cache_disabled = len(successful_sides) >= 6 and all(
            result.get("metric_observations", {}).get("fedot_cache_policy") == "disabled-sqlite-v1"
            and bool(result.get("metric_observations", {}).get("fedot_cache_flags"))
            and all(value is False for value in result["metric_observations"]["fedot_cache_flags"].values())
            for result in successful_sides)
        artifact_report = validate_worker_artifacts(workspace, events)
        report.update(
            stage="finished", metrics=metrics, queue_counts=queue_counts,
            failed_proposal_did_not_block=failure_handled and control_measured,
            unexpected_failures=unexpected_failures,
            recovery_without_refitting=recovery_ok, source_unchanged=source_unchanged,
            sqlite_cache_files=sqlite_files, ml_caches_disabled=cache_disabled,
            worker_artifact_bytes=artifact_bytes,
            worker_artifacts_complete=artifact_report["complete"],
            worker_artifacts_checked=artifact_report["checked_workers"],
            worker_artifacts_reused=artifact_report["reused_results"],
            worker_artifact_issues=artifact_report["issues"],
            passed=valid_metrics and failure_handled and control_measured and recovery_ok and source_unchanged
                   and cache_disabled and not sqlite_files and artifact_report["complete"],
        )
    except BaseException as exc:
        report.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        write_json_atomic(workspace / "acceptance.json", report)
        (logs / "drain.rc").write_text("3\n")
        raise
    write_json_atomic(workspace / "acceptance.json", report)
    (logs / "drain.rc").write_text("0\n" if report["passed"] else "3\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--fedot-source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    args = parser.parse_args(argv)
    for name in ("fedot_source", "python", "workspace"):
        if not getattr(args, name).is_absolute():
            parser.error(f"{name} must be absolute")
    if not args.python.is_file() or not (args.fedot_source / "fedot/__init__.py").is_file():
        parser.error("selected Python or FEDOT source missing")
    if args.workspace.exists() or args.workspace.is_symlink():
        parser.error("workspace already exists; preserving previous run")
    try:
        _validate_workspace(args.fedot_source, args.workspace)
    except ValueError as exc:
        parser.error(str(exc))
    version = controller_version()
    if version != args.expected_controller_sha256:
        parser.error("controller hash mismatch")
    if args.mode == "check":
        print(json.dumps({"result":"checked; no run started", "purpose":"controller_acceptance",
                          "controller_sha256":version, "profile":asdict(smoke_registry())}))
        return 0
    os.environ["FEDOTLLM_REPO_PYTHON"] = str(args.python)
    report = run_smoke(args.fedot_source, args.workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
