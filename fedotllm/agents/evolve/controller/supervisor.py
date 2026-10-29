"""Bounded unattended Evolve producer/consumer service for one campaign.

Only the fixed hunt and quality-drain CLI children are launched. Hunt and
quality run sequentially: the next hunt starts after the available quality
queue drains. Service duration includes both phases and idle queue waits.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.execution.checkout import source_commit, source_fingerprint
from fedotllm.agents.evolve.controller.release import (
    controller_version as _controller_version,
)
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import (
    clean_subprocess_env,
    fedot_python,
    run_worker,
)
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.quality_events import read_quality_events
from fedotllm.agents.evolve.execution.storage_guard import (
    require_storage,
    storage_failure_text,
    terminal_service_record,
)


@dataclass(frozen=True)
class QualityServiceConfig:
    duration_seconds: float = 604800
    poll_seconds: float = 30
    controller_timeout_seconds: float = 43200
    max_controller_restarts: int = 3
    hunt: bool = False
    target_proposals: int = 100
    max_hunt_campaigns: int = 200
    hunt_timeout_seconds: float = 3600
    llm_max_queries: int = 30
    llm_max_cost_usd: float = 0.25
    hunt_total_cost_usd: float = 10
    presets: str = "fedotllm:openrouter"

    def validate(self):
        for name in (
            "duration_seconds",
            "poll_seconds",
            "controller_timeout_seconds",
            "hunt_timeout_seconds",
            "llm_max_cost_usd",
            "hunt_total_cost_usd",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_controller_restarts",
            "target_proposals",
            "max_hunt_campaigns",
            "llm_max_queries",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "max_controller_restarts" else 1)
            ):
                raise ValueError(f"{name} must be a bounded integer")
        if not self.presets.strip():
            raise ValueError("presets must be explicit")


_PENDING = {"queued", "running", "partial", "skipped"}
_TERMINAL = {
    "keep",
    "drop",
    "timeout",
    "invalid",
    "candidate_error",
    "evaluation_error",
    "fedot_budget_timeout",
    "unverified_metric",
    "infrastructure_error",
    "failed",
    "apply_failed",
    "blocked",
}
_CLEANUP_SECONDS = 8


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_durable(path: Path, payload: dict):
    """Flush file and directory before treating a lifecycle transition as saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _service_event(workspace: Path, event: str, **fields):
    """Append one durable lifecycle transition without replacing old evidence."""
    append_journal(
        workspace / "logs" / "service_events.jsonl",
        {
            "event": event,
            "workspace": str(workspace),
            **fields,
        },
    )


def _validate_ledger(ledger, config):
    if ledger.get("budget_ledger_version") != 1:
        raise ValueError("service budget ledger is missing its version")
    for name in (
        "controller_restarts",
        "total_controller_restarts",
        "controller_launches",
        "hunt_campaigns",
    ):
        value = ledger.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid persisted budget ledger field: {name}")
    reserved = ledger.get("hunt_reserved_cost_usd")
    if (
        not isinstance(reserved, (int, float))
        or not math.isfinite(reserved)
        or reserved + 1e-8 < ledger["hunt_campaigns"] * config.llm_max_cost_usd
    ):
        raise ValueError("invalid persisted hunt cost reservation")
    children = ledger.get("children")
    if (
        not isinstance(children, list)
        or not all(isinstance(child, dict) for child in children)
        or len(children) != ledger["controller_launches"] + ledger["hunt_campaigns"]
    ):
        raise ValueError("incomplete persisted child launch ledger")


def _progress(workspace: Path, *, child_running: bool = False):
    journal = workspace / "quality_jobs.jsonl"
    # Only an active owned child permits a still-writing last row. Earlier
    # corruption always fails, even when durable side snapshots are available.
    rows = read_quality_events(journal, ignore_incomplete_tail=child_running)
    journal_bytes = journal.stat().st_size if journal.exists() else 0
    events = {}
    for row in rows:
        if row.get("event") in {
            "quality_task_side_finished",
            "quality_task_side_reused",
        }:
            events[
                (
                    str(row.get("candidate")),
                    str(row.get("task_id")),
                    str(row.get("side")),
                )
            ] = row.get("result") or {}
    groups = defaultdict(list)
    counts = Counter()
    pending = []
    for path in sorted((workspace / "quality_queue").glob("*.json")):
        if path.name.startswith("._"):
            continue
        item = _read_object(path)
        status = str(item.get("status") or "")
        if status not in _PENDING | _TERMINAL:
            raise ValueError(f"{path}: unknown proposal status {status!r}")
        candidate = str(item.get("candidate_id") or path.stem)
        proposal = str(item.get("proposal_id") or candidate)
        counts[status] += 1
        if status in _PENDING:
            pending.append(candidate)
        tasks = item.get("required_task_ids") or item.get("task_ids") or []
        measured = status in {"keep", "drop"} and bool(tasks)
        for task in tasks:
            for side in ("stock", "patch"):
                result = events.get((candidate, str(task), side), {})
                score = result.get("score")
                measured = (
                    measured
                    and result.get("status") == "ok"
                    and isinstance(score, (int, float))
                    and math.isfinite(score)
                    and isinstance(result.get("metric_observations"), dict)
                    and result["metric_observations"].get("metric_valid") is True
                )
        groups[proposal].append((status in _TERMINAL, bool(measured), status))
    terminal = sum(all(row[0] for row in revisions) for revisions in groups.values())
    measured = sum(all(row[1] for row in revisions) for revisions in groups.values())
    skipped = sum(
        all(row[2] == "skipped" for row in revisions) for revisions in groups.values()
    )
    finishes = {}
    for row in rows:
        if row.get("event") == "quality_task_side_finished":
            identity = (
                row.get("candidate"),
                row.get("task_id"),
                row.get("side"),
                row.get("attempt_id") or row.get("attempt", 1),
            )
            finishes[identity] = row
    return {
        "total_proposals": len(groups),
        "terminal_proposals": terminal,
        "measured_proposals": measured,
        "skipped_proposals": skipped,
        "pending_proposals": pending,
        "queue_counts": dict(counts),
        "journal_bytes": journal_bytes,
        "task_finished_events": len(finishes),
        "last_task_event_at": max(
            (str(row.get("ts") or "") for row in finishes.values()), default=""
        )
        or None,
    }


def _command(source, workspace, config, n_jobs, cpu_quota, stock_cache, kind):
    command = [
        fedot_python(source),
        "-m",
        "fedotllm.agents.evolve.controller"
        if kind == "quality-drain"
        else "fedotllm.agents.evolve",
        "drain" if kind == "quality-drain" else kind,
        "--fedot",
        str(source),
        "--workspace",
        str(workspace),
    ]
    if kind == "quality-drain":
        command += [
            "--continue-on-error",
            "--n-jobs",
            str(n_jobs),
            "--cpu-quota",
            str(cpu_quota),
        ]
        if stock_cache is not None:
            command += ["--stock-cache", str(stock_cache)]
    else:
        command += [
            "--presets",
            config.presets,
            "--llm-max-queries",
            str(config.llm_max_queries),
            "--llm-max-cost-usd",
            str(config.llm_max_cost_usd),
            "--findings",
            str(workspace / "findings.jsonl"),
        ]
    return command


def _run_child(command, *, source, workspace, timeout, log_path, kind, config, publish):
    cancellation = threading.Event()
    future = Future()
    env = clean_subprocess_env(
        source,
        repo_root=repo_root(),
        extra={
            "FEDOTLLM_REPO_PATH": str(source),
            "FEDOTLLM_REPO_PYTHON": fedot_python(source),
            "EVOLVE_AGENT_TMP": str(workspace / "worker_artifacts"),
        },
    )
    if "EVOLVE_QUALITY_DATA_CACHE" in os.environ:
        env["EVOLVE_QUALITY_DATA_CACHE"] = os.environ["EVOLVE_QUALITY_DATA_CACHE"]
    if kind == "run":
        for name in ("FEDOTLLM_LLM_API_KEY", "OPENROUTER_API_KEY"):
            if name in os.environ:
                env[name] = os.environ[name]

    def execute():
        try:
            future.set_result(
                run_worker(
                    command,
                    cwd=workspace,
                    env=env,
                    timeout=timeout,
                    log_path=log_path,
                    cancellation_event=cancellation,
                    resource_check=lambda: require_storage(workspace),
                )
            )
        except BaseException as exc:
            future.set_exception(exc)

    thread = threading.Thread(target=execute, daemon=True, name="quality-service-child")
    thread.start()
    child_deadline = time.monotonic() + timeout + _CLEANUP_SECONDS
    try:
        while not future.done():
            publish(child_running=True)
            remaining = child_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "owned child did not finish within watchdog cleanup deadline"
                )
            try:
                future.result(timeout=min(config.poll_seconds, remaining))
            except FutureTimeout:
                pass
        return future.result()
    finally:
        cancellation.set()
        if not future.done():
            try:
                future.result(timeout=_CLEANUP_SECONDS)
            except FutureTimeout as exc:
                raise RuntimeError(
                    "owned child cancellation did not finish in bounded cleanup"
                ) from exc


def run_quality_service(
    source: Path,
    workspace: Path,
    *,
    config: QualityServiceConfig,
    n_jobs: int = 4,
    cpu_quota: int = 28,
    stock_cache: Path | None = None,
) -> dict:
    """Serve this immutable campaign until its deadline or terminal target.

    Aggregate hunt cost is conservatively reserved at the full campaign cap
    BEFORE launch, never refunded, and survives service restarts. Provider
    per-request overshoot remains governed by the existing LLM budget boundary.
    """
    config.validate()
    if n_jobs != 4 or cpu_quota != 28:
        raise ValueError(
            "quality service requires the agreed 4 CPU/task, 28 CPU profile"
        )
    source, workspace = source.resolve(), workspace.resolve()
    if not source.is_dir():
        raise ValueError(f"source does not exist: {source}")
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".quality_service.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"quality service already owns {workspace}") from exc
        return _serve_locked(source, workspace, config, n_jobs, cpu_quota, stock_cache)


def _serve_locked(source, workspace, config, n_jobs, cpu_quota, stock_cache):
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    state_path = logs / "service_status.json"
    manifest_path = workspace / "service_manifest.json"
    frozen = {
        "config": asdict(config),
        "source": str(source),
        "source_hash": source_fingerprint(source),
        "source_commit": source_commit(source),
        "controller_hash": _controller_version(),
        "python": fedot_python(source),
        "python_version": sys.version,
        "n_jobs": n_jobs,
        "cpu_quota": cpu_quota,
        "stock_cache": str(stock_cache.resolve()) if stock_cache is not None else None,
        "throughput": "sequential hunt then available quality queue",
    }
    state = {
        "budget_ledger_version": 1,
        "state": "starting",
        "completion": "incomplete",
        "controller_restarts": 0,
        "total_controller_restarts": 0,
        "controller_launches": 0,
        "hunt_campaigns": 0,
        "hunt_reserved_cost_usd": 0.0,
        "child_running": False,
        "active_child": None,
        "children": [],
        "started_at": _now(),
    }
    ledger_trusted = False
    failed_child = None
    try:
        _service_event(
            workspace,
            "service_starting",
            controller_hash=frozen["controller_hash"],
            source_hash=frozen["source_hash"],
            profile={"n_jobs": n_jobs, "cpu_quota": cpu_quota},
        )
        if manifest_path.exists():
            manifest = _read_object(manifest_path)
            if manifest.get("frozen") != frozen:
                raise ValueError(
                    "frozen service source/configuration mismatch; use a fresh workspace"
                )
            launch_log = logs / "service-launch.log"
            final_record = (
                terminal_service_record(launch_log) if launch_log.is_file() else None
            )
            storage_record = (
                final_record
                if (
                    final_record
                    and final_record.get("child_running") is False
                    and final_record.get("active_child") is None
                    and (
                        final_record.get("state") == "storage_error"
                        or (
                            final_record.get("state") == "artifact_error"
                            and storage_failure_text(
                                str(final_record.get("error") or "")
                            )
                        )
                    )
                )
                else None
            )
            if not state_path.exists() and storage_record is None:
                raise ValueError(
                    "missing durable service ledger; refusing to reset hunt/restart budget"
                )
            saved_state = (
                dict(storage_record)
                if storage_record is not None
                else _read_object(state_path)
            )
            if storage_record is not None:
                saved_state["state"] = "storage_error"
            _validate_ledger(saved_state, config)
            state.update(saved_state)
            ledger_trusted = True
            if saved_state.get("state") == "storage_error":
                # Capacity recovery alone does not authorize repeating an
                # interrupted benchmark with unrecorded results.
                return state
            interrupted_child = state.get("active_child")
        else:
            if state_path.exists():
                raise ValueError("service ledger exists without frozen manifest")
            manifest = {
                "frozen": frozen,
                "started_at": state["started_at"],
                "deadline_epoch": time.time() + config.duration_seconds,
            }
            _write_durable(manifest_path, manifest)
            ledger_trusted = True
            interrupted_child = None
        remaining = max(0.0, manifest["deadline_epoch"] - time.time())
        deadline = time.monotonic() + remaining
        state.update(
            state="running",
            child_running=False,
            active_child=None,
            completion="incomplete",
        )

        def publish(*, child_running=False):
            state.update(_progress(workspace, child_running=child_running))
            state.update(
                heartbeat_at=_now(),
                child_running=child_running,
                remaining_seconds=max(0.0, deadline - time.monotonic()),
            )
            _write_durable(state_path, state)

        publish()
        _service_event(
            workspace,
            "service_started" if interrupted_child is None else "service_resumed",
            started_at=state.get("started_at"),
            deadline_epoch=manifest.get("deadline_epoch"),
            interrupted_child=interrupted_child,
            restart_count=state.get("total_controller_restarts"),
        )
        if interrupted_child is not None:
            # Guardian cleanup owns only the interrupted service's children.
            # Side events, not heartbeat freshness, reset the failure streak.
            state["last_interrupted_child"] = interrupted_child
            if interrupted_child.get("kind") == "quality-drain":
                state["total_controller_restarts"] += 1
                if state["task_finished_events"] > interrupted_child.get(
                    "task_finished_events", state["task_finished_events"]
                ):
                    state["controller_restarts"] = 0
                else:
                    state["controller_restarts"] += 1
            publish()
        while True:
            require_storage(workspace)
            publish()
            if (
                state["terminal_proposals"] >= config.target_proposals
                or (not config.hunt and state["terminal_proposals"] > 0)
            ) and not state["pending_proposals"]:
                state.update(
                    state="completed",
                    completion=(
                        "all_measured"
                        if state["measured_proposals"] == state["terminal_proposals"]
                        else "terminal_with_errors"
                    ),
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state["state"] = "duration_exhausted"
                break
            kind = None
            if state["pending_proposals"]:
                if state["controller_restarts"] > config.max_controller_restarts:
                    state["state"] = "restart_exhausted"
                    break
                kind = "quality-drain"
                state["controller_launches"] += 1
                number = state["controller_launches"]
                prefix = "controller"
                timeout = min(remaining, config.controller_timeout_seconds)
            elif (
                config.hunt
                and state["total_proposals"] < config.target_proposals
                and state["hunt_campaigns"] < config.max_hunt_campaigns
                and state["hunt_reserved_cost_usd"] + config.llm_max_cost_usd
                <= config.hunt_total_cost_usd + 1e-12
            ):
                kind = "run"
                state["hunt_campaigns"] += 1
                state["hunt_reserved_cost_usd"] += config.llm_max_cost_usd
                number = state["hunt_campaigns"]
                prefix = "hunt"
                timeout = min(remaining, config.hunt_timeout_seconds)
            if kind is None:
                if config.hunt and state["total_proposals"] < config.target_proposals:
                    state["state"] = (
                        "hunt_campaigns_exhausted"
                        if state["hunt_campaigns"] >= config.max_hunt_campaigns
                        else "hunt_budget_exhausted"
                    )
                    break
                time.sleep(min(config.poll_seconds, remaining))
                continue
            command = _command(
                source, workspace, config, n_jobs, cpu_quota, stock_cache, kind
            )
            log_path = logs / f"{prefix}-{number:04d}-{uuid.uuid4().hex[:8]}.log"
            state["active_child"] = {
                "kind": kind,
                "command": command,
                "log": str(log_path),
                "timeout_seconds": timeout,
                "started_at": _now(),
                "task_finished_events": state["task_finished_events"],
            }
            state["children"].append(state["active_child"])
            publish()
            _service_event(
                workspace,
                "child_started",
                kind=kind,
                child_number=number,
                log=str(log_path),
                timeout_seconds=timeout,
                task_finished_events=state["task_finished_events"],
                pending_proposals=len(state["pending_proposals"]),
            )
            previous_pending = state["pending_proposals"]
            outcome = _run_child(
                command,
                source=source,
                workspace=workspace,
                timeout=timeout,
                log_path=log_path,
                kind=kind,
                config=config,
                publish=publish,
            )
            state["last_child"] = dict(
                state["active_child"],
                returncode=outcome.returncode,
                timed_out=outcome.timed_out,
                cancelled=outcome.cancelled,
                finished_at=_now(),
            )
            state["children"][-1] = state["last_child"]
            previous_finished = state["active_child"]["task_finished_events"]
            _service_event(
                workspace,
                "child_finished",
                kind=kind,
                child_number=number,
                log=str(log_path),
                returncode=outcome.returncode,
                timed_out=outcome.timed_out,
                cancelled=outcome.cancelled,
                duration_seconds=(
                    datetime.fromisoformat(state["last_child"]["finished_at"])
                    - datetime.fromisoformat(state["active_child"]["started_at"])
                ).total_seconds(),
                task_finished_events=state["task_finished_events"],
                pending_proposals=len(state["pending_proposals"]),
            )
            state["active_child"] = None
            publish()
            if outcome.returncode == 75 or storage_failure_text(
                outcome.stdout + outcome.stderr
            ):
                state.update(
                    state="storage_error",
                    completion="incomplete",
                    error=f"storage failure in {log_path}; no automatic retry",
                )
                break
            if deadline - time.monotonic() <= 0:
                state["state"] = "duration_exhausted"
                break
            if kind == "quality-drain" and state["pending_proposals"]:
                # rc=3 is a normal drain with terminal dataset errors. A remaining
                # queue means an interrupted/abnormal attempt even when rc=0.
                state["total_controller_restarts"] += 1
                if state["task_finished_events"] > previous_finished:
                    state["controller_restarts"] = 0
                else:
                    state["controller_restarts"] += 1
                if state["controller_restarts"] > config.max_controller_restarts:
                    state["state"] = "restart_exhausted"
                    break
                if previous_pending == state["pending_proposals"]:
                    time.sleep(
                        min(config.poll_seconds, max(0.0, deadline - time.monotonic()))
                    )
        state.update(finished_at=_now(), child_running=False, active_child=None)
        _service_event(
            workspace,
            "service_finished",
            state=state.get("state"),
            completion=state.get("completion"),
            terminal_proposals=state.get("terminal_proposals"),
            measured_proposals=state.get("measured_proposals"),
            pending_proposals=len(state.get("pending_proposals") or []),
            finished_at=state["finished_at"],
        )
        publish()
        return state
    except ValueError as exc:
        if "frozen" in str(exc):
            raise
        failed_child = dict(state.get("active_child") or {}) or None
        state.update(
            state="artifact_error",
            completion="incomplete",
            error=f"{type(exc).__name__}: {exc}",
            child_running=False,
        )
    except (OSError, RuntimeError) as exc:
        failed_child = dict(state.get("active_child") or {}) or None
        state.update(
            state="storage_error"
            if isinstance(exc, OSError) and exc.errno in {28, 122}
            else "artifact_error",
            completion="incomplete",
            error=f"{type(exc).__name__}: {exc}",
            child_running=False,
            active_child=None,
        )
    active = failed_child or state.get("active_child")
    if active is not None:
        try:
            _service_event(
                workspace,
                "child_interrupted",
                kind=active.get("kind"),
                log=active.get("log"),
                started_at=active.get("started_at"),
                reason=f"{type(state.get('error') or 'interrupted').__name__}: {state.get('error')}",
                state=state.get("state"),
            )
        except OSError:
            pass
    try:
        _service_event(
            workspace,
            "service_failed",
            state=state.get("state"),
            error=state.get("error"),
            active_child=active,
        )
    except OSError:
        pass
    print(
        f"quality service failed in {workspace}: {state['error']}",
        file=sys.stderr,
        flush=True,
    )
    try:
        _write_durable(
            state_path if ledger_trusted else logs / "service_failure.json", state
        )
    except OSError as exc:
        print(
            f"quality service could not persist failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return state
