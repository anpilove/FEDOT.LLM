"""Executor for paired hour-long Fedot quality jobs."""

from __future__ import annotations

import json
import fcntl
import os
import hashlib
import math
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fedotllm.agents.evolve.controller.quality_queue import (
    campaign_workspace,
    list_queued_jobs,
    quality_result_message,
    write_job_status,
)
from fedotllm.agents.evolve.evaluation.fedot_quality import (
    _score_from_payload,
    create_worker_artifact_plan,
    decision_from_quality_jobs,
    run_quality_jobs,
    worker_artifact_paths,
)
from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples
from fedotllm.agents.evolve.evaluation.quality_registry import (
    QualityRegistry,
    list_quality_task_ids,
    load_quality_registry,
    select_task_ids_for_job,
)
from fedotllm.agents.evolve.evaluation.quality_scope import analyze_quality_scope
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.execution.process import interpreter_identity
from fedotllm.agents.evolve.execution.storage_guard import (
    StorageCapacityError,
    storage_failure_text,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.storage.journal import (
    JournalCorruptionError,
    append_journal,
    repair_incomplete_jsonl_tail,
    write_json_atomic,
)
from fedotllm.agents.evolve.storage.quality_events import (
    read_quality_events,
    save_quality_event,
)
from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    PatchEdit,
    ScoreResult,
)


class QualityQueueAlreadyRunning(RuntimeError):
    """Raised when another controller owns this campaign's quality queue."""


_CONTROLLER_TRACE_RESERVATION = threading.Lock()


@contextmanager
def _controller_stack_trace(workspace: Path):
    """Observe this process's drain threads without competing sampler owners."""
    if os.environ.get("EVOLVE_QUALITY_TRACE_PROCESSES") != "1":
        yield
        return

    logs = Path(workspace) / "logs"
    stack_path = logs / "controller.stacks.txt"
    metadata_path = logs / "controller.trace.json"
    metadata = {
        "schema_version": 1,
        "pid": os.getpid(),
        "scope": "controller_process_threads",
        "stack_trace_path": str(stack_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "enabled": False,
    }

    def warn(error):
        try:
            print(
                f"controller stack diagnostic warning: {error}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

    def save_metadata():
        try:
            write_json_atomic(metadata_path, metadata)
        except Exception as exc:
            warn(f"{type(exc).__name__}: {exc}")

    if not _CONTROLLER_TRACE_RESERVATION.acquire(blocking=False):
        metadata["state"] = "skipped_process_sampler_busy"
        save_metadata()
        warn(
            "another drain in this process already owns the stack sampler; capture skipped"
        )
        yield
        return

    sampler_context = None
    trace_status = {}
    try:
        try:
            interval = float(
                os.environ.get("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "15")
            )
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError(
                    "controller trace interval must be finite and positive"
                )
            sampler_context = periodic_stack_samples(
                stack_path,
                interval_seconds=interval,
                scope="controller_process_threads",
            )
            trace_status = sampler_context.__enter__()
            metadata.update(trace_status)
            metadata["state"] = "active" if trace_status["enabled"] else "disabled"
            if trace_status.get("error"):
                warn(trace_status["error"])
        except Exception as exc:
            metadata.update(state="disabled", error=f"{type(exc).__name__}: {exc}")
            warn(metadata["error"])
        save_metadata()
        yield
    finally:
        if sampler_context is not None:
            try:
                sampler_context.__exit__(None, None, None)
            except Exception as exc:
                warn(f"{type(exc).__name__}: {exc}")
            metadata.update(trace_status)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if metadata["enabled"]:
            metadata["state"] = "stopped"
        save_metadata()
        _CONTROLLER_TRACE_RESERVATION.release()


def _recover_durable_worker_result(
    event: dict,
    *,
    candidate_id: str,
    task_id: str,
    side: str,
    attempt: int,
    registry_identity: str,
) -> ScoreResult | None:
    """Read a finite worker result only when its adjacent stage identity matches."""

    try:
        paths = event.get("worker_paths")
        if not isinstance(paths, dict):
            return None
        recorded_result = event.get("worker_result_path") or paths.get(
            "worker_result_path"
        )
        recorded_stage = event.get("stage_status_path") or paths.get(
            "stage_status_path"
        )
        if not recorded_result or not recorded_stage:
            return None
        result_path = Path(recorded_result).expanduser()
        stage_path = Path(recorded_stage).expanduser()
        expected_attempt_id = f"{candidate_id}:{task_id}:{side}:{attempt}"
        evaluation_identity = event.get("evaluation_identity")
        expected_spec = event.get("spec")
        event_attempt = event.get("attempt")
        if (
            event.get("candidate") != candidate_id
            or event.get("task_id") != task_id
            or event.get("side") != side
            or type(event_attempt) is not int
            or event_attempt != attempt
            or event.get("attempt_id") != expected_attempt_id
            or not isinstance(evaluation_identity, dict)
            or not evaluation_identity
            or not isinstance(expected_spec, dict)
            or not {
                "n_jobs",
                "timeout_seconds",
                "fedot_search_budget_seconds",
            }.issubset(expected_spec)
        ):
            return None
        invocation_id = event.get("worker_invocation_id")
        if invocation_id is not None or paths.get("worker_invocation_id") is not None:
            if (
                not isinstance(invocation_id, str)
                or len(invocation_id) != 32
                or any(c not in "0123456789abcdef" for c in invocation_id)
                or paths.get("worker_invocation_id") != invocation_id
                or result_path.parent.name != invocation_id
            ):
                return None
            # The journal is the source of truth after controller restart; the
            # current environment's artifact root belongs to another invocation.
            expected_paths = paths
            if str(result_path) != paths.get("worker_result_path") or str(
                stage_path
            ) != paths.get("stage_status_path"):
                return None
        else:
            # Historical events predate physical invocation nonces.
            expected_paths = worker_artifact_paths(
                task_id, side, candidate_id, attempt, root=result_path.parent
            )
        if (
            result_path.is_symlink()
            or stage_path.is_symlink()
            or result_path.parent.is_symlink()
            or stage_path.parent.is_symlink()
            or result_path.resolve()
            != Path(expected_paths["worker_result_path"]).resolve()
            or stage_path.resolve()
            != Path(expected_paths["stage_status_path"]).resolve()
            or stage_path.resolve() != result_path.with_suffix(".stages.json").resolve()
        ):
            return None
        timeout_path = result_path.with_suffix(".timeout.json")
        recorded_timeout = paths.get("timeout_manifest_path")
        if (
            recorded_timeout
            and Path(recorded_timeout).expanduser().resolve() != timeout_path.resolve()
        ):
            return None
        if (
            timeout_path.is_file()
            and not timeout_path.is_symlink()
            and not timeout_path.parent.is_symlink()
            and timeout_path.resolve()
            == Path(expected_paths["timeout_manifest_path"]).resolve()
        ):
            timeout_payload = json.loads(timeout_path.read_text(encoding="utf-8"))
            if (
                not isinstance(timeout_payload, dict)
                or type(timeout_payload.get("schema_version")) is not int
                or timeout_payload.get("schema_version") != 1
                or timeout_payload.get("event") != "quality_worker_parent_timeout"
            ):
                return None
            timeout_identity = {
                "task_id": task_id,
                "source_dataset": task_id,
                "side": side,
                "attempt": attempt,
                "attempt_id": expected_attempt_id,
                "proposal_id": candidate_id,
                "registry_identity": registry_identity,
                "evaluation_identity": evaluation_identity,
            }
            if invocation_id is not None:
                timeout_identity["worker_invocation_id"] = invocation_id
            if any(
                timeout_payload.get(key) != value
                for key, value in timeout_identity.items()
            ):
                return None
            for key in ("worker_result_path", "stage_status_path", "worker_log_path"):
                expected_path = paths.get(key)
                if not expected_path or timeout_payload.get(key) != str(
                    Path(expected_path).expanduser()
                ):
                    return None
            timeout_kind = timeout_payload.get("timeout_kind")
            if (
                timeout_payload.get("status") != "timeout"
                or timeout_payload.get("failure_kind") != "timeout"
                or timeout_payload.get("metric_outcome") != "not_measured"
                or timeout_payload.get("metric_valid") is not False
                or timeout_kind not in {"task_timeout", "watchdog_timeout"}
            ):
                return None
            elapsed_values = (
                timeout_payload.get("side_elapsed_seconds"),
                timeout_payload.get("worker_elapsed_seconds"),
            )
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in elapsed_values
            ):
                return None
            side_elapsed, worker_elapsed = (float(value) for value in elapsed_values)
            if not all(
                math.isfinite(value) and value >= 0
                for value in (side_elapsed, worker_elapsed)
            ):
                return None
            observations = {
                "task_id": task_id,
                "side": side,
                "attempt": attempt,
                "attempt_id": expected_attempt_id,
                "proposal_id": candidate_id,
                "registry_identity": registry_identity,
                "evaluation_identity": evaluation_identity,
                "failure_kind": "timeout",
                "metric_outcome": "not_measured",
                "metric_valid": False,
                "timeout_kind": timeout_kind,
                "elapsed_seconds": side_elapsed,
                "side_elapsed_seconds": side_elapsed,
                "worker_elapsed_seconds": worker_elapsed,
                "timeout_manifest_path": str(timeout_path),
                "worker_result_path": str(result_path),
                "stage_status_path": str(stage_path),
                "worker_log_path": str(paths.get("worker_log_path") or ""),
                "stack_trace_path": timeout_payload.get("stack_trace_path"),
                "fedot_trace_dir": timeout_payload.get("fedot_trace_dir"),
                "recovered_parent_timeout_manifest": True,
                "worker_invocation_id": invocation_id,
                "worker_paths": dict(paths),
            }
            stage_observations = timeout_payload.get("stage_observations")
            if isinstance(stage_observations, dict):
                protected = {
                    "task_id",
                    "side",
                    "attempt",
                    "attempt_id",
                    "proposal_id",
                    "registry_identity",
                    "evaluation_identity",
                    "failure_kind",
                    "metric_outcome",
                    "metric_valid",
                    "timeout_kind",
                    "elapsed_seconds",
                    "side_elapsed_seconds",
                    "worker_elapsed_seconds",
                    "timeout_manifest_path",
                    "worker_result_path",
                    "stage_status_path",
                    "worker_log_path",
                    "recovered_parent_timeout_manifest",
                    "worker_invocation_id",
                    "worker_paths",
                }
                observations.update(
                    {
                        key: value
                        for key, value in stage_observations.items()
                        if key not in protected
                    }
                )
            return ScoreResult(
                task_id=task_id,
                status="timeout",
                score=float("nan"),
                detail=str(
                    timeout_payload.get("detail")
                    or f"{timeout_kind} recovered from timeout manifest"
                ),
                duration_s=side_elapsed,
                env_hash=f"{registry_identity}|{task_id}|{side}|{expected_spec.get('n_jobs', '')}",
                cmd="",
                log_tail=str(timeout_payload.get("log_tail") or "")[-4000:],
                metric_observations=observations,
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(stage, dict):
            return None
        if payload.get("task_id") != task_id:
            return None
        for artifact in (payload, stage):
            if (
                invocation_id is not None
                and artifact.get("worker_invocation_id") != invocation_id
            ):
                return None
            if (
                artifact.get("proposal_id") != candidate_id
                or artifact.get("attempt_id") != expected_attempt_id
                or artifact.get("source_dataset") != task_id
                or artifact.get("side") != side
                or type(artifact.get("attempt")) is not int
                or artifact.get("attempt") != attempt
                or artifact.get("registry_identity") != registry_identity
                or artifact.get("evaluation_identity") != evaluation_identity
            ):
                return None
        observations = payload.get("metric_observations")
        if not isinstance(observations, dict):
            return None
        expected_identity = {
            "proposal_id": candidate_id,
            "attempt_id": f"{candidate_id}:{task_id}:{side}:{attempt}",
            "task_id": task_id,
            "source_dataset": task_id,
            "side": side,
            "attempt": attempt,
            "registry_identity": registry_identity,
            "evaluation_identity": evaluation_identity,
        }
        if any(stage.get(key) != value for key, value in expected_identity.items()):
            return None
        if (
            payload.get("stage_status_path")
            and Path(payload["stage_status_path"]).expanduser().resolve()
            != stage_path.resolve()
        ):
            return None
        if payload.get("status") != "ok":
            # The worker publishes its exact terminal failure before the parent
            # can append the finished journal event. Recover that failure so a
            # controller restart does not erase FEDOT's phase/budget diagnosis.
            if (
                payload.get("status") not in {"crash", "timeout", "invalid"}
                or observations.get("metric_valid") is not False
                or observations.get("metric_outcome") != "not_measured"
                or not (payload.get("failure_kind") or observations.get("failure_kind"))
            ):
                return None
            result = _score_from_payload(
                task_id,
                payload,
                env_hash=f"{registry_identity}|{task_id}|{side}|{expected_spec.get('n_jobs', '')}",
                cmd="",
            )
            result.metric_observations.update(
                {
                    "worker_result_path": str(result_path),
                    "stage_status_path": str(stage_path),
                    "worker_log_path": str(paths.get("worker_log_path") or ""),
                    "recovered_worker_result": True,
                    "worker_invocation_id": invocation_id,
                    "worker_paths": dict(paths),
                }
            )
            return result
        if observations.get("metric_valid") is not True:
            return None
        for key in (
            "n_jobs",
            "timeout_seconds",
            "fedot_search_budget_seconds",
            "preset",
            "seed",
            "cv_folds",
            "max_pipeline_fit_time_seconds",
            "max_pipeline_fit_policy",
            "max_pipeline_evaluation_time_seconds",
            "max_pipeline_evaluation_policy",
        ):
            if key in expected_spec and stage.get(key) != expected_spec[key]:
                return None
        result = _score_from_payload(
            task_id,
            payload,
            env_hash=f"{registry_identity}|{task_id}|{side}|{expected_spec.get('n_jobs', '')}",
            cmd="",
        )
        result.metric_observations.update(
            {
                "worker_result_path": str(result_path),
                "stage_status_path": str(stage_path),
                "worker_log_path": str(paths.get("worker_log_path") or ""),
                "recovered_worker_result": True,
                "worker_invocation_id": invocation_id,
                "worker_paths": dict(paths),
            }
        )
        return result
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError, KeyError):
        return None


@contextmanager
def quality_queue_lock(workspace: Path):
    """Hold an OS-backed, non-blocking lock for the campaign's quality queue."""

    root = campaign_workspace(workspace)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".quality_queue.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QualityQueueAlreadyRunning(
                f"quality queue already has an active controller: {root}"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _recover_side_state(
    journal: Path,
    *,
    candidate_id: str,
    task_ids: tuple[str, ...],
    registry_identity: str,
    evaluation_identity: dict | None = None,
) -> tuple[dict[tuple[str, str], ScoreResult], dict[tuple[str, str], int]]:
    """Restore the latest finished side result and attempt count from durable events."""

    wanted_tasks = set(task_ids)
    attempts: dict[tuple[str, str], int] = {}
    results: dict[tuple[str, str], ScoreResult] = {}
    latest_phase: dict[tuple[str, str], str] = {}
    latest_event: dict[tuple[str, str], dict] = {}
    if journal.exists() and not journal.is_char_device():
        tail_repair = repair_incomplete_jsonl_tail(journal)
        if tail_repair is not None and journal != Path("/dev/null"):
            repaired = save_quality_event(
                journal,
                {"event": "quality_journal_tail_repaired", **tail_repair},
            )
            append_journal(journal, repaired)
    for event in read_quality_events(
        journal, candidate_id=candidate_id, ignore_incomplete_tail=True
    ):
        if (
            str(event.get("candidate") or "") != candidate_id
            or str(event.get("registry_identity") or "") != registry_identity
            or (
                evaluation_identity is not None
                and event.get("evaluation_identity") != evaluation_identity
            )
        ):
            continue
        task_id = str(event.get("task_id") or "")
        side = str(event.get("side") or "")
        if task_id not in wanted_tasks or side not in {"stock", "patch"}:
            continue
        key = (task_id, side)
        phase = str(event.get("event") or "")
        attempt = int(event.get("attempt") or 0)
        attempts[key] = max(attempts.get(key, 0), attempt)
        latest_event[key] = event
        is_active_attempt = phase in {
            "quality_task_side_started",
            "quality_task_worker_spawned",
            "quality_task_side_waiting_stock_cache",
        }
        latest_phase[key] = "quality_task_side_started" if is_active_attempt else phase
        if is_active_attempt:
            results.pop(key, None)
        if phase not in {
            "quality_task_side_finished",
            "quality_task_side_reused",
            "quality_task_side_interrupted",
        }:
            continue
        payload = event.get("result") or {}
        if not isinstance(payload, dict):
            continue
        try:
            result = _score_from_payload(
                task_id,
                payload,
                env_hash=str(payload.get("env_hash") or ""),
                cmd=str(payload.get("cmd") or ""),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raw_observations = payload.get("metric_observations")
            observations = (
                dict(raw_observations) if isinstance(raw_observations, dict) else {}
            )
            observations["failure_kind"] = "unverified_metric"
            observations["metric_validation_error"] = str(exc)
            result = ScoreResult(
                task_id=task_id,
                status="invalid",
                score=float("nan"),
                detail="recovered result failed metric validation",
                duration_s=0.0,
                metric_observations=observations,
            )
        if not isinstance((result.metric_observations or {}).get("worker_paths"), dict):
            recorded_paths = event.get("worker_paths")
            if isinstance(recorded_paths, dict):
                result.metric_observations["worker_paths"] = dict(recorded_paths)
                if recorded_paths.get("worker_invocation_id"):
                    result.metric_observations["worker_invocation_id"] = recorded_paths[
                        "worker_invocation_id"
                    ]
        results[key] = result

    # A process can stop after the durable `started` event but before `finished`.
    # Preserve that attempt and allow exactly one recovery attempt under the normal
    # retry budget; an unfinished second attempt remains unresolved, not looped.
    for key, phase in latest_phase.items():
        if phase != "quality_task_side_started" or key in results:
            continue
        task_id, _side = key
        previous = latest_event[key]
        recovered_result = _recover_durable_worker_result(
            previous,
            candidate_id=candidate_id,
            task_id=task_id,
            side=_side,
            attempt=attempts[key],
            registry_identity=registry_identity,
        )
        if recovered_result is not None:
            results[key] = recovered_result
            if journal != Path("/dev/null") and not journal.is_char_device():
                terminal = {
                    **{
                        name: previous[name]
                        for name in (
                            "candidate",
                            "candidate_revision_id",
                            "proposal_id",
                            "patch_revision",
                            "task_id",
                            "side",
                            "attempt",
                            "attempt_id",
                            "registry_identity",
                            "evaluation_identity",
                            "n_jobs_per_side",
                            "cpu_quota",
                            "dataset_pair_slots",
                            "resource_budget_scope",
                            "worker_paths",
                            "worker_invocation_id",
                            "worker_spec_path",
                            "worker_result_path",
                            "worker_log_path",
                            "stage_status_path",
                            "hypothesis",
                            "quality_scope",
                            "spec",
                        )
                        if name in previous
                    },
                    "event": "quality_task_side_finished",
                    "artifact_state": "recovered_worker_result",
                    "result": {
                        "status": recovered_result.status,
                        "score": (
                            recovered_result.score
                            if math.isfinite(float(recovered_result.score))
                            else None
                        ),
                        "detail": recovered_result.detail,
                        "traceback": recovered_result.traceback,
                        "duration_s": recovered_result.duration_s,
                        "n_train": recovered_result.n_train,
                        "seed": recovered_result.seed,
                        "env_hash": recovered_result.env_hash,
                        "cmd": recovered_result.cmd,
                        "worker_log_path": recovered_result.metric_observations.get(
                            "worker_log_path"
                        ),
                        "metric_observations": recovered_result.metric_observations,
                        "log_tail": recovered_result.log_tail,
                    },
                }
                append_journal(journal, save_quality_event(journal, terminal))
            continue
        detail = "controller interrupted while this task side was running"
        results[key] = ScoreResult(
            task_id=task_id,
            status="crash",
            score=float("nan"),
            detail=detail,
            metric_observations={
                "failure_kind": "controller_interrupted",
                "interrupted_after_event": latest_event[key].get("event"),
            },
        )
        # Make recovery itself durable. Otherwise the next launch only sees an
        # old "started" row and cannot distinguish recovery from a fresh retry.
        if journal != Path("/dev/null") and not journal.is_char_device():
            recovered = {
                **{
                    name: previous[name]
                    for name in (
                        "candidate",
                        "candidate_revision_id",
                        "proposal_id",
                        "patch_revision",
                        "task_id",
                        "side",
                        "attempt",
                        "attempt_id",
                        "registry_identity",
                        "evaluation_identity",
                        "n_jobs_per_side",
                        "cpu_quota",
                        "dataset_pair_slots",
                        "resource_budget_scope",
                        "worker_paths",
                        "worker_invocation_id",
                        "worker_spec_path",
                        "worker_result_path",
                        "worker_log_path",
                        "stage_status_path",
                        "hypothesis",
                        "quality_scope",
                        "spec",
                    )
                    if name in previous
                },
                "event": "quality_task_side_interrupted",
                "artifact_state": "interrupted",
                "interrupted_after_event": previous.get("event"),
                "result": {
                    "status": "crash",
                    "score": None,
                    "detail": detail,
                    "duration_s": 0.0,
                    "metric_observations": {
                        "failure_kind": "controller_interrupted",
                        "interrupted_after_event": previous.get("event"),
                    },
                },
            }
            append_journal(journal, save_quality_event(journal, recovered))
    return results, attempts


def _quality_delta(spec: dict, stock: dict, patched: dict) -> float | None:
    if (
        stock.get("status") != "ok"
        or patched.get("status") != "ok"
        or stock.get("score") is None
        or patched.get("score") is None
        or (stock.get("metric_observations") or {}).get("metric_valid") is not True
        or (patched.get("metric_observations") or {}).get("metric_valid") is not True
    ):
        return None
    before = float(stock["score"])
    after = float(patched["score"])
    delta = after - before if spec["higher_is_better"] else before - after
    if spec.get("delta_mode", "absolute") == "relative":
        delta /= max(abs(before), 1e-12)
    return delta


def measure_fedot_quality(
    source: Path,
    experiment: Path,
    *,
    journal: Path | None = None,
    candidate: PatchCandidate | None = None,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    hypothesis: dict | None = None,
    quality_scope: dict | None = None,
    required_task_ids: tuple[str, ...] | None = None,
    registry: QualityRegistry | None = None,
    pair_scheduler: SharedPairScheduler | None = None,
    pair_group_id: int | None = None,
    proposal_id: str | None = None,
    patch_revision: str | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
) -> Decision:
    """Hour-long Fedot is the only quality verdict. Hunt must enqueue, not inline-run."""

    if artifact_root is None and journal is not None:
        artifact_root = journal.parent / "worker_artifacts"
    if stock_cache_root is None and journal is not None:
        stock_cache_root = journal.parent / "stock_cache"
    legacy_callback_plans: dict[tuple[str, str, int], dict] = {}

    def event_worker_paths(spec, side, attempt):
        paths = spec.get("worker_paths")
        if isinstance(paths, dict) and paths.get("worker_result_path"):
            return dict(paths)
        # Older injected runners do not carry a plan. Give their callbacks a
        # stable plan too; production pairs always supply their own before start.
        key = (str(spec["source_dataset"]), side, attempt)
        if key not in legacy_callback_plans:
            legacy_callback_plans[key] = create_worker_artifact_plan(
                key[0],
                side,
                candidate.candidate_id if candidate is not None else experiment.name,
                attempt,
                artifact_root=artifact_root,
            )
        return dict(legacy_callback_plans[key])

    registry = registry or load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    effective_n_jobs = int(registry.n_jobs_per_job if n_jobs is None else n_jobs)
    effective_cpu_quota = int(registry.cpu_quota if cpu_quota is None else cpu_quota)
    evaluation_identity = {
        "registry_sha256": hashlib.sha256(
            json.dumps(asdict(registry), sort_keys=True).encode()
        ).hexdigest(),
        "stock_source_sha256": source_fingerprint(source),
        "patch_source_sha256": source_fingerprint(experiment),
        "n_jobs": effective_n_jobs,
        "interpreter": interpreter_identity(source),
    }

    def effective_pair_slots() -> int:
        if pair_scheduler is not None:
            return pair_scheduler.max_pairs
        return max(1, effective_cpu_quota // effective_n_jobs)

    def scheduler_observations() -> dict:
        if pair_scheduler is None:
            return {
                "campaign_pair_cap": effective_pair_slots(),
                "scheduler_active_pair_slots": None,
                "scheduler_queued_pair_tasks": None,
            }
        return {
            "campaign_pair_cap": pair_scheduler.max_pairs,
            "scheduler_active_pair_slots": pair_scheduler.active_pair_slots,
            "scheduler_queued_pair_tasks": pair_scheduler.queued_pair_tasks,
            "scheduler_active_compute_workers": pair_scheduler.active_compute_workers,
            "scheduler_peak_compute_workers": pair_scheduler.peak_compute_workers,
        }

    def record_side_event(spec, side, phase, result, attempt):
        if journal is None:
            return
        candidate_id = (
            candidate.candidate_id if candidate is not None else experiment.name
        )
        source_proposal_id = proposal_id or candidate_id
        worker_paths = event_worker_paths(spec, side, attempt)
        event_name = (
            "quality_task_worker_spawned"
            if phase == "spawned"
            else f"quality_task_side_{phase}"
        )
        payload = {
            "event": event_name,
            "candidate": candidate_id,
            "candidate_revision_id": candidate_id,
            "proposal_id": source_proposal_id,
            "patch_revision": patch_revision,
            "task_id": spec["source_dataset"],
            "side": side,
            "attempt": attempt,
            "attempt_id": f"{candidate_id}:{spec['source_dataset']}:{side}:{attempt}",
            "registry_identity": registry.identity,
            "evaluation_identity": evaluation_identity,
            "n_jobs_per_side": effective_n_jobs,
            "cpu_quota": effective_cpu_quota,
            "dataset_pair_slots": effective_pair_slots(),
            "resource_budget_scope": "campaign-shared"
            if pair_scheduler is not None
            else "call-local",
            **scheduler_observations(),
            "stock_flight_id": spec.get("stock_flight_id"),
            "worker_paths": worker_paths,
            "worker_invocation_id": worker_paths.get("worker_invocation_id"),
            "worker_spec_path": worker_paths["worker_spec_path"],
            "worker_result_path": worker_paths["worker_result_path"],
            "worker_log_path": worker_paths["worker_log_path"],
            "stage_status_path": worker_paths["stage_status_path"],
            "artifact_state": {
                "started": "expected",
                "spawned": "spawned",
                "finished": "finished",
                "reused": "reused",
            }.get(phase, phase),
            "hypothesis": hypothesis or {},
            "quality_scope": quality_scope or {},
            "spec": spec,
        }
        worker_pid = spec.get("worker_pid") if phase == "spawned" else None
        if (
            isinstance(worker_pid, int)
            and not isinstance(worker_pid, bool)
            and worker_pid > 0
        ):
            payload["worker_pid"] = worker_pid
        if "scheduler_queue_wait_seconds" in spec:
            payload["scheduler_queue_wait_seconds"] = spec[
                "scheduler_queue_wait_seconds"
            ]
        if result is not None:
            observations = dict(result.metric_observations or {})
            if "worker_spawned" in observations:
                payload["worker_spawned"] = observations["worker_spawned"]
            observed_paths = {
                "worker_log_path": str(
                    observations.get("worker_log_path")
                    or worker_paths["worker_log_path"]
                ),
                "worker_result_path": str(
                    observations.get("worker_result_path")
                    or worker_paths["worker_result_path"]
                ),
                "stage_status_path": str(
                    observations.get("stage_status_path")
                    or worker_paths["stage_status_path"]
                ),
            }
            if observations.get("singleflight_shared") or observations.get("cache_hit"):
                payload["shared_worker_paths"] = dict(
                    observations.get("worker_paths") or observed_paths
                )
                if observations.get("worker_invocation_id"):
                    payload["shared_worker_paths"]["worker_invocation_id"] = (
                        observations["worker_invocation_id"]
                    )
            else:
                # Production worker paths are immutable from the started event.
                # Legacy test runners can report only logs, without a plan.
                if not spec.get("worker_paths"):
                    worker_paths.update(observed_paths)
            payload.update(
                {
                    "worker_paths": worker_paths,
                    "worker_log_path": worker_paths["worker_log_path"],
                    "worker_result_path": worker_paths["worker_result_path"],
                    "stage_status_path": worker_paths["stage_status_path"],
                    "artifact_state": (
                        "reused"
                        if phase == "reused"
                        else "stock_cache_hit"
                        if observations.get("cache_hit")
                        else "singleflight_shared"
                        if observations.get("singleflight_shared")
                        else "timed_out_waiting_stock_cache"
                        if observations.get("worker_spawned") is False
                        else "finished"
                    ),
                }
            )
            for key in (
                "wait_kind",
                "timeout_kind",
                "timeout_limit_seconds",
                "side_elapsed_seconds",
                "worker_elapsed_seconds",
                "worker_timeout_budget_seconds",
            ):
                if key in observations:
                    payload[key] = observations[key]
            payload["result"] = {
                "status": result.status,
                "score": result.score if math.isfinite(float(result.score)) else None,
                "detail": result.detail,
                "traceback": result.traceback,
                "duration_s": result.duration_s,
                "n_train": result.n_train,
                "seed": result.seed,
                "env_hash": result.env_hash,
                "cmd": result.cmd,
                "worker_log_path": (result.metric_observations or {}).get(
                    "worker_log_path"
                ),
                "metric_observations": result.metric_observations,
                "log_tail": result.log_tail,
            }
        append_journal(journal, save_quality_event(journal, payload))

    def record_side_wait(spec, side, wait_kind, elapsed_seconds, attempt):
        if journal is None:
            return
        candidate_id = (
            candidate.candidate_id if candidate is not None else experiment.name
        )
        event = save_quality_event(
            journal,
            {
                "event": "quality_task_side_waiting_stock_cache",
                "candidate": candidate_id,
                "candidate_revision_id": candidate_id,
                "proposal_id": proposal_id or candidate_id,
                "patch_revision": patch_revision,
                "task_id": spec["source_dataset"],
                "side": side,
                "attempt": attempt,
                "attempt_id": f"{candidate_id}:{spec['source_dataset']}:{side}:{attempt}",
                "registry_identity": registry.identity,
                "evaluation_identity": evaluation_identity,
                "n_jobs_per_side": effective_n_jobs,
                "cpu_quota": effective_cpu_quota,
                "dataset_pair_slots": effective_pair_slots(),
                "resource_budget_scope": "campaign-shared"
                if pair_scheduler is not None
                else "call-local",
                **scheduler_observations(),
                "worker_paths": event_worker_paths(spec, side, attempt),
                "worker_invocation_id": event_worker_paths(spec, side, attempt)[
                    "worker_invocation_id"
                ],
                "wait_kind": wait_kind,
                "wait_elapsed_seconds": elapsed_seconds,
                "pair_slot_held_during_wait": False,
                "wait_does_not_hold_compute_worker": pair_scheduler is not None,
                "singleflight_role": "waiter"
                if pair_scheduler is not None
                else "synchronous_waiter",
                "stock_flight_id": spec.get("stock_flight_id"),
                "owner_worker_shared": pair_scheduler is not None,
                "worker_spawned": False,
                "stock_wait_subscriber_mode": (
                    "nonblocking_future"
                    if pair_scheduler is not None
                    else "synchronous"
                ),
                "artifact_state": "waiting_stock_cache",
                "hypothesis": hypothesis or {},
                "quality_scope": quality_scope or {},
                "spec": spec,
            },
        )
        append_journal(journal, event)

    prior_side_results, prior_attempt_counts = _recover_side_state(
        journal or Path("/dev/null"),
        candidate_id=candidate.candidate_id
        if candidate is not None
        else experiment.name,
        task_ids=tuple(ids),
        registry_identity=registry.identity,
        evaluation_identity=evaluation_identity,
    )
    rows = run_quality_jobs(
        stock_checkout=source,
        patch_checkout=experiment,
        task_ids=ids,
        n_jobs=n_jobs,
        cpu_quota=cpu_quota,
        registry=registry,
        on_side_event=record_side_event if journal is not None else None,
        prior_side_results=prior_side_results,
        prior_attempt_counts=prior_attempt_counts,
        proposal_id=candidate.candidate_id
        if candidate is not None
        else experiment.name,
        evaluation_identity=evaluation_identity,
        artifact_root=artifact_root,
        stock_cache_root=stock_cache_root,
        pair_scheduler=pair_scheduler,
        pair_group_id=pair_group_id,
        on_side_wait=record_side_wait if journal is not None else None,
    )
    decision = decision_from_quality_jobs(rows)
    if required_task_ids is not None and set(ids) != set(required_task_ids):
        decision.keep = None
        decision.final_keep = None
        decision.quality_outcome = "not_evaluated"
        decision.reason = "partial_dataset_check; full frozen scope remains required"
        decision.unresolved_task_ids = tuple(
            dict.fromkeys(
                (
                    *decision.unresolved_task_ids,
                    *(item for item in required_task_ids if item not in ids),
                )
            )
        )
    if journal is not None:
        append_journal(
            journal,
            {
                "event": "fedot_quality",
                "candidate": None if candidate is None else candidate.candidate_id,
                "registry_identity": registry.identity,
                "evaluation_identity": evaluation_identity,
                "task_ids": list(ids),
                "required_task_ids": list(required_task_ids or ids),
                "hypothesis": hypothesis or {},
                "quality_scope": quality_scope or {},
                "decision": asdict(decision),
                "jobs": [
                    {
                        "spec": row["spec"],
                        "search_ran": row["search_ran"],
                        "stock_status": row["stock"]["status"],
                        "patched_status": row["patched"]["status"],
                        "stock_score": row["stock"]["score"],
                        "patched_score": row["patched"]["score"],
                        "stock_detail": row["stock_result"].detail[:1000],
                        "patched_detail": row["patched_result"].detail[:1000],
                        "stock_traceback": row["stock_result"].traceback[-4000:],
                        "patched_traceback": row["patched_result"].traceback[-4000:],
                        "stock_observations": row["stock_result"].metric_observations,
                        "patched_observations": row[
                            "patched_result"
                        ].metric_observations,
                        "quality_delta": _quality_delta(
                            row["spec"], row["stock"], row["patched"]
                        ),
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


def apply_queue_patch(
    checkout: Path, *, candidate_id: str, file_path: str, patch_text: str
) -> bool:
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
    """Return every required dataset, recovering old empty-scope skips safely."""

    all_ids = tuple(list_quality_task_ids(registry))
    recorded = tuple(str(item) for item in (job.get("task_ids") or ()) if str(item))
    required = tuple(
        str(item) for item in (job.get("required_task_ids") or ()) if str(item)
    )
    if job.get("quality_scope_frozen"):
        # A frozen required list is the contract; don't silently skip its
        # datasets when a requested pool has no overlap. Ordinary queued
        # entries may run an explicit subset and remain partial. Old skipped
        # entries are different: retry their entire required set, even if their
        # runnable list was overwritten with an empty value.
        was_skipped = str(job.get("status") or "") == "skipped"
        frozen_required = (
            (required or recorded) if was_skipped else (recorded or required)
        )
        selected = tuple(task_id for task_id in frozen_required if task_id in all_ids)
        if selected:
            if was_skipped:
                return selected
            available = tuple(task_id for task_id in selected if task_id in pool)
            return available or selected
        if job.get("quality_scope"):
            selected = select_task_ids_for_job(
                job, requested=all_ids, registry=registry
            )
            if selected:
                return tuple(selected)
        # Legacy skipped entries may have lost both lists and scope evidence.
        # Recheck the full registry rather than treating the proposal as done.
        return all_ids
    selected = select_task_ids_for_job(
        job, requested=recorded or required or pool or all_ids, registry=registry
    )
    if selected:
        return tuple(task_id for task_id in selected if task_id in all_ids) or tuple(
            selected
        )
    fallback = select_task_ids_for_job(
        job, requested=required or all_ids, registry=registry
    )
    return tuple(fallback) if fallback else all_ids


def _drain_quality_queue_locked(
    source: Path,
    workspace: Path,
    *,
    journal: Path | None = None,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    max_pair_slots: int = 7,
    stock_cache: Path | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
    continue_on_error: bool = False,
    registry: QualityRegistry | None = None,
) -> dict:
    """Reuse matching stock cache, warm only missing tasks, then score patches.

    Toy δ=0 / probe ``no_change`` never skip a job. Hunt does not run this.
    Default: classification/regression OpenML for tabular patches, all eight
    registered TS tasks for forecasting patches.
    Inapplicable tasks are omitted from the intended frozen scope. A legacy
    skipped entry with no runnable IDs is recovered from required IDs, then
    saved scope, and finally the full registry if that routing evidence was lost.
    Skipped work is never a quality verdict.
    Re-scan the queue after each batch so later rsync arrivals are scored.
    """

    registry = registry or load_quality_registry()
    pool = task_ids or list_quality_task_ids(registry)
    workspace = workspace.resolve()
    artifact_root = (
        Path(artifact_root).expanduser().resolve()
        if artifact_root is not None
        else workspace / "worker_artifacts"
    )
    stock_cache_root = (
        Path(stock_cache_root or stock_cache).expanduser().resolve()
        if (stock_cache_root is not None or stock_cache is not None)
        else workspace / "stock_cache"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    journal = journal or (workspace / "quality_jobs.jsonl")
    processed: set[str] = set()
    results: list[dict] = []
    result_order: dict[str, int] = {}

    paused_for_retry = threading.Event()
    while True:
        queued = [
            job
            for job in list_queued_jobs(workspace)
            if str(job.get("status") or "queued")
            in {"queued", "running", "partial", "skipped", ""}
            and str(job.get("candidate_id") or job.get("_path") or "") not in processed
        ]
        if not queued:
            break

        for job in queued:
            candidate_id = str(job.get("candidate_id") or job.get("_path") or "")
            result_order.setdefault(candidate_id, len(result_order))

        def _process_job(job, proposal_cpu_quota, pair_group_id):
            candidate_id = str(job.get("candidate_id") or "")
            job_key = candidate_id or str(job.get("_path") or "")
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
            selection_job = dict(job)
            if continue_on_error and job.get("required_task_ids"):
                selection_job["task_ids"] = job["required_task_ids"]
            job_ids = _job_quality_task_ids(selection_job, pool, registry)
            scope = (
                job.get("quality_scope") if job.get("quality_scope_frozen") else None
            )
            scope = scope or analyze_quality_scope(job, registry)
            required_ids = tuple(
                job.get("required_task_ids") or job.get("task_ids") or job_ids
            )
            blocked_reason = ""
            if (
                job.get("registry_identity")
                and job["registry_identity"] != registry.identity
            ):
                blocked_reason = "registry_mismatch"
            elif (job.get("check_identity") or {}).get("registry_sha256") not in {
                None,
                hashlib.sha256(
                    json.dumps(asdict(registry), sort_keys=True).encode()
                ).hexdigest(),
            }:
                blocked_reason = "registry_mismatch"
            elif set(required_ids) - set(list_quality_task_ids(registry)):
                blocked_reason = "unknown_required_datasets"
            elif job.get("source_sha256"):
                from fedotllm.agents.evolve.execution.checkout import source_fingerprint

                if source_fingerprint(source) != job["source_sha256"]:
                    blocked_reason = "source_mismatch"
            if job.get("quality_scope_frozen") and not job.get("scope_policy_version"):
                fresh_job = {
                    key: value for key, value in job.items() if key != "quality_scope"
                }
                expected_ids = select_task_ids_for_job(
                    fresh_job,
                    requested=list_quality_task_ids(registry),
                    registry=registry,
                )
                if set(required_ids) != set(expected_ids):
                    blocked_reason = blocked_reason or "routing_scope_changed"
            if blocked_reason:
                item = {
                    "candidate_id": candidate_id,
                    "status": "blocked",
                    "reason": blocked_reason,
                    "keep": None,
                    "quality_outcome": "not_evaluated",
                    "task_ids": list(required_ids),
                }
                write_job_status(job_path, "blocked", item)
                append_journal(
                    journal,
                    {**item, "event": "quality_disposition", "candidate": candidate_id},
                )
                results.append(item)
                processed.add(job_key)
                return
            if not job_ids:
                quality_message = quality_result_message(
                    {
                        "candidate_id": candidate_id,
                        "status": "blocked",
                        "reason": "required_datasets_unavailable",
                        "task_ids": [],
                        "hypothesis": job.get("hypothesis") or {},
                    }
                )
                write_job_status(
                    job_path,
                    "blocked",
                    {
                        "reason": "required_datasets_unavailable",
                        "keep": None,
                        "quality_message": quality_message,
                    },
                )
                append_journal(
                    journal,
                    {
                        "event": "quality_disposition",
                        "candidate": candidate_id,
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "task_ids": [],
                        "status": "blocked",
                        "reason": "required_datasets_unavailable",
                        "quality_message": quality_message,
                    },
                )
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "blocked",
                        "reason": "required_datasets_unavailable",
                        "keep": None,
                        "task_ids": [],
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "quality_message": quality_message,
                        "priority": job.get("priority"),
                        "probe_status": job.get("probe_status"),
                        "toy_metric": job.get("toy_metric"),
                    }
                )
                processed.add(job_key)
                return
            write_job_status(
                job_path,
                "running",
                {
                    "required_task_ids": list(required_ids),
                    "quality_scope": scope,
                    "evaluated_task_ids": list(job_ids),
                },
            )
            append_journal(
                journal,
                {
                    "event": "quality_candidate_started",
                    "candidate": candidate_id,
                    "candidate_revision_id": job.get("candidate_revision_id")
                    or candidate_id,
                    "proposal_id": job.get("proposal_id") or candidate_id,
                    "patch_revision": job.get("patch_revision"),
                    "patch_sha256": job.get("patch_sha256"),
                    "title": job.get("title") or "",
                    "issue_url": job.get("issue_url") or "",
                    "source_revision": job.get("source_revision") or "",
                    "registry_identity": registry.identity,
                    "n_jobs_per_side": proposal_n_jobs,
                    "cpu_quota": proposal_cpu_quota,
                    "dataset_pair_slots": (
                        pair_scheduler.max_pairs
                        if pair_scheduler is not None
                        else max(1, proposal_cpu_quota // proposal_n_jobs)
                    ),
                    "resource_budget_scope": "campaign-shared",
                    "hypothesis": job.get("hypothesis") or {},
                    "quality_scope": scope,
                    "task_ids": list(job_ids),
                },
            )
            tree: Path | None = None
            try:
                if not patch_path.is_file():
                    reason = "missing_patch"
                    quality_message = quality_result_message(
                        {
                            "candidate_id": candidate_id,
                            "status": "apply_failed",
                            "reason": reason,
                            "task_ids": list(job_ids),
                            "hypothesis": job.get("hypothesis") or {},
                        }
                    )
                    write_job_status(
                        job_path,
                        "apply_failed",
                        {
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "unresolved_task_ids": list(required_ids),
                            "quality_message": quality_message,
                        },
                    )
                    append_journal(
                        journal,
                        {
                            "event": "quality_disposition",
                            "candidate": candidate_id,
                            "registry_identity": registry.identity,
                            "status": "apply_failed",
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "task_ids": list(required_ids),
                            "evaluated_task_ids": [],
                            "unresolved_task_ids": list(required_ids),
                            "quality_message": quality_message,
                        },
                    )
                    results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "apply_failed",
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "unresolved_task_ids": list(required_ids),
                            "task_ids": list(job_ids),
                            "quality_message": quality_message,
                        }
                    )
                    return
                expected_patch_hash = str(job.get("patch_sha256") or "")
                if (
                    expected_patch_hash
                    and hashlib.sha256(patch_path.read_bytes()).hexdigest()
                    != expected_patch_hash
                ):
                    item = {
                        "candidate_id": candidate_id,
                        "status": "apply_failed",
                        "reason": "patch_hash_mismatch",
                        "keep": None,
                        "quality_outcome": "not_evaluated",
                        "task_ids": list(required_ids),
                    }
                    write_job_status(job_path, "apply_failed", item)
                    append_journal(
                        journal,
                        {
                            **item,
                            "event": "quality_disposition",
                            "candidate": candidate_id,
                        },
                    )
                    results.append(item)
                    return
                tree = create_experiment_checkout(
                    source,
                    workspace,
                    run_id="quality-drain",
                    candidate_id=candidate_id,
                )
                applied = apply_queue_patch(
                    tree,
                    candidate_id=candidate_id,
                    file_path=str(job.get("file_path") or ""),
                    patch_text=patch_path.read_text(encoding="utf-8"),
                )
                if not applied:
                    reason = "patch_rejected"
                    quality_message = quality_result_message(
                        {
                            "candidate_id": candidate_id,
                            "status": "apply_failed",
                            "reason": reason,
                            "task_ids": list(job_ids),
                            "hypothesis": job.get("hypothesis") or {},
                        }
                    )
                    write_job_status(
                        job_path,
                        "apply_failed",
                        {
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "unresolved_task_ids": list(required_ids),
                            "quality_message": quality_message,
                        },
                    )
                    append_journal(
                        journal,
                        {
                            "event": "quality_disposition",
                            "candidate": candidate_id,
                            "registry_identity": registry.identity,
                            "status": "apply_failed",
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "task_ids": list(required_ids),
                            "evaluated_task_ids": [],
                            "unresolved_task_ids": list(required_ids),
                            "quality_message": quality_message,
                        },
                    )
                    results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "apply_failed",
                            "reason": reason,
                            "keep": None,
                            "quality_outcome": "not_evaluated",
                            "unresolved_task_ids": list(required_ids),
                            "task_ids": list(job_ids),
                            "quality_message": quality_message,
                        }
                    )
                    return
                decision = measure_fedot_quality(
                    source,
                    tree,
                    journal=journal,
                    candidate=PatchCandidate(
                        candidate_id=candidate_id,
                        file_path=str(job.get("file_path") or ""),
                    ),
                    task_ids=job_ids,
                    n_jobs=proposal_n_jobs,
                    cpu_quota=proposal_cpu_quota,
                    hypothesis=job.get("hypothesis") or {},
                    quality_scope=scope,
                    required_task_ids=required_ids,
                    registry=registry,
                    pair_scheduler=pair_scheduler,
                    pair_group_id=pair_group_id,
                    proposal_id=str(job.get("proposal_id") or candidate_id),
                    patch_revision=job.get("patch_revision"),
                    artifact_root=artifact_root,
                    stock_cache_root=stock_cache_root,
                )
                status = (
                    "keep"
                    if decision.keep is True
                    else "drop"
                    if decision.keep is False
                    else "partial"
                )
                if decision.infrastructure_error:
                    status = "infrastructure_error"
                elif decision.quality_failure:
                    status = decision.quality_failure
                elif decision.keep is None and decision.unresolved_task_ids:
                    status = "partial"
                incomplete_scope = set(job_ids) != set(required_ids)
                if (
                    incomplete_scope
                    and not decision.quality_failure
                    and not decision.infrastructure_error
                ):
                    status = "partial"
                    decision.keep = None
                    decision.final_keep = None
                    decision.quality_outcome = "not_evaluated"
                    decision.reason = (
                        "partial_dataset_check; full frozen scope remains required"
                    )
                    decision.unresolved_task_ids = tuple(
                        dict.fromkeys(
                            (
                                *decision.unresolved_task_ids,
                                *(item for item in required_ids if item not in job_ids),
                            )
                        )
                    )
                quality_message = quality_result_message(
                    {
                        "candidate_id": candidate_id,
                        "status": status,
                        "reason": decision.reason,
                        "task_ids": list(job_ids),
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_outcome": decision.quality_outcome,
                        "improved_task_count": decision.improved_task_count,
                    }
                )
                write_job_status(
                    job_path,
                    status,
                    {
                        "reason": decision.reason,
                        "keep": None if status == "partial" else decision.keep,
                        "target_delta": decision.target_delta,
                        "regression_deltas": decision.regression_deltas,
                        "task_deltas": decision.regression_deltas,
                        "quality_outcome": decision.quality_outcome,
                        "improved_task_count": decision.improved_task_count,
                        "regressed_task_count": decision.regressed_task_count,
                        "unresolved_task_ids": list(decision.unresolved_task_ids),
                        "infrastructure_error": decision.infrastructure_error,
                        "quality_failure": decision.quality_failure,
                        "task_ids": list(required_ids),
                        "evaluated_task_ids": list(job_ids),
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "quality_message": quality_message,
                    },
                )
                append_journal(
                    journal,
                    {
                        "event": "quality_disposition",
                        "candidate": candidate_id,
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "task_ids": list(job_ids),
                        "status": status,
                        "reason": decision.reason,
                        "keep": decision.keep,
                        "infrastructure_error": decision.infrastructure_error,
                        "quality_failure": decision.quality_failure,
                        "quality_outcome": decision.quality_outcome,
                        "task_deltas": decision.regression_deltas,
                        "improved_task_count": decision.improved_task_count,
                        "regressed_task_count": decision.regressed_task_count,
                        "unresolved_task_ids": list(decision.unresolved_task_ids),
                        "quality_message": quality_message,
                    },
                )
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": status,
                        "reason": decision.reason,
                        "keep": None if status == "partial" else decision.keep,
                        "target_delta": decision.target_delta,
                        "task_deltas": decision.regression_deltas,
                        "infrastructure_error": decision.infrastructure_error,
                        "quality_failure": decision.quality_failure,
                        "quality_outcome": decision.quality_outcome,
                        "improved_task_count": decision.improved_task_count,
                        "regressed_task_count": decision.regressed_task_count,
                        "unresolved_task_ids": list(decision.unresolved_task_ids),
                        "task_ids": list(job_ids),
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "quality_message": quality_message,
                        "priority": job.get("priority"),
                        "probe_status": job.get("probe_status"),
                        "toy_metric": job.get("toy_metric"),
                    }
                )
                if decision.infrastructure_error and not continue_on_error:
                    paused_for_retry.set()
            except (OSError, JournalCorruptionError) as exc:
                # Storage/launch failures are systemic: starting the next
                # improvement would risk unrecorded measurements.
                if isinstance(exc, shutil.Error) and storage_failure_text(str(exc)):
                    entries = (
                        exc.args[0]
                        if exc.args and isinstance(exc.args[0], list)
                        else []
                    )
                    destinations = [
                        str(entry[1])
                        for entry in entries
                        if isinstance(entry, (tuple, list))
                        and len(entry) == 3
                        and storage_failure_text(str(entry[2]))
                    ]
                    destination = (
                        destinations[0]
                        if destinations
                        else str(
                            tree
                            or workspace / "experiments/quality-drain" / candidate_id
                        )
                    )
                    raise StorageCapacityError(
                        28,
                        f"checkout storage failure for proposal {candidate_id}: {exc}",
                        destination,
                    ) from exc
                raise
            except Exception as exc:
                if pair_scheduler.fatal_error is exc:
                    # Shared journal/worker orchestration failure: keep the
                    # remaining proposals queued for a safe resume instead of
                    # converting a campaign-wide failure into per-proposal
                    # verdicts.
                    raise
                reason = f"executor_error:{type(exc).__name__}"
                quality_message = quality_result_message(
                    {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "reason": reason,
                        "task_ids": list(job_ids),
                        "hypothesis": job.get("hypothesis") or {},
                    }
                )
                write_job_status(
                    job_path,
                    "failed",
                    {
                        "reason": reason,
                        "keep": None,
                        "infrastructure_error": True,
                        "quality_outcome": "not_evaluated",
                        "unresolved_task_ids": list(required_ids),
                        "quality_message": quality_message,
                    },
                )
                append_journal(
                    journal,
                    {
                        "event": "quality_disposition",
                        "candidate": candidate_id,
                        "registry_identity": registry.identity,
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "task_ids": list(required_ids),
                        "evaluated_task_ids": list(job_ids),
                        "status": "failed",
                        "reason": reason,
                        "keep": None,
                        "quality_outcome": "not_evaluated",
                        "unresolved_task_ids": list(required_ids),
                        "infrastructure_error": True,
                        "quality_message": quality_message,
                    },
                )
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "reason": reason,
                        "keep": None,
                        "infrastructure_error": True,
                        "quality_outcome": "not_evaluated",
                        "unresolved_task_ids": list(required_ids),
                        "task_ids": list(job_ids),
                        "hypothesis": job.get("hypothesis") or {},
                        "quality_scope": scope,
                        "quality_message": quality_message,
                        "priority": job.get("priority"),
                        "probe_status": job.get("probe_status"),
                        "toy_metric": job.get("toy_metric"),
                    }
                )
            finally:
                original_error = sys.exc_info()[1]
                if tree is not None:
                    try:
                        if original_error is not None and pair_scheduler is not None:
                            idle_timeout = float(registry.watchdog_seconds) + 30.0
                            if not pair_scheduler.wait_for_compute_idle(
                                timeout=idle_timeout
                            ):
                                raise TimeoutError(
                                    "quality workers did not exit before checkout cleanup; "
                                    f"preserving proposal checkout after {idle_timeout:g}s"
                                )
                        discard_experiment_checkout(
                            tree, workspace=workspace, source=source
                        )
                    except Exception as cleanup_error:
                        if original_error is None:
                            raise
                        if hasattr(original_error, "add_note"):
                            original_error.add_note(
                                f"experiment cleanup failed at {tree}: {cleanup_error}"
                            )
                try:
                    payload = json.loads(job_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # Never let a broken job file in ``finally`` mask the real error.
                    payload = {}
                status = str(payload.get("status") or "")
                if status == "running":
                    try:
                        write_job_status(
                            job_path,
                            "queued",
                            {
                                "reason": "executor_interrupted; resumable from quality journal",
                                "keep": None,
                            },
                        )
                    except OSError as persist_error:
                        if original_error is None:
                            raise
                        if hasattr(original_error, "add_note"):
                            original_error.add_note(
                                f"queue recovery status could not be saved at {job_path}: {persist_error}"
                            )
                    status = "queued"
                elif status not in {"queued", "running", ""}:
                    processed.add(job_key)
                if status == "partial":
                    processed.add(job_key)
                if status == "infrastructure_error" and not continue_on_error:
                    paused_for_retry.set()

        def process_job(job, proposal_cpu_quota):
            group_id = pair_scheduler.register_group()
            try:
                return _process_job(job, proposal_cpu_quota, group_id)
            except BaseException as exc:
                # Storage/journal/orchestration failures must not open the
                # startup barrier and let sibling proposals begin more work.
                pair_scheduler.fail_closed(exc)
                raise
            finally:
                # A proposal can be blocked or fail before its pairs
                # are submitted. It must still release the initial fair-start gate.
                pair_scheduler.skip_group(group_id)

        proposal_n_jobs = int(n_jobs if n_jobs is not None else registry.n_jobs_per_job)
        total_cpu_quota = int(
            cpu_quota if cpu_quota is not None else registry.cpu_quota
        )
        if proposal_n_jobs <= 0 or total_cpu_quota < proposal_n_jobs:
            raise ValueError("cpu_quota cannot fund one n_jobs dataset pair")
        if isinstance(max_pair_slots, bool) or not isinstance(max_pair_slots, int):
            raise ValueError("max_pair_slots must be an integer")
        if max_pair_slots < 1 or max_pair_slots > 8:
            raise ValueError("max_pair_slots must be between 1 and 8")
        pair_slots = min(max_pair_slots, total_cpu_quota // proposal_n_jobs)
        append_journal(
            journal,
            {
                "event": "quality_scheduler_batch_started",
                "registry_identity": registry.identity,
                "candidate_ids": [str(job.get("candidate_id") or "") for job in queued],
                "parallel_proposals": min(len(queued), pair_slots),
                "dataset_pair_slots": pair_slots,
                "cpu_slots": pair_slots * proposal_n_jobs,
                "cpu_quota": total_cpu_quota,
                "n_jobs_per_side": proposal_n_jobs,
                "policy": "fair-shared-pair-pool-v3",
            },
        )
        initial_groups = min(len(queued), pair_slots)
        pair_scheduler = SharedPairScheduler(
            max_pairs=pair_slots, initial_groups=initial_groups
        )
        proposal_executor = ThreadPoolExecutor(max_workers=min(len(queued), pair_slots))
        pending = {}
        next_job = 0
        first_error = None
        try:
            for _ in range(min(len(queued), pair_slots)):
                future = proposal_executor.submit(
                    process_job, queued[next_job], total_cpu_quota
                )
                pending[future] = total_cpu_quota
                next_job += 1
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                completed_count = len(completed)
                for future in completed:
                    pending.pop(future)
                    try:
                        future.result()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                if first_error is None and not paused_for_retry.is_set():
                    for _ in range(completed_count):
                        if next_job >= len(queued):
                            break
                        future = proposal_executor.submit(
                            process_job, queued[next_job], total_cpu_quota
                        )
                        pending[future] = total_cpu_quota
                        next_job += 1
        except BaseException as exc:
            pair_scheduler.fail_closed(exc)
            raise
        finally:
            proposal_executor.shutdown(wait=True, cancel_futures=True)
            pair_scheduler.shutdown(wait=True)
        if first_error is not None:
            raise first_error
        if paused_for_retry.is_set():
            break

    stock_rows: dict[tuple[str, str], dict] = {}
    for event in read_quality_events(journal):
        if (
            event.get("event")
            not in {"quality_task_side_finished", "quality_task_side_reused"}
            or event.get("side") != "stock"
            or event.get("registry_identity") != registry.identity
        ):
            continue
        task_id = str(event.get("task_id") or "")
        candidate_id = str(event.get("candidate") or "")
        result = event.get("result") or {}
        if task_id and isinstance(result, dict):
            stock_rows[(candidate_id, task_id)] = {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "status": result.get("status"),
                "score": result.get("score"),
                "search_ran": bool(
                    (result.get("metric_observations") or {}).get("search_ran")
                ),
                "cache_hit": bool(
                    (result.get("metric_observations") or {}).get("cache_hit")
                ),
                "singleflight_shared": bool(
                    (result.get("metric_observations") or {}).get("singleflight_shared")
                ),
            }
    stock_summary = list(stock_rows.values())
    results.sort(
        key=lambda row: result_order.get(
            str(row.get("candidate_id") or ""), len(result_order)
        )
    )
    return {
        "mode": "drain",
        "registry_identity": registry.identity,
        "task_ids": list(dict.fromkeys(row["task_id"] for row in stock_summary)),
        "stock_ok": bool(stock_summary)
        and all(row["status"] == "ok" for row in stock_summary),
        "stock": stock_summary,
        "stock_cache_hits": [
            f"{row['candidate_id']}:{row['task_id']}"
            for row in stock_summary
            if row["cache_hit"]
        ],
        "stock_shared": [
            f"{row['candidate_id']}:{row['task_id']}"
            for row in stock_summary
            if row["singleflight_shared"]
        ],
        "stock_warmed": [
            f"{row['candidate_id']}:{row['task_id']}"
            for row in stock_summary
            if not row["cache_hit"]
        ],
        "jobs": results,
        "paused_for_retry": paused_for_retry.is_set(),
    }


def drain_quality_queue(
    source: Path,
    workspace: Path,
    *,
    journal: Path | None = None,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    max_pair_slots: int = 7,
    stock_cache: Path | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
    continue_on_error: bool = False,
    registry: QualityRegistry | None = None,
) -> dict:
    """Run or resume one campaign queue under an exclusive workspace lock."""

    with _controller_stack_trace(workspace), quality_queue_lock(workspace):
        return _drain_quality_queue_locked(
            source,
            workspace,
            journal=journal,
            task_ids=task_ids,
            n_jobs=n_jobs,
            cpu_quota=cpu_quota,
            max_pair_slots=max_pair_slots,
            stock_cache=stock_cache,
            artifact_root=artifact_root,
            stock_cache_root=stock_cache_root,
            continue_on_error=continue_on_error,
            registry=registry,
        )
