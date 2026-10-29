"""Paired FEDOT quality jobs with separate fit and total-side budgets."""

from __future__ import annotations

import json
import hashlib
import math
import fcntl
import os
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from queue import SimpleQueue
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.evaluation.quality_registry import (
    QualityDataset,
    QualityRegistry,
    get_dataset,
    job_spec,
    list_quality_task_ids,
    load_quality_registry,
)
from fedotllm.agents.evolve.evaluation.pair_scheduler import SharedPairScheduler
from fedotllm.agents.evolve.evaluation.stock_flights import StockFlightRegistry
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.execution.process import (
    clean_subprocess_env,
    fedot_python,
    interpreter_identity,
    run_worker,
)
from fedotllm.agents.evolve.storage.journal import write_json_atomic
from fedotllm.agents.evolve.types import Decision, ScoreResult

WORKER_MODULE = "fedotllm.agents.evolve.evaluation._fedot_quality_worker"
STOCK_CACHE_ENV = "EVOLVE_QUALITY_STOCK_CACHE"
DEFAULT_STOCK_CACHE = Path("/tmp/evolve-fedot-quality-stock")
SideEventCallback = Callable[[dict, str, str, ScoreResult | None, int], None]
SideWaitCallback = Callable[[dict, str, str, float, int], None]
_stock_flights_lock = threading.Lock()
_stock_flights: dict[str, Future] = {}
_async_stock_flights: StockFlightRegistry[ScoreResult] = StockFlightRegistry()


class StockCacheWaitTimeout(TimeoutError):
    """A side deadline expired while waiting for a shared stock cache."""

    def __init__(self, wait_kind: str, elapsed_seconds: float):
        self.wait_kind = wait_kind
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"{wait_kind} wait exceeded side deadline after {elapsed_seconds:.3f}s"
        )


def worker_artifact_paths(
    task_id: str,
    side: str,
    proposal_id: str | None,
    attempt: int,
    *,
    root: Path | str | None = None,
    worker_invocation_id: str | None = None,
) -> dict[str, str]:
    """Describe physical paths; no nonce is reserved for legacy recovery only."""

    if side not in {"stock", "patch"}:
        raise ValueError("side must be stock or patch")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    root_path = Path(
        root
        if root is not None
        else os.environ.get("EVOLVE_AGENT_TMP", "/tmp/evolve-agent")
    ).expanduser()
    artifact_root = root_path
    if worker_invocation_id is not None:
        if len(worker_invocation_id) != 32 or not all(
            c in "0123456789abcdef" for c in worker_invocation_id
        ):
            raise ValueError(
                "worker_invocation_id must be a 32-character hexadecimal nonce"
            )
        root_path = root_path / worker_invocation_id
    attempt_id = f"{proposal_id or 'quality'}:{task_id}:{side}:{attempt}"
    token = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:10]
    stem = f"quality-out-{task_id}-{side}-{token}"
    paths = {
        "attempt_id": attempt_id,
        "worker_spec_path": str(
            root_path / f"quality-spec-{task_id}-{side}-{token}.json"
        ),
        "worker_result_path": str(root_path / f"{stem}.json"),
        "timeout_manifest_path": str(root_path / f"{stem}.timeout.json"),
        "worker_log_path": str(root_path / f"quality-log-{task_id}-{side}-{token}.log"),
        "stage_status_path": str(root_path / f"{stem}.stages.json"),
        "stack_trace_path": str(root_path / f"{stem}.stacks.txt"),
        "process_trace_dir": str(root_path / f"{stem}.process-traces"),
        "process_trace_manifest_path": str(
            root_path / f"{stem}.process-traces" / "manifest.jsonl"
        ),
    }
    if worker_invocation_id is not None:
        paths.update(
            worker_invocation_id=worker_invocation_id, artifact_root=str(artifact_root)
        )
    if os.environ.get("EVOLVE_QUALITY_TRACE_FEDOT") == "1":
        paths["fedot_trace_dir"] = str(root_path / f"{stem}.fedot-traces")
    return paths


def create_worker_artifact_plan(
    task_id: str,
    side: str,
    proposal_id: str | None,
    attempt: int,
    *,
    artifact_root: Path | str | None = None,
) -> dict[str, str]:
    """Reserve an invocation identity before journaling or scheduling a worker."""

    root = (
        Path(
            artifact_root
            if artifact_root is not None
            else os.environ.get("EVOLVE_AGENT_TMP", "/tmp/evolve-agent")
        )
        .expanduser()
        .resolve()
    )
    return worker_artifact_paths(
        task_id,
        side,
        proposal_id,
        attempt,
        root=root,
        worker_invocation_id=uuid.uuid4().hex,
    )


def _check_cancellation(cancellation_event: threading.Event | None) -> None:
    if cancellation_event is not None and cancellation_event.is_set():
        raise CancelledError("quality evaluation cancelled after sibling failure")


def _stock_wait_timeout_result(
    dataset: QualityDataset,
    *,
    env_hash: str,
    side_started: float,
    side_timeout: float,
    error: StockCacheWaitTimeout,
    wait_elapsed_seconds: float | None = None,
    queue_wait_seconds: float | None = None,
) -> ScoreResult:
    elapsed = max(0.0, time.monotonic() - side_started)
    observations = {
        "search_ran": False,
        "search_detail": error.wait_kind,
        "failure_kind": "timeout",
        "timeout_kind": error.wait_kind,
        "wait_kind": error.wait_kind,
        "elapsed_seconds": elapsed,
        "side_elapsed_seconds": elapsed,
        "timeout_limit_seconds": side_timeout,
        "worker_spawned": False,
    }
    if wait_elapsed_seconds is not None:
        observations["wait_elapsed_seconds"] = max(0.0, wait_elapsed_seconds)
    if queue_wait_seconds is not None:
        observations["shared_stock_queue_wait_seconds"] = max(0.0, queue_wait_seconds)
    return ScoreResult(
        task_id=dataset.task_id,
        status="timeout",
        score=float("nan"),
        detail=str(error),
        duration_s=elapsed,
        env_hash=env_hash,
        metric_observations=observations,
    )


def _retryable_quality_failure(result: ScoreResult) -> bool:
    if result.status == "timeout":
        return False
    if "controller interrupted" in result.detail.lower():
        return True
    text = "\n".join((result.detail, result.traceback, result.log_tail)).lower()
    transient_markers = (
        "connectionreseterror",
        "connectionerror",
        "remotedisconnected",
        "temporary failure in name resolution",
        "name or service not known",
        "readtimeout",
        "service unavailable",
        "http error 429",
        "http error 502",
        "http error 503",
        "http error 504",
        "status code 429",
        "status code 502",
        "status code 503",
        "status code 504",
    )
    return any(marker in text for marker in transient_markers)


def _async_stock_flight_key(
    dataset: QualityDataset,
    checkout: Path,
    *,
    n_jobs: int,
    registry: QualityRegistry,
    python: str,
    stock_cache_root: Path | str | None = None,
) -> str:
    cache_path = _stock_cache_path(
        dataset, registry, n_jobs, stock_cache_root=stock_cache_root
    )
    identity = stock_cache_identity(
        dataset,
        registry,
        fedot_commit=_fedot_commit(checkout),
        source_sha256=_source_sha256(str(checkout.resolve())),
        n_jobs=n_jobs,
        worker_runtime=interpreter_identity(checkout, executable=python),
    )
    identity_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{cache_path.resolve()}|{identity_key}"


def bind_stock_cache(path: Path | str | None) -> Path:
    """Pin local and cluster jobs to the same cache directory."""

    if path is None:
        return _stock_cache_root()
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ[STOCK_CACHE_ENV] = str(resolved)
    return resolved


def _stock_cache_root(root: Path | str | None = None) -> Path:
    return (
        Path(
            root
            if root is not None
            else os.environ.get(STOCK_CACHE_ENV, str(DEFAULT_STOCK_CACHE))
        )
        .expanduser()
        .resolve()
    )


def _safe_cache_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def stock_cache_identity(
    dataset: QualityDataset,
    registry: QualityRegistry,
    *,
    fedot_commit: str = "",
    source_sha256: str = "",
    n_jobs: int | None = None,
    worker_runtime: str | None = None,
) -> dict:
    """Skip-warm key includes the exact registered input asset where available."""

    cv_folds = 3 if dataset.problem == "ts_forecasting" else 5
    candidate_evaluation_seconds = int(registry.fedot_search_budget_seconds) // cv_folds
    fit_seconds_per_fold = (
        candidate_evaluation_seconds // cv_folds
        if registry.fedot_timeout_policy == "return_best"
        else candidate_evaluation_seconds
    )
    return {
        "task_id": dataset.task_id,
        "data_source": str(dataset.source),
        "openml_task": int(dataset.openml_task or 0),
        "openml_data": int(dataset.openml_data or 0),
        "data_path": str(dataset.data_path or ""),
        "target": str(dataset.target or ""),
        "forecast_horizon": int(dataset.forecast_horizon or 0),
        "fold": int(dataset.fold),
        "repeat": int(dataset.repeat),
        "seed": int(registry.seed),
        "timeout_seconds": int(registry.timeout_seconds),
        "fedot_search_budget_seconds": int(registry.fedot_search_budget_seconds),
        "fedot_timeout_policy": str(registry.fedot_timeout_policy),
        "final_fit_reserve_seconds": int(registry.final_fit_reserve_seconds),
        "cv_folds": cv_folds,
        "max_pipeline_fit_time_seconds": fit_seconds_per_fold,
        "max_pipeline_fit_policy": str(registry.max_pipeline_fit_policy),
        "max_pipeline_evaluation_time_seconds": candidate_evaluation_seconds,
        "max_pipeline_evaluation_policy": str(registry.max_pipeline_evaluation_policy),
        "tuning_evaluation_policy": str(registry.tuning_evaluation_policy),
        "composer_options": dict(registry.composer_options),
        "watchdog_seconds": int(registry.watchdog_seconds),
        "preset": str(registry.preset),
        "registry_identity": str(registry.identity),
        "registry_sha256": hashlib.sha256(
            json.dumps(asdict(registry), sort_keys=True).encode()
        ).hexdigest(),
        "fedot_commit": str(fedot_commit or ""),
        "source_sha256": str(source_sha256 or ""),
        "n_jobs": int(n_jobs if n_jobs is not None else registry.n_jobs_per_job),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": f"{platform.system()}-{platform.machine()}",
        "runtime_versions": _runtime_versions(),
        "worker_runtime": worker_runtime
        if worker_runtime is not None
        else interpreter_identity(Path.cwd(), executable=sys.executable),
        "data_sha256": str(dataset.data_sha256 or ""),
    }


def _runtime_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "fedot",
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "giotto-ph",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _fedot_commit(checkout: Path | None) -> str:
    if checkout is None:
        return ""
    try:
        from fedotllm.agents.evolve.execution.checkout import source_commit

        return source_commit(checkout) or ""
    except OSError:
        return ""


@lru_cache(maxsize=16)
def _source_sha256(checkout_path: str) -> str:
    """Fingerprint one immutable FEDOT source tree once per controller process."""

    try:
        from fedotllm.agents.evolve.execution.checkout import source_fingerprint

        return source_fingerprint(Path(checkout_path))
    except OSError:
        return ""


def _stock_cache_path(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    stock_cache_root: Path | str | None = None,
) -> Path:
    name = f"{registry.identity}__{dataset.task_id}__n{n_jobs}__s{registry.seed}.json"
    return _stock_cache_root(stock_cache_root) / _safe_cache_name(name)


@contextmanager
def _stock_cache_singleflight(
    cache_path: Path,
    *,
    cancellation_event: threading.Event | None = None,
    on_waiting: Callable[[str, float], None] | None = None,
    deadline: float | None = None,
):
    """Serialize baseline misses by exact cache key across controller processes."""

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        wait_started = time.monotonic()
        next_report = wait_started
        while True:
            _check_cancellation(cancellation_event)
            if deadline is not None and time.monotonic() >= deadline:
                raise StockCacheWaitTimeout(
                    "stock cache file lock", time.monotonic() - wait_started
                )
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = time.monotonic() - wait_started
                if on_waiting is not None and time.monotonic() >= next_report:
                    on_waiting("stock_cache_file_lock", elapsed)
                    next_report = time.monotonic() + 30.0
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _join_stock_flight(
    cache_path: Path,
    *,
    identity_key: str,
    cancellation_event: threading.Event | None,
    compute: Callable[[], ScoreResult],
    on_waiting: Callable[[str, float], None] | None = None,
    deadline: float | None = None,
) -> ScoreResult:
    """Share one in-process stock outcome, including a timeout, among waiters."""

    key = f"{cache_path.resolve()}|{identity_key}"
    with _stock_flights_lock:
        future = _stock_flights.get(key)
        owner = future is None
        if future is None:
            future = Future()
            _stock_flights[key] = future
    if not owner:
        wait_started = time.monotonic()
        if on_waiting is not None:
            on_waiting("shared_stock_fit", 0.0)
        next_report = wait_started + 30.0
        while True:
            _check_cancellation(cancellation_event)
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise StockCacheWaitTimeout(
                    "shared stock fit", time.monotonic() - wait_started
                )
            try:
                result = future.result(
                    timeout=0.1 if remaining is None else min(0.1, remaining)
                )
                observations = dict(result.metric_observations or {})
                observations["singleflight_shared"] = True
                return replace(result, metric_observations=observations)
            except StockCacheWaitTimeout:
                raise
            except TimeoutError:
                elapsed = time.monotonic() - wait_started
                if on_waiting is not None and time.monotonic() >= next_report:
                    on_waiting("shared_stock_fit", elapsed)
                    next_report = time.monotonic() + 30.0
                continue
            except CancelledError:
                _check_cancellation(cancellation_event)
                with _stock_flights_lock:
                    owner_still_running = _stock_flights.get(key) is future
                if owner_still_running:
                    time.sleep(0.1)
                    continue
                return _join_stock_flight(
                    cache_path,
                    identity_key=identity_key,
                    cancellation_event=cancellation_event,
                    compute=compute,
                    on_waiting=on_waiting,
                    deadline=deadline,
                )
    try:
        result = compute()
        future.set_result(result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _stock_flights_lock:
            if _stock_flights.get(key) is future:
                del _stock_flights[key]


def _iter_stock_cache_paths(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    stock_cache_root: Path | str | None = None,
) -> list[Path]:
    """Return only cache entries from the exact per-task compute profile."""

    exact = _stock_cache_path(
        dataset, registry, n_jobs, stock_cache_root=stock_cache_root
    )
    return [exact] if exact.is_file() else []


def _identity_matches(payload: dict, expected: dict) -> bool:
    identity = (
        payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    )
    blob = {**payload, **identity}
    # Old cache rows without the full identity are history only, never a hit.
    if not identity:
        return False
    return all(blob.get(key) == value for key, value in expected.items())


def _load_stock_cache(
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    env_hash: str,
    fedot_commit: str = "",
    source_sha256: str = "",
    worker_runtime: str | None = None,
    stock_cache_root: Path | str | None = None,
) -> ScoreResult | None:
    expected = stock_cache_identity(
        dataset,
        registry,
        fedot_commit=fedot_commit,
        source_sha256=source_sha256,
        n_jobs=n_jobs,
        worker_runtime=worker_runtime,
    )
    for path in _iter_stock_cache_paths(
        dataset, registry, n_jobs, stock_cache_root=stock_cache_root
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "ok":
            continue
        if not _identity_matches(payload, expected):
            continue
        try:
            result = _score_from_payload(
                dataset.task_id,
                payload,
                env_hash=env_hash,
                cmd=str(payload.get("cmd") or "stock-cache"),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if result.status != "ok":
            continue
        observations = dict(result.metric_observations or {})
        observations["cache_hit"] = True
        observations["cache_path"] = str(path)
        result.metric_observations = observations
        return result
    return None


def _store_stock_cache(
    result: ScoreResult,
    dataset: QualityDataset,
    registry: QualityRegistry,
    n_jobs: int,
    *,
    fedot_commit: str = "",
    source_sha256: str = "",
    worker_runtime: str | None = None,
    stock_cache_root: Path | str | None = None,
) -> None:
    if (
        result.status != "ok"
        or not math.isfinite(float(result.score))
        or (result.metric_observations or {}).get("metric_valid") is not True
    ):
        return
    root = _stock_cache_root(stock_cache_root)
    root.mkdir(parents=True, exist_ok=True)
    identity = stock_cache_identity(
        dataset,
        registry,
        fedot_commit=fedot_commit,
        source_sha256=source_sha256,
        n_jobs=n_jobs,
        worker_runtime=worker_runtime,
    )
    payload = {
        "task_id": result.task_id,
        "status": result.status,
        "score": result.score if math.isfinite(float(result.score)) else None,
        "traceback": result.traceback,
        "detail": result.detail,
        "duration_s": result.duration_s,
        "n_train": result.n_train,
        "seed": result.seed,
        "cmd": result.cmd,
        "log_tail": result.log_tail,
        "metric_observations": result.metric_observations,
        "identity": identity,
        "fold": identity["fold"],
        "repeat": identity["repeat"],
        "timeout_seconds": identity["timeout_seconds"],
        "fedot_search_budget_seconds": identity["fedot_search_budget_seconds"],
        "preset": identity["preset"],
        "registry_identity": identity["registry_identity"],
        "fedot_commit": identity["fedot_commit"],
    }
    write_json_atomic(
        _stock_cache_path(dataset, registry, n_jobs, stock_cache_root=stock_cache_root),
        payload,
    )


def _stage_status_observations(path: Path) -> dict:
    observations = {"stage_status_path": str(path)}
    if not path.is_file():
        return observations
    try:
        stage_status = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stage_status, dict):
            raise ValueError("stage status must be a JSON object")
        for key in (
            "failed_stage",
            "last_stage",
            "active_stage",
            "stage_timings",
            "event",
            "updated_at",
            "stage_started_at",
            "stage_elapsed_seconds",
            "last_worker_output_line",
            "last_worker_output_at",
            "last_optimizer_event_at",
            "last_fedot_log_line",
            "last_fedot_log_at",
            "fedot_phase",
            "fedot_phase_at",
            "fedot_phase_line",
            "stack_trace_path",
            "fedot_trace_dir",
            "proposal_id",
            "attempt_id",
            "task_id",
            "source_dataset",
            "side",
            "attempt",
        ):
            if key in stage_status:
                target = {
                    "event": "stage_event",
                    "updated_at": "stage_updated_at",
                }.get(key, key)
                observations[target] = stage_status[key]
    except (OSError, TypeError, ValueError) as exc:
        observations["stage_status_error"] = f"{type(exc).__name__}: {exc}"
    return observations


def _score_from_payload(
    task_id: str, payload: dict, *, env_hash: str, cmd: str
) -> ScoreResult:
    if not isinstance(payload, dict):
        raise ValueError("worker result must be a JSON object")
    observations = payload.get("metric_observations")
    if observations is None:
        observations = {}
    if not isinstance(observations, dict):
        raise ValueError("metric_observations must be a JSON object")
    observations = dict(observations)
    for key in (
        "failure_kind",
        "failed_stage",
        "last_stage",
        "stage_timings",
        "stage_status_path",
    ):
        if key in payload:
            observations[key] = payload[key]
    status = payload.get("status")
    if status not in {"ok", "crash", "timeout", "invalid"}:
        raise ValueError(f"unknown worker result status: {status}")
    duration = float(payload.get("duration_s") or 0.0)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_s must be finite and nonnegative")
    raw_score = payload.get("score")
    if status == "ok":
        if observations.get("metric_valid") is not True:
            raise ValueError(
                "successful worker result lacks metric_valid=true attestation"
            )
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("successful worker score must be a number")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("successful worker score must be finite")
    else:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = float("nan")
    failure_kind = payload.get("failure_kind")
    if not failure_kind and status == "crash":
        failed_stage = str(payload.get("failed_stage") or "")
        failure_kind = (
            "scoring_error"
            if failed_stage in {"predict", "metrics"}
            else "fedot_fit_error"
            if failed_stage == "fit"
            else "worker_error"
        )
    if failure_kind:
        observations["failure_kind"] = str(failure_kind)
    observations.setdefault("failed_stage", str(payload.get("failed_stage") or ""))
    observations.setdefault("last_stage", str(payload.get("last_stage") or ""))
    if isinstance(payload.get("stage_timings"), dict):
        observations.setdefault("stage_timings", payload["stage_timings"])
    if payload.get("stage_status_path"):
        observations.setdefault("stage_status_path", str(payload["stage_status_path"]))
    return ScoreResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        score=score,
        traceback=str(payload.get("traceback") or ""),
        detail=str(payload.get("detail") or ""),
        duration_s=duration,
        n_train=int(payload.get("n_train") or 0),
        env_hash=env_hash,
        cmd=cmd,
        log_tail=str(payload.get("log_tail") or ""),
        seed=int(payload.get("seed") or 42),
        metric_observations=observations,
    )


def run_quality_side(
    dataset: QualityDataset | str,
    *,
    checkout: Path,
    side: str,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
    python_executable: str | None = None,
    cancellation_event: threading.Event | None = None,
    proposal_id: str | None = None,
    attempt: int = 1,
    evaluation_identity: dict | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
    worker_paths: dict[str, str] | None = None,
    _stock_lock_held: bool = False,
    _side_deadline: float | None = None,
    _side_started: float | None = None,
    on_worker_spawned: Callable[[int], None] | None = None,
    on_waiting: Callable[[str, float], None] | None = None,
) -> ScoreResult:
    _check_cancellation(cancellation_event)
    registry = registry or load_quality_registry()
    stock_cache_root = _stock_cache_root(stock_cache_root)
    side_timeout = float(registry.timeout_seconds)
    side_started = time.monotonic() if _side_started is None else _side_started
    side_deadline = (
        side_started + side_timeout if _side_deadline is None else _side_deadline
    )
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, registry)
    spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
    spec["side"] = side
    if evaluation_identity is not None:
        spec["evaluation_identity"] = dict(evaluation_identity)
    env_hash = f"{registry.identity}|{dataset.task_id}|{side}|{spec['n_jobs']}"
    fedot_commit = _fedot_commit(checkout)
    source_sha256 = _source_sha256(str(checkout.resolve()))
    python = python_executable or fedot_python(checkout)
    worker_runtime = interpreter_identity(checkout, executable=python)
    cache_path = (
        _stock_cache_path(
            dataset, registry, spec["n_jobs"], stock_cache_root=stock_cache_root
        )
        if side == "stock"
        else None
    )
    if side == "stock" and cache_path is not None and not _stock_lock_held:
        identity_key = hashlib.sha256(
            json.dumps(
                stock_cache_identity(
                    dataset,
                    registry,
                    fedot_commit=fedot_commit,
                    source_sha256=source_sha256,
                    n_jobs=spec["n_jobs"],
                    worker_runtime=worker_runtime,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def compute_stock() -> ScoreResult:
            with _stock_cache_singleflight(
                cache_path,
                cancellation_event=cancellation_event,
                on_waiting=on_waiting,
                deadline=side_deadline,
            ):
                return run_quality_side(
                    dataset,
                    checkout=checkout,
                    side=side,
                    n_jobs=n_jobs,
                    registry=registry,
                    python_executable=python_executable,
                    cancellation_event=cancellation_event,
                    proposal_id=proposal_id,
                    attempt=attempt,
                    evaluation_identity=evaluation_identity,
                    artifact_root=artifact_root,
                    stock_cache_root=stock_cache_root,
                    worker_paths=worker_paths,
                    _stock_lock_held=True,
                    _side_deadline=side_deadline,
                    _side_started=side_started,
                    on_worker_spawned=on_worker_spawned,
                )

        try:
            return _join_stock_flight(
                cache_path,
                identity_key=identity_key,
                cancellation_event=cancellation_event,
                compute=compute_stock,
                on_waiting=on_waiting,
                deadline=side_deadline,
            )
        except StockCacheWaitTimeout as exc:
            return _stock_wait_timeout_result(
                dataset,
                env_hash=env_hash,
                side_started=side_started,
                side_timeout=side_timeout,
                error=exc,
            )
    if side == "stock":
        cached = _load_stock_cache(
            dataset,
            registry,
            spec["n_jobs"],
            env_hash=env_hash,
            fedot_commit=fedot_commit,
            source_sha256=source_sha256,
            worker_runtime=worker_runtime,
            stock_cache_root=stock_cache_root,
        )
        if cached is not None:
            hit_path = (cached.metric_observations or {}).get(
                "cache_path"
            ) or cache_path
            print(f"quality stock cache-hit {dataset.task_id} {hit_path}", flush=True)
            return cached
    paths = (
        dict(worker_paths)
        if worker_paths is not None
        else create_worker_artifact_plan(
            dataset.task_id, side, proposal_id, attempt, artifact_root=artifact_root
        )
    )
    if not paths.get("worker_invocation_id"):
        raise ValueError("new workers require a worker_invocation_id")
    tmp = Path(paths["worker_spec_path"]).parent
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()  # A physical invocation is never allowed to reuse a directory.
    from fedotllm.agents.evolve.execution.storage_guard import require_storage

    require_storage(tmp)
    spec_path = Path(paths["worker_spec_path"])
    out_path = Path(paths["worker_result_path"])
    log_path = Path(paths["worker_log_path"])
    spec.update(
        {
            "proposal_id": proposal_id,
            "worker_invocation_id": paths["worker_invocation_id"],
            "attempt": attempt,
            "attempt_id": f"{proposal_id or 'quality'}:{dataset.task_id}:{side}:{attempt}",
            "registry_identity": registry.identity,
            "worker_log_path": str(log_path),
            "stack_trace_path": paths["stack_trace_path"],
        }
    )
    fedot_trace_interval = None
    if os.environ.get("EVOLVE_QUALITY_TRACE_FEDOT") == "1":
        fedot_trace_interval = float(
            os.environ.get("EVOLVE_QUALITY_TRACE_FEDOT_INTERVAL_SECONDS", "15")
        )
        if not math.isfinite(fedot_trace_interval) or fedot_trace_interval < 0.05:
            raise ValueError(
                "FEDOT trace interval must be finite and at least 0.05 seconds"
            )
        spec.update(
            {
                "fedot_trace_dir": paths["fedot_trace_dir"],
                "fedot_trace_interval_seconds": fedot_trace_interval,
            }
        )
    if os.environ.get("EVOLVE_QUALITY_TRACE_PROCESSES") == "1":
        interval = float(os.environ.get("EVOLVE_QUALITY_TRACE_INTERVAL_SECONDS", "15"))
        if not math.isfinite(interval) or interval < 0.05:
            raise ValueError(
                "diagnostic stack interval must be finite and at least 0.05 seconds"
            )
        spec.update(
            {
                "process_trace_dir": paths["process_trace_dir"],
                "process_trace_manifest_path": paths["process_trace_manifest_path"],
                "process_trace_interval_seconds": interval,
                "stack_dump_interval_seconds": interval,
            }
        )
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    cmd = [
        python,
        "-m",
        WORKER_MODULE,
        "--spec-json",
        str(spec_path),
        "--out",
        str(out_path),
    ]
    extra = {
        key: os.environ[key]
        for key in (
            "EVOLVE_QUALITY_STOCK_CACHE",
            "EVOLVE_QUALITY_DATA_CACHE",
            "EVOLVE_OPENML_CACHE",
            "EVOLVE_QUALITY_DATA_OFFLINE",
            "SKLEARN_DATA",
            "EVOLVE_AGENT_TMP",
            "EVOLVE_TRACE_PYTHON_LINES",
            "EVOLVE_TRACE_LINE_INTERVAL_SECONDS",
            "FEDOTLLM_REPO_CACHE",
            "FEDOT_TRACE_OPERATIONS",
            "FEDOT_TRACE_OPERATIONS_SLOW_SECONDS",
            "FEDOT_TRACE_OPERATIONS_ALL",
            "FEDOT_TRACE_OPERATIONS_MAX_EVENTS",
            "FEDOT_TRACE_RIPSER",
            "FEDOT_TRACE_RIPSER_SLOW_SECONDS",
            "FEDOT_TRACE_RIPSER_ALL",
            "FEDOT_TRACE_RIPSER_MAX_EVENTS",
        )
        if os.environ.get(key)
    }
    if fedot_trace_interval is not None:
        extra.update(
            {
                "FEDOT_TRACE_DIR": paths["fedot_trace_dir"],
                "FEDOT_TRACE_RUN_ID": paths["attempt_id"],
                "FEDOT_TRACE_INTERVAL_SECONDS": str(fedot_trace_interval),
            }
        )
    extra["EVOLVE_AGENT_TMP"] = str(tmp)
    extra[STOCK_CACHE_ENV] = str(stock_cache_root)
    extra["FEDOT_EVOLVE_JOB_ID"] = paths["attempt_id"]
    extra["FEDOTLLM_REPO_PATH"] = str(checkout.resolve())
    env = clean_subprocess_env(checkout, repo_root=repo_root(), extra=extra)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    task_timeout = side_timeout
    watchdog_timeout = float(registry.watchdog_seconds)
    print(
        f"quality {side} start {dataset.task_id} "
        f"side_timeout_s={spec['timeout_seconds']} "
        f"fedot_search_budget_s={spec['fedot_search_budget_seconds']} "
        f"watchdog_s={registry.watchdog_seconds} "
        f"n_jobs={spec['n_jobs']} "
        f"preset={spec['preset']} seed={spec['seed']} "
        f"cache={cache_path} log={log_path}",
        flush=True,
    )
    physical_observations = {
        "worker_paths": dict(paths),
        "worker_invocation_id": paths["worker_invocation_id"],
    }

    def check_storage():
        try:
            require_storage(tmp)
        except OSError as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"dataset={dataset.task_id}, side={side}, worker_log={log_path}, "
                    f"stage_status={out_path.with_suffix('.stages.json')}"
                )
            raise

    _check_cancellation(cancellation_event)
    options = (
        {"cancellation_event": cancellation_event}
        if cancellation_event is not None
        else {}
    )
    options["resource_check"] = check_storage
    if on_worker_spawned is not None:
        options["on_spawned"] = on_worker_spawned
    worker_timeout = max(0.0, side_deadline - time.monotonic())
    if worker_timeout <= 0:
        return _stock_wait_timeout_result(
            dataset,
            env_hash=env_hash,
            side_started=side_started,
            side_timeout=side_timeout,
            error=StockCacheWaitTimeout("side setup", time.monotonic() - side_started),
        )
    worker_started = time.monotonic()
    print(
        f"quality {side} worker-deadline {dataset.task_id} "
        f"side_elapsed_s={worker_started - side_started:.3f} "
        f"worker_timeout_s={worker_timeout:.3f} side_timeout_s={task_timeout:g}",
        flush=True,
    )
    proc = run_worker(
        cmd,
        cwd=repo_root(),
        env=env,
        timeout=worker_timeout,
        watchdog_timeout=watchdog_timeout,
        log_path=log_path,
        **options,
    )
    elapsed_seconds = max(0.0, time.monotonic() - worker_started)
    if proc.cancelled:
        raise CancelledError("quality worker cancelled after sibling failure")
    _check_cancellation(cancellation_event)
    log_text = proc.stdout
    from fedotllm.agents.evolve.execution.storage_guard import (
        StorageCapacityError,
        storage_failure_text,
    )

    if storage_failure_text(log_text):
        raise StorageCapacityError(
            28,
            f"worker storage failure: dataset={dataset.task_id}, "
            f"side={side}, log={log_path}",
            str(tmp),
        )
    if proc.timed_out:
        timeout_kind = proc.timeout_kind or "task_timeout"
        timeout_limit = (
            task_timeout if timeout_kind == "task_timeout" else watchdog_timeout
        )
        side_elapsed = max(0.0, time.monotonic() - side_started)
        stage_path = out_path.with_suffix(".stages.json")
        stage_observations = _stage_status_observations(stage_path)
        timeout_manifest_path = out_path.with_suffix(".timeout.json")
        timeout_observations = {
            **physical_observations,
            "search_ran": False,
            "search_detail": timeout_kind,
            "worker_log_path": str(log_path),
            "worker_result_path": str(out_path),
            "timeout_manifest_path": str(timeout_manifest_path),
            "metric_outcome": "not_measured",
            "metric_valid": False,
            "failure_kind": "timeout",
            "timeout_kind": timeout_kind,
            "elapsed_seconds": side_elapsed,
            "side_elapsed_seconds": side_elapsed,
            "worker_elapsed_seconds": elapsed_seconds,
            "worker_timeout_budget_seconds": worker_timeout,
            "timeout_limit_seconds": timeout_limit,
            "watchdog_limit_seconds": watchdog_timeout,
            "stack_trace_path": stage_observations.get("stack_trace_path")
            or spec.get("stack_trace_path"),
            "fedot_trace_dir": stage_observations.get("fedot_trace_dir")
            or spec.get("fedot_trace_dir"),
            "stack_trace_artifact_state": (
                "present"
                if Path(spec.get("stack_trace_path", "")).is_file()
                else "not_written"
            ),
            "timeout_seconds": registry.timeout_seconds,
            "watchdog_seconds": registry.watchdog_seconds,
            "worker_return_code": proc.returncode,
            **stage_observations,
        }
        timeout_manifest = {
            "schema_version": 1,
            "event": "quality_worker_parent_timeout",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "task_id": dataset.task_id,
            "source_dataset": spec.get("source_dataset"),
            "side": side,
            "attempt": spec.get("attempt"),
            "attempt_id": spec.get("attempt_id"),
            "worker_invocation_id": paths["worker_invocation_id"],
            "proposal_id": spec.get("proposal_id"),
            "evaluation_identity": spec.get("evaluation_identity"),
            "registry_identity": spec.get("registry_identity"),
            "status": "timeout",
            "failure_kind": "timeout",
            "metric_outcome": "not_measured",
            "metric_valid": False,
            "timeout_kind": timeout_kind,
            "detail": (
                f"{timeout_kind} after {side_elapsed:.3f}s "
                f"(side limit {timeout_limit:g}s)"
            ),
            "side_timeout_seconds": task_timeout,
            "worker_timeout_budget_seconds": worker_timeout,
            "watchdog_timeout_seconds": watchdog_timeout,
            "worker_elapsed_seconds": elapsed_seconds,
            "side_elapsed_seconds": side_elapsed,
            "worker_return_code": proc.returncode,
            "worker_spec_path": str(spec_path),
            "worker_result_path": str(out_path),
            "worker_log_path": str(log_path),
            "stage_status_path": str(stage_path),
            "active_stage": stage_observations.get("active_stage"),
            "last_stage": stage_observations.get("last_stage"),
            "stage_updated_at": stage_observations.get("stage_updated_at"),
            "last_worker_output_line": stage_observations.get(
                "last_worker_output_line"
            ),
            "stack_trace_path": timeout_observations.get("stack_trace_path"),
            "fedot_trace_dir": timeout_observations.get("fedot_trace_dir"),
            "log_tail": log_text[-8000:],
            "stage_observations": stage_observations,
        }
        # The parent owns this terminal record: the worker may have been
        # killed before it could publish its normal result JSON.
        try:
            write_json_atomic(timeout_manifest_path, timeout_manifest)
        except OSError as exc:
            timeout_observations["timeout_manifest_write_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        return ScoreResult(
            task_id=dataset.task_id,
            status="timeout",
            score=float("nan"),
            detail=f"{timeout_kind} after {side_elapsed:.3f}s (side limit {timeout_limit:g}s)",
            duration_s=side_elapsed,
            env_hash=env_hash,
            cmd=" ".join(cmd),
            seed=registry.seed,
            log_tail=log_text[-4000:],
            metric_observations=timeout_observations,
        )
    if not out_path.is_file():
        return ScoreResult(
            task_id=dataset.task_id,
            status="invalid",
            score=float("nan"),
            duration_s=elapsed_seconds,
            traceback=log_text[-4000:],
            detail=f"worker exit {proc.returncode}",
            env_hash=env_hash,
            cmd=" ".join(cmd),
            log_tail=log_text[-4000:],
            seed=registry.seed,
            metric_observations={
                **physical_observations,
                "search_ran": False,
                "worker_log_path": str(log_path),
                "worker_result_path": str(out_path),
                "stack_trace_path": paths["stack_trace_path"],
                "failure_kind": "worker_exit",
                "elapsed_seconds": elapsed_seconds,
                "worker_return_code": proc.returncode,
                **_stage_status_observations(out_path.with_suffix(".stages.json")),
            },
        )
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("worker result must be a JSON object")
        expected_identity = {
            "task_id": dataset.task_id,
            "side": side,
            "attempt": spec["attempt"],
            "attempt_id": spec["attempt_id"],
            "registry_identity": spec["registry_identity"],
            "evaluation_identity": spec.get("evaluation_identity"),
            "worker_invocation_id": paths["worker_invocation_id"],
        }
        mismatched_fields = [
            key
            for key, expected in expected_identity.items()
            if payload.get(key) != expected
            or (key == "attempt" and type(payload.get(key)) is not int)
        ]
        if mismatched_fields:
            observations = payload.get("metric_observations")
            observations = dict(observations) if isinstance(observations, dict) else {}
            observations.update(
                {
                    "metric_valid": False,
                    "failure_kind": "worker_identity_mismatch",
                    "identity_mismatch_fields": mismatched_fields,
                }
            )
            payload = dict(
                status="invalid",
                score=None,
                detail=(
                    "worker result identity mismatch: " + ", ".join(mismatched_fields)
                ),
                failure_kind="worker_identity_mismatch",
                metric_observations=observations,
            )

        if payload.get("failure_kind") == "storage_error" or storage_failure_text(
            "\n".join(
                str(payload.get(key) or "")
                for key in ("detail", "traceback", "log_tail")
            )
        ):
            raise StorageCapacityError(
                28,
                f"structured worker storage failure: "
                f"dataset={dataset.task_id}, side={side}, log={log_path}, result={out_path}",
                str(tmp),
            )
        if proc.returncode != 0 and payload.get("status") == "ok":
            # A worker that failed its process lifecycle cannot publish a
            # trusted metric, even if it wrote a plausible result first.
            payload.update(
                status="invalid", score=None, detail=f"worker exit {proc.returncode}"
            )
        payload["log_tail"] = (str(payload.get("log_tail") or "") + "\n" + log_text)[
            -2000:
        ]
        result = _score_from_payload(
            dataset.task_id, payload, env_hash=env_hash, cmd=" ".join(cmd)
        )
        # Keep the structured outcome until the parent has durably recorded
        # the finished-side event. If the parent is interrupted in between,
        # recovery still has the worker's complete result file to inspect.
    except (TypeError, ValueError, OverflowError) as exc:
        return ScoreResult(
            task_id=dataset.task_id,
            status="invalid",
            score=float("nan"),
            duration_s=elapsed_seconds,
            detail=f"invalid worker result at {out_path}: {type(exc).__name__}: {exc}",
            env_hash=env_hash,
            cmd=" ".join(cmd),
            seed=registry.seed,
            traceback=log_text[-4000:],
            log_tail=log_text[-4000:],
            metric_observations={
                **physical_observations,
                "search_ran": False,
                "worker_log_path": str(log_path),
                "worker_result_path": str(out_path),
                "failure_kind": "invalid_worker_result",
                "elapsed_seconds": elapsed_seconds,
                "worker_return_code": proc.returncode,
                **_stage_status_observations(out_path.with_suffix(".stages.json")),
            },
        )
    result.metric_observations["worker_paths"] = dict(paths)
    result.metric_observations["worker_invocation_id"] = paths["worker_invocation_id"]
    result.metric_observations["worker_log_path"] = str(log_path)
    result.metric_observations["worker_result_path"] = str(out_path)
    result.metric_observations["worker_return_code"] = proc.returncode
    result.duration_s = elapsed_seconds
    result.metric_observations["elapsed_seconds"] = elapsed_seconds
    result.metric_observations["worker_elapsed_seconds"] = elapsed_seconds
    result.metric_observations["side_elapsed_seconds"] = max(
        0.0, time.monotonic() - side_started
    )
    result.metric_observations["side_timeout_seconds"] = side_timeout
    result.metric_observations.update(
        _stage_status_observations(out_path.with_suffix(".stages.json"))
    )
    if side == "stock":
        _store_stock_cache(
            result,
            dataset,
            registry,
            spec["n_jobs"],
            fedot_commit=fedot_commit,
            source_sha256=source_sha256,
            worker_runtime=worker_runtime,
            stock_cache_root=stock_cache_root,
        )
    return result


def run_quality_pair(
    dataset: QualityDataset | str,
    *,
    stock_checkout: Path,
    patch_checkout: Path,
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
    on_side_event: SideEventCallback | None = None,
    on_side_wait: SideWaitCallback | None = None,
    prior_side_results: dict[tuple[str, str], ScoreResult] | None = None,
    prior_attempt_counts: dict[tuple[str, str], int] | None = None,
    cancellation_event: threading.Event | None = None,
    proposal_id: str | None = None,
    evaluation_identity: dict | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
) -> dict:
    """One indivisible stock-then-patch job on the same host."""

    stock_cache_root = _stock_cache_root(stock_cache_root)

    _check_cancellation(cancellation_event)
    registry = registry or load_quality_registry()
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, registry)
    spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
    prior_side_results = prior_side_results or {}
    prior_attempt_counts = prior_attempt_counts or {}
    python = fedot_python(stock_checkout)

    def evaluate_side(side: str, checkout: Path) -> ScoreResult:
        side_spec = dict(spec)
        _check_cancellation(cancellation_event)
        key = (dataset.task_id, side)
        previous = prior_side_results.get(key)
        attempt_count = int(prior_attempt_counts.get(key, 0))
        if previous is not None and (
            not _retryable_quality_failure(previous) or attempt_count >= 2
        ):
            original_paths = (previous.metric_observations or {}).get("worker_paths")
            if isinstance(original_paths, dict):
                side_spec["worker_paths"] = dict(original_paths)
            if on_side_event is not None:
                on_side_event(side_spec, side, "reused", previous, attempt_count or 1)
            return previous
        attempt = attempt_count + 1
        while True:
            _check_cancellation(cancellation_event)
            side_spec = dict(spec)
            side_spec["worker_paths"] = create_worker_artifact_plan(
                dataset.task_id, side, proposal_id, attempt, artifact_root=artifact_root
            )
            if on_side_event is not None:
                on_side_event(side_spec, side, "started", None, attempt)

            def record_worker_spawned(worker_pid: int) -> None:
                side_spec["worker_pid"] = worker_pid
                if on_side_event is not None:
                    on_side_event(side_spec, side, "spawned", None, attempt)

            result = run_quality_side(
                dataset,
                checkout=checkout,
                side=side,
                n_jobs=n_jobs,
                registry=registry,
                python_executable=python,
                cancellation_event=cancellation_event,
                proposal_id=proposal_id,
                attempt=attempt,
                evaluation_identity=evaluation_identity,
                artifact_root=artifact_root,
                stock_cache_root=stock_cache_root,
                worker_paths=side_spec["worker_paths"],
                on_worker_spawned=record_worker_spawned,
                on_waiting=(
                    lambda wait_kind, elapsed: on_side_wait(
                        side_spec, side, wait_kind, elapsed, attempt
                    )
                    if on_side_wait is not None
                    else None
                ),
            )
            if on_side_event is not None:
                on_side_event(side_spec, side, "finished", result, attempt)
            if attempt >= 2 or not _retryable_quality_failure(result):
                return result
            _check_cancellation(cancellation_event)
            attempt += 1

    stock = evaluate_side("stock", stock_checkout)
    _check_cancellation(cancellation_event)
    patched = evaluate_side("patch", patch_checkout)
    return {
        "spec": spec,
        "stock": asdict(stock),
        "patched": asdict(patched),
        "stock_result": stock,
        "patched_result": patched,
        "search_ran": bool(
            (stock.metric_observations or {}).get("search_ran")
            and (patched.metric_observations or {}).get("search_ran")
        ),
    }


def _run_quality_pair_async(
    dataset: QualityDataset | str,
    *,
    stock_checkout: Path,
    patch_checkout: Path,
    submit_compute: Callable[..., Future[ScoreResult]],
    n_jobs: int | None = None,
    registry: QualityRegistry | None = None,
    on_side_event: SideEventCallback | None = None,
    on_side_wait: SideWaitCallback | None = None,
    prior_side_results: dict[tuple[str, str], ScoreResult] | None = None,
    prior_attempt_counts: dict[tuple[str, str], int] | None = None,
    cancellation_event: threading.Event | None = None,
    proposal_id: str | None = None,
    evaluation_identity: dict | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
) -> Future[dict]:
    """Schedule a pair without retaining a pair-scheduler slot while it runs.

    The returned Future represents the stock-then-patch result. Actual stock
    and patch calls are submitted to the scheduler's bounded compute pool;
    stock subscribers only attach callbacks and therefore consume no worker.
    """

    stock_cache_root = _stock_cache_root(stock_cache_root)

    result: Future[dict] = Future()
    try:
        _check_cancellation(cancellation_event)
        registry = registry or load_quality_registry()
        if isinstance(dataset, str):
            dataset = get_dataset(dataset, registry)
        spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
        previous_results = prior_side_results or {}
        attempt_counts = prior_attempt_counts or {}
        python = fedot_python(stock_checkout)
    except BaseException as exc:
        result.set_exception(exc)
        return result

    timers: set[threading.Timer] = set()
    active_side_views: set[Future] = set()
    state_lock = threading.RLock()

    def cancel_children(done: Future) -> None:
        if not done.cancelled():
            return
        if cancellation_event is not None:
            cancellation_event.set()
        with state_lock:
            for view in tuple(active_side_views):
                view.cancel()
            for timer in tuple(timers):
                timer.cancel()

    result.add_done_callback(cancel_children)

    def evaluate_side(side: str, checkout: Path, attempt: int) -> Future[ScoreResult]:
        side_spec = dict(spec)
        key = (dataset.task_id, side)
        attempt_count = int(attempt_counts.get(key, 0))
        prior = previous_results.get(key)
        if prior is not None and (
            not _retryable_quality_failure(prior) or attempt_count >= 2
        ):
            original_paths = (prior.metric_observations or {}).get("worker_paths")
            if isinstance(original_paths, dict):
                side_spec["worker_paths"] = dict(original_paths)
            if on_side_event is not None:
                on_side_event(side_spec, side, "reused", prior, attempt_count or 1)
            reused: Future[ScoreResult] = Future()
            reused.set_result(prior)
            return reused

        current_attempt = attempt_count + attempt + 1
        side_spec["worker_paths"] = create_worker_artifact_plan(
            dataset.task_id,
            side,
            proposal_id,
            current_attempt,
            artifact_root=artifact_root,
        )
        stock_flight_key = None
        if (
            side == "stock"
            and _stock_cache_path(
                dataset,
                registry,
                side_spec["n_jobs"],
                stock_cache_root=stock_cache_root,
            )
            is not None
        ):
            stock_flight_key = _async_stock_flight_key(
                dataset,
                checkout,
                n_jobs=side_spec["n_jobs"],
                registry=registry,
                python=python,
                stock_cache_root=stock_cache_root,
            )
            side_spec["stock_flight_id"] = hashlib.sha256(
                stock_flight_key.encode("utf-8")
            ).hexdigest()[:16]
        _check_cancellation(cancellation_event)
        if on_side_event is not None:
            on_side_event(side_spec, side, "started", None, current_attempt)
        side_started = time.monotonic()
        side_timeout = float(registry.timeout_seconds)
        wait_state = {
            "waiting": False,
            "started_at": None,
            "owner_started_at": None,
            "worker_spawned": False,
        }
        wait_report_lock = threading.Lock()
        visible: Future[ScoreResult] = Future()
        with state_lock:
            active_side_views.add(visible)
        wait_heartbeat_timer: threading.Timer | None = None
        terminal_claimed = False

        def publish_result(side_result: ScoreResult) -> None:
            nonlocal terminal_claimed
            # Claim, journal the durable terminal event, and publish its result
            # as one serialized transition so cancellation and completion cannot
            # publish competing terminal states.
            with state_lock:
                if terminal_claimed or visible.done():
                    return
                terminal_claimed = True
                try:
                    if on_side_event is not None:
                        on_side_event(
                            side_spec, side, "finished", side_result, current_attempt
                        )
                    visible.set_result(side_result)
                except BaseException as exc:
                    if not visible.done():
                        visible.set_exception(exc)

        def publish_exception(error: BaseException) -> None:
            nonlocal terminal_claimed
            with state_lock:
                if terminal_claimed or visible.done():
                    return
                terminal_claimed = True
                visible.set_exception(error)

        def cancel_visible() -> None:
            with state_lock:
                if not terminal_claimed and not visible.done():
                    visible.cancel()

        def on_spawned(worker_pid: int) -> None:
            # Owner admission and waiter telemetry share a flight, but only
            # this proposal owns the worker. Keep late wait writes from hiding
            # the owner's spawned event; followers keep their own wait state.
            with wait_report_lock:
                with state_lock:
                    wait_state["worker_spawned"] = True
                    side_spec["worker_pid"] = worker_pid
                if on_side_event is not None:
                    on_side_event(side_spec, side, "spawned", None, current_attempt)

        def wait_started() -> None:
            with state_lock:
                wait_state["waiting"] = True
                wait_state["started_at"] = time.monotonic()
                if wait_state["owner_started_at"] is not None:
                    side_spec["shared_stock_queue_wait_seconds"] = 0.0
            report_wait_state()

        def report_wait_state() -> None:
            if on_side_wait is None:
                return
            # Serialize these small telemetry writes and derive their phase only
            # when emitting: owner admission can race a subscriber's first wait
            # event, but must never leave a newer queue event after fit began.
            with wait_report_lock:
                with state_lock:
                    if not wait_state["waiting"] or wait_state["worker_spawned"]:
                        return
                    owner_started_at = wait_state["owner_started_at"]
                    wait_started_at = wait_state["started_at"]
                    if owner_started_at is None:
                        wait_kind = "scheduler_queue"
                        baseline = wait_started_at
                    else:
                        wait_kind = "shared_stock_fit"
                        baseline = owner_started_at
                    elapsed = max(
                        0.0,
                        time.monotonic() - (baseline or time.monotonic()),
                    )
                on_side_wait(side_spec, side, wait_kind, elapsed, current_attempt)

        def owner_compute_started(started_at: float) -> None:
            report = False
            with state_lock:
                wait_state["owner_started_at"] = started_at
                wait_started_at = wait_state["started_at"]
                if wait_state["waiting"] and wait_started_at is not None:
                    side_spec["shared_stock_queue_wait_seconds"] = max(
                        0.0, started_at - wait_started_at
                    )
                    report = True
            if report:
                report_wait_state()

        def compute(cancel_token=None) -> ScoreResult:
            compute_cancellation = cancel_token or cancellation_event
            _check_cancellation(compute_cancellation)
            # A side can spend an arbitrary amount of time queued behind the
            # shared compute cap. Start its FEDOT budget only after a compute
            # worker actually admits this callback; retain the queue delay as
            # telemetry instead of charging it against training time.
            compute_started = time.monotonic()
            compute_deadline = compute_started + side_timeout
            side_spec["scheduler_queue_wait_seconds"] = max(
                0.0, compute_started - side_started
            )
            return run_quality_side(
                dataset,
                checkout=checkout,
                side=side,
                n_jobs=n_jobs,
                registry=registry,
                python_executable=python,
                cancellation_event=compute_cancellation,
                proposal_id=proposal_id,
                attempt=current_attempt,
                evaluation_identity=evaluation_identity,
                artifact_root=artifact_root,
                stock_cache_root=stock_cache_root,
                worker_paths=side_spec["worker_paths"],
                _side_deadline=compute_deadline,
                _side_started=compute_started,
                on_worker_spawned=on_spawned,
            )

        def submit_side_compute(work):
            return submit_compute(
                work,
                cancellation_event=cancellation_event,
                cooperative_cancellation=True,
            )

        def submit_shared_stock_compute(work):
            # A stock flight belongs to the shared owner, not to the proposal
            # that happened to create it. Proposal cancellation must not abort
            # baseline work while another proposal still waits for that result;
            # scheduler-wide fail-closed cancellation still reaches the worker.
            return submit_compute(
                work,
                cancellation_event=None,
                cooperative_cancellation=True,
            )

        if stock_flight_key is not None:
            shared = _async_stock_flights.get_or_submit(
                stock_flight_key,
                submit=submit_shared_stock_compute,
                compute=compute,
                on_waiting=wait_started,
                on_owner_started=owner_compute_started,
            )
            source = shared
        else:
            try:
                source = submit_side_compute(compute)
            except BaseException as exc:
                source = Future()
                source.set_exception(exc)

        def finish_source(completed: Future[ScoreResult]) -> None:
            with state_lock:
                if terminal_claimed or visible.done():
                    return
            if completed.cancelled():
                cancel_visible()
                return
            try:
                side_result = completed.result()
            except BaseException as exc:
                publish_exception(exc)
                return
            if side == "stock" and stock_flight_key is not None and completed is shared:
                # Per-subscriber annotations must not mutate another proposal's
                # result object or the cache-owned metric payload.
                side_result = replace(
                    side_result,
                    metric_observations=dict(side_result.metric_observations or {}),
                )
                if wait_state["waiting"] and not wait_state["worker_spawned"]:
                    side_result.metric_observations["singleflight_shared"] = True
            publish_result(side_result)

        source.add_done_callback(finish_source)
        if wait_state["waiting"] and not source.done() and not visible.done():

            def emit_wait_heartbeat() -> None:
                nonlocal wait_heartbeat_timer
                with state_lock:
                    previous_timer = wait_heartbeat_timer
                    wait_heartbeat_timer = None
                    if previous_timer is not None:
                        timers.discard(previous_timer)
                    if (
                        terminal_claimed
                        or visible.done()
                        or wait_state["worker_spawned"]
                    ):
                        return
                try:
                    report_wait_state()
                except BaseException as exc:
                    if cancellation_event is not None:
                        cancellation_event.set()
                    publish_exception(exc)
                    return
                schedule_wait_heartbeat()

            def schedule_wait_heartbeat() -> None:
                nonlocal wait_heartbeat_timer
                with state_lock:
                    if (
                        terminal_claimed
                        or visible.done()
                        or wait_state["worker_spawned"]
                    ):
                        return
                    timer = threading.Timer(30.0, emit_wait_heartbeat)
                    timer.daemon = True
                    wait_heartbeat_timer = timer
                    timers.add(timer)
                    timer.start()

            schedule_wait_heartbeat()

        def finalize_side(completed: Future[ScoreResult]) -> None:
            with state_lock:
                if wait_heartbeat_timer is not None:
                    wait_heartbeat_timer.cancel()
                    timers.discard(wait_heartbeat_timer)
                active_side_views.discard(completed)

        visible.add_done_callback(finalize_side)
        return visible

    def retrying_side(side: str, checkout: Path, retry: int = 0) -> Future[ScoreResult]:
        current = evaluate_side(side, checkout, retry)
        retried: Future[ScoreResult] = Future()

        def finish_or_retry(done: Future[ScoreResult]) -> None:
            if done.cancelled():
                retried.cancel()
                return
            try:
                side_result = done.result()
            except BaseException as exc:
                retried.set_exception(exc)
                return
            current_attempt = (
                int(attempt_counts.get((dataset.task_id, side), 0)) + retry + 1
            )
            if current_attempt < 2 and _retryable_quality_failure(side_result):
                try:
                    _check_cancellation(cancellation_event)
                    next_future = retrying_side(side, checkout, retry + 1)
                except BaseException as exc:
                    retried.set_exception(exc)
                    return
                next_future.add_done_callback(copy_retry)
                return
            retried.set_result(side_result)

        def copy_retry(done: Future[ScoreResult]) -> None:
            if done.cancelled():
                retried.cancel()
            else:
                try:
                    retried.set_result(done.result())
                except BaseException as exc:
                    retried.set_exception(exc)

        current.add_done_callback(finish_or_retry)
        return retried

    stock_future = retrying_side("stock", stock_checkout)

    def after_stock(stock_done: Future[ScoreResult]) -> None:
        if result.done():
            return
        if stock_done.cancelled():
            result.cancel()
            return
        try:
            stock = stock_done.result()
            _check_cancellation(cancellation_event)
            patch_future = retrying_side("patch", patch_checkout)
        except BaseException as exc:
            result.set_exception(exc)
            return

        def after_patch(patch_done: Future[ScoreResult]) -> None:
            if result.done():
                return
            if patch_done.cancelled():
                result.cancel()
                return
            try:
                patched = patch_done.result()
                result.set_result(
                    {
                        "spec": spec,
                        "stock": asdict(stock),
                        "patched": asdict(patched),
                        "stock_result": stock,
                        "patched_result": patched,
                        "search_ran": bool(
                            (stock.metric_observations or {}).get("search_ran")
                            and (patched.metric_observations or {}).get("search_ran")
                        ),
                    }
                )
            except BaseException as exc:
                result.set_exception(exc)

        patch_future.add_done_callback(after_patch)

    stock_future.add_done_callback(after_stock)
    return result


def run_quality_stock_jobs(
    *,
    stock_checkout: Path,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    registry: QualityRegistry | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
) -> list[dict]:
    """Stock-only baselines on the pre-registered set (cache for later pairs)."""

    stock_cache_root = _stock_cache_root(stock_cache_root)

    def one(
        task_id: str,
        *,
        registry: QualityRegistry,
        n_jobs: int,
        cancellation_event: threading.Event,
    ) -> dict:
        dataset = get_dataset(task_id, registry)
        spec = job_spec(dataset, n_jobs=n_jobs, registry=registry)
        stock = run_quality_side(
            dataset,
            checkout=stock_checkout,
            side="stock",
            n_jobs=n_jobs,
            registry=registry,
            cancellation_event=cancellation_event,
            artifact_root=artifact_root,
            stock_cache_root=stock_cache_root,
        )
        return {
            "spec": spec,
            "stock": asdict(stock),
            "search_ran": bool((stock.metric_observations or {}).get("search_ran")),
            "cache_hit": bool((stock.metric_observations or {}).get("cache_hit")),
        }

    return _run_over_tasks(
        one, task_ids=task_ids, n_jobs=n_jobs, cpu_quota=cpu_quota, registry=registry
    )


def run_quality_jobs(
    *,
    stock_checkout: Path,
    patch_checkout: Path,
    task_ids: tuple[str, ...] | None = None,
    n_jobs: int | None = None,
    cpu_quota: int | None = None,
    registry: QualityRegistry | None = None,
    on_side_event: SideEventCallback | None = None,
    on_side_wait: SideWaitCallback | None = None,
    prior_side_results: dict[tuple[str, str], ScoreResult] | None = None,
    prior_attempt_counts: dict[tuple[str, str], int] | None = None,
    proposal_id: str | None = None,
    evaluation_identity: dict | None = None,
    artifact_root: Path | str | None = None,
    stock_cache_root: Path | str | None = None,
    pair_scheduler: SharedPairScheduler | None = None,
    pair_group_id: int | None = None,
) -> list[dict]:
    """Run 1–N registered pairs. Parallelism is datasets × n_jobs under cpu_quota."""

    stock_cache_root = _stock_cache_root(stock_cache_root)

    def one(
        task_id: str,
        *,
        registry: QualityRegistry,
        n_jobs: int,
        cancellation_event: threading.Event,
    ) -> dict:
        return run_quality_pair(
            task_id,
            stock_checkout=stock_checkout,
            patch_checkout=patch_checkout,
            n_jobs=n_jobs,
            registry=registry,
            on_side_event=on_side_event,
            on_side_wait=on_side_wait,
            prior_side_results=prior_side_results,
            prior_attempt_counts=prior_attempt_counts,
            cancellation_event=cancellation_event,
            proposal_id=proposal_id,
            evaluation_identity=evaluation_identity,
            artifact_root=artifact_root,
            stock_cache_root=stock_cache_root,
        )

    def async_one(
        task_id: str,
        *,
        registry: QualityRegistry,
        n_jobs: int,
        cancellation_event: threading.Event,
    ) -> Future[dict]:
        if pair_scheduler is None:
            raise RuntimeError("async quality pair requires a shared pair scheduler")
        return _run_quality_pair_async(
            task_id,
            stock_checkout=stock_checkout,
            patch_checkout=patch_checkout,
            submit_compute=pair_scheduler.submit_compute,
            n_jobs=n_jobs,
            registry=registry,
            on_side_event=on_side_event,
            on_side_wait=on_side_wait,
            prior_side_results=prior_side_results,
            prior_attempt_counts=prior_attempt_counts,
            cancellation_event=cancellation_event,
            proposal_id=proposal_id,
            evaluation_identity=evaluation_identity,
            artifact_root=artifact_root,
            stock_cache_root=stock_cache_root,
        )

    return _run_over_tasks(
        one,
        task_ids=task_ids,
        n_jobs=n_jobs,
        cpu_quota=cpu_quota,
        registry=registry,
        pair_scheduler=pair_scheduler,
        pair_group_id=pair_group_id,
        async_one=async_one if pair_scheduler is not None else None,
    )


def _run_over_tasks(
    one,
    *,
    task_ids: tuple[str, ...] | None,
    n_jobs: int | None,
    cpu_quota: int | None,
    registry: QualityRegistry | None,
    pair_scheduler: SharedPairScheduler | None = None,
    pair_group_id: int | None = None,
    async_one=None,
) -> list[dict]:
    """Run one stock/patch pair per dataset, optionally through a shared pool."""

    registry = registry or load_quality_registry()
    ids = task_ids or list_quality_task_ids(registry)
    jobs = n_jobs if n_jobs is not None else registry.n_jobs_per_job
    quota = cpu_quota if cpu_quota is not None else registry.cpu_quota
    if (
        isinstance(jobs, bool)
        or not isinstance(jobs, int)
        or jobs <= 0
        or isinstance(quota, bool)
        or not isinstance(quota, int)
        or quota <= 0
    ):
        raise ValueError("n_jobs and cpu_quota must be positive integers")
    if quota < jobs:
        raise ValueError("cpu_quota cannot fund one n_jobs task")
    parallel = min(7, len(ids), quota // jobs)
    cancellation = threading.Event()
    original_error: list[BaseException] = []
    error_lock = threading.Lock()

    def evaluate(task_id):
        try:
            _check_cancellation(cancellation)
            return one(
                task_id, registry=registry, n_jobs=jobs, cancellation_event=cancellation
            )
        except BaseException as exc:
            # Set before this worker exits, so a newly scheduled task cannot
            # start ML while the coordinator has not yet observed the failure.
            with error_lock:
                if not original_error:
                    original_error.append(exc)
                cancellation.set()
            raise

    if pair_scheduler is not None:
        try:
            if async_one is not None:

                def schedule(task_id):
                    _check_cancellation(cancellation)
                    return async_one(
                        task_id,
                        registry=registry,
                        n_jobs=jobs,
                        cancellation_event=cancellation,
                    )

                pair_futures = (
                    pair_scheduler.run_group(ids, schedule)
                    if pair_group_id is None
                    else pair_scheduler.run_registered_group(
                        pair_group_id, ids, schedule
                    )
                )
                by_future = {}
                completed_pairs: SimpleQueue[Future] = SimpleQueue()
                for future in pair_futures:
                    future.add_done_callback(completed_pairs.put)
                first_error: BaseException | None = None
                for _ in pair_futures:
                    future = completed_pairs.get()
                    try:
                        by_future[future] = future.result()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                            cancellation.set()
                            for pending in pair_futures:
                                pending.cancel()
                            pair_scheduler.fail_closed(exc)
                if first_error is not None:
                    raise pair_scheduler.fatal_error or first_error
                return [by_future[future] for future in pair_futures]
            if pair_group_id is None:
                return pair_scheduler.run_group(ids, evaluate)
            return pair_scheduler.run_registered_group(pair_group_id, ids, evaluate)
        except BaseException:
            cancellation.set()
            raise

    if parallel <= 1 or len(ids) == 1:
        return [evaluate(task_id) for task_id in ids]
    pool = ThreadPoolExecutor(max_workers=parallel)
    futures = {}
    try:
        futures = {pool.submit(evaluate, task_id): task_id for task_id in ids}
        by_id = {futures[future]: future.result() for future in as_completed(futures)}
    except BaseException as exc:
        with error_lock:
            failure = (
                original_error[0]
                if original_error
                and not isinstance(exc, (KeyboardInterrupt, SystemExit))
                else exc
            )
        cancellation.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise failure
    else:
        pool.shutdown(wait=True)
    return [by_id[task_id] for task_id in ids]


def decision_from_quality_jobs(rows: list[dict]) -> Decision:
    skipped_rows = [
        row
        for row in rows
        if str(row.get("status") or "") in {"skipped", "inapplicable"}
        or str((row.get("spec") or {}).get("status") or "") == "skipped"
    ]
    if skipped_rows:
        unresolved = tuple(
            dict.fromkeys(
                str((row.get("spec") or {}).get("source_dataset") or "unknown_dataset")
                for row in skipped_rows
            )
        )
        return Decision(
            keep=None,
            reason="required_dataset_skipped:" + ",".join(unresolved),
            target_delta=None,
            stage="quality",
            quality_outcome="not_evaluated",
            unresolved_task_ids=unresolved,
        )
    if not rows:
        return Decision(
            keep=None,
            reason="no_applicable_quality_tasks",
            target_delta=None,
            stage="quality",
            infrastructure_error=False,
            quality_outcome="not_evaluated",
        )
    deltas: dict[str, float | None] = {}
    unresolved: list[str] = []
    failure_reasons: list[str] = []
    failure_kind: str | None = None
    timed_out = False
    fedot_budget_exhausted = False
    retryable_failure = False
    for row in rows:
        task_id = row["spec"]["source_dataset"]
        stock_result = row["stock_result"]
        patched_result = row["patched_result"]
        retryable_failure = retryable_failure or _retryable_quality_failure(
            stock_result
        )
        retryable_failure = retryable_failure or _retryable_quality_failure(
            patched_result
        )
        timeout_results = [
            result
            for result in (stock_result, patched_result)
            if result.status == "timeout"
        ]
        if timeout_results:
            budget_timeout_results = [
                result
                for result in timeout_results
                if (result.metric_observations or {}).get("failure_kind")
                == "fedot_budget_timeout"
            ]
            if budget_timeout_results:
                fedot_budget_exhausted = True
                failure_kind = failure_kind or "fedot_budget_timeout"
                failure_reasons.append(
                    f"FEDOT fit budget exhausted for {task_id}; metric not measured"
                )
            else:
                timed_out = True
                failure_reasons.append(f"quality {task_id} worker deadline timeout")
            unresolved.append(task_id)
            continue
        if stock_result.status != "ok" or not math.isfinite(float(stock_result.score)):
            unresolved.append(task_id)
            failure_kind = failure_kind or "evaluation_error"
            observations = stock_result.metric_observations or {}
            failure_reasons.append(
                f"stock_evaluation_failed:{task_id}:"
                f"{stock_result.status}:score={stock_result.score}:"
                f"failure_kind={observations.get('failure_kind', 'unknown')}:"
                f"failed_stage={observations.get('failed_stage', '')}"
            )
            continue
        stock_observations = stock_result.metric_observations or {}
        if stock_observations.get("metric_valid") is not True:
            unresolved.append(task_id)
            failure_kind = failure_kind or "unverified_metric"
            failure_reasons.append(
                f"stock_metric_unverified:{task_id}:metric_valid!=true"
            )
            continue
        if patched_result.status != "ok" or not math.isfinite(
            float(patched_result.score)
        ):
            unresolved.append(task_id)
            observations = patched_result.metric_observations or {}
            failure_kind = (
                "candidate_error"
                if observations.get("failure_kind")
                in {"fedot_import_error", "fedot_fit_error"}
                else "evaluation_error"
            )
            failure_reasons.append(
                f"patched_quality_failed:{task_id}:"
                f"{patched_result.status}:score={patched_result.score}:"
                f"failure_kind={observations.get('failure_kind', 'unknown')}:"
                f"failed_stage={observations.get('failed_stage', '')}"
            )
            continue
        patched_observations = patched_result.metric_observations or {}
        if patched_observations.get("metric_valid") is not True:
            unresolved.append(task_id)
            failure_kind = failure_kind or "unverified_metric"
            failure_reasons.append(
                f"patched_metric_unverified:{task_id}:metric_valid!=true"
            )
            continue
        spec = row["spec"]
        before = float(stock_result.score)
        after = float(patched_result.score)
        delta = after - before if spec["higher_is_better"] else before - after
        if spec.get("delta_mode", "absolute") == "relative":
            delta /= max(abs(before), 1e-12)
        deltas[task_id] = delta

    if unresolved:
        return Decision(
            keep=None,
            reason="; ".join(failure_reasons),
            target_delta=None,
            regression_deltas=deltas,
            stage="quality",
            infrastructure_error=timed_out or retryable_failure,
            quality_failure=None
            if timed_out or retryable_failure
            else "fedot_budget_timeout"
            if fedot_budget_exhausted
            else failure_kind or "evaluation_error",
            quality_outcome="not_evaluated",
            improved_task_count=sum(
                1
                for row in rows
                if row["spec"]["source_dataset"] in deltas
                and deltas[row["spec"]["source_dataset"]]
                >= float(row["spec"]["min_delta"])
            ),
            regressed_task_count=sum(
                1 for delta in deltas.values() if delta is not None and delta < -1e-12
            ),
            unresolved_task_ids=tuple(dict.fromkeys(unresolved)),
        )

    improved_task_ids: list[str] = []
    regressed_task_ids: list[str] = []
    for row in rows:
        spec = row["spec"]
        task_id = spec["source_dataset"]
        delta = deltas[task_id]
        assert delta is not None
        rounding_tolerance = 1e-12 * max(
            1.0,
            abs(float(row["stock_result"].score)),
            abs(float(row["patched_result"].score)),
        )
        if delta < -rounding_tolerance:
            regressed_task_ids.append(task_id)
        if delta >= float(spec["min_delta"]):
            improved_task_ids.append(task_id)
    if regressed_task_ids:
        return Decision(
            keep=False,
            reason=f"regression {regressed_task_ids[0]} delta {deltas[regressed_task_ids[0]]:.4f}",
            target_delta=None,
            regression_deltas=deltas,
            stage="quality",
            quality_outcome="regression",
            improved_task_count=len(improved_task_ids),
            regressed_task_count=len(regressed_task_ids),
        )
    if len(improved_task_ids) > len(rows) / 2:
        outcome = "broad_positive"
        keep = True
        reason = "quality_improved"
    elif len(improved_task_ids) >= 2:
        outcome = "local_positive"
        keep = True
        reason = "quality_improved"
    else:
        outcome = "insufficient_effect"
        keep = False
        reason = "no_quality_gain"
    return Decision(
        keep=keep,
        reason=reason,
        target_delta=None,
        regression_deltas=deltas,
        stage="quality",
        final_keep=keep,
        quality_outcome=outcome,
        improved_task_count=len(improved_task_ids),
    )
