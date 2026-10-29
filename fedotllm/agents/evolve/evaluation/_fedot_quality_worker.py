"""Subprocess worker: one bounded FEDOT AutoML side on a registered fold."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import logging
import os
import math
import re
import sys
import threading
import time
import traceback as tb_mod
from pathlib import Path

from fedotllm.agents.evolve.storage.journal import write_json_atomic
from fedotllm.agents.evolve.evaluation.quality_registry import (
    TUNING_EVALUATION_POLICIES,
)

STACK_DUMP_INTERVAL_SECONDS = 60.0
SUPPORTED_CANDIDATE_DISPATCHERS = {
    "golem-joblib-v1",
    "isolated-candidate-process-v1",
}


class _BoundedLogBuffer:
    """Retain at most 1 MiB of UTF-8 log bytes plus persistent search evidence."""

    def __init__(self, max_bytes: int = 1024 * 1024):
        if max_bytes <= 0:
            raise ValueError("diagnostic buffer size must be positive")
        self.max_bytes = max_bytes
        self._parts = deque()
        self._size = 0
        self.composition_started = False
        self.composition_skipped = False
        self.last_log_line = ""
        self.last_log_at = ""
        self.fedot_phase = ""
        self.fedot_phase_at = ""
        self.fedot_phase_line = ""

    def write(self, text: str) -> int:
        lower = text.lower()
        self.composition_started |= "pipeline composition started" in lower
        self.composition_skipped |= "timeout is too small for composing" in lower
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.last_log_line = lines[-1][-500:]
            self.last_log_at = datetime.now(timezone.utc).isoformat()
        phase_markers = (
            ("initial pipeline fitting started", "initial_assumption_fit"),
            ("pipeline composition started", "composition"),
            ("hyperparameters tuning started", "tuning"),
            ("timeout is too small for composing", "initial_assumption_fallback"),
            ("model generation finished", "post_search_fit"),
            ("final pipeline was fitted", "fit_complete"),
            ("already fitted initial pipeline is used", "fit_complete"),
        )
        for line in lines:
            lower_line = line.lower()
            for marker, phase in phase_markers:
                if marker in lower_line:
                    self.fedot_phase = phase
                    self.fedot_phase_at = datetime.now(timezone.utc).isoformat()
                    self.fedot_phase_line = line[-500:]
                    break
        data = text.encode("utf-8", errors="replace")[-self.max_bytes :]
        if not data:
            return len(text)
        self._parts.append(data)
        self._size += len(data)
        while self._size > self.max_bytes:
            oldest = self._parts.popleft()
            excess = self._size - self.max_bytes
            if len(oldest) > excess:
                self._parts.appendleft(oldest[excess:])
                self._size -= excess
            else:
                self._size -= len(oldest)
        return len(text)

    def getvalue(self) -> str:
        return b"".join(self._parts).decode("utf-8", errors="replace")

    def flush(self) -> None:
        pass


def _summarize_fedot_log(tail: str) -> dict:
    """Extract best-effort optimizer progress without claiming active work."""

    summary = {}
    for line in tail.splitlines():
        generation = re.search(r"Generation num:\s*(\d+)\s+size:\s*(\d+)", line)
        if generation:
            summary["last_observed_generation"] = int(generation.group(1))
            summary["last_observed_population_size"] = int(generation.group(2))
        evaluated = re.search(
            r"(\d+) individuals out of (\d+) in previous population were evaluated successfully",
            line,
            re.IGNORECASE,
        )
        if evaluated:
            summary["last_population_evaluated"] = int(evaluated.group(1))
            summary["last_population_size"] = int(evaluated.group(2))
        lagged = re.search(
            r"((?:Sparse)?LaggedTransformationImplementation).*?Window size.*?changed.*?"
            r"from\s+(\d+)\s+to\s+(\d+)",
            line,
            re.IGNORECASE,
        )
        if lagged:
            summary["last_observed_operation"] = lagged.group(1)
            summary["last_observed_window_size_before"] = int(lagged.group(2))
            summary["last_observed_window_size"] = int(lagged.group(3))
    if summary:
        summary["progress_source"] = "best_effort_worker_log_tail"
    return summary


def _is_fedot_budget_timeout(exc: BaseException) -> bool:
    """Recognize only the typed timeout from the installed FEDOT runtime."""
    # A runtime that raised its typed exception has already loaded this module.
    # Importing FEDOT here for unrelated validation/setup errors can initialize
    # its logging handlers before the worker runtime has even been configured.
    module = sys.modules.get("fedot.api.api_utils.timeout")
    timeout_type = getattr(module, "FedotBudgetExceeded", None)
    return isinstance(timeout_type, type) and isinstance(exc, timeout_type)


def _read_cpu_counters() -> dict:
    """Read this worker's CPU counters and, when available, reaped children."""

    try:
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {
            "source": "resource.getrusage",
            "scope": "worker_and_reaped_children",
            "children_supported": True,
            "worker_user_seconds": float(own.ru_utime),
            "worker_system_seconds": float(own.ru_stime),
            "reaped_children_user_seconds": float(children.ru_utime),
            "reaped_children_system_seconds": float(children.ru_stime),
        }
    except (ImportError, AttributeError, OSError, ValueError):
        try:
            process_seconds = getattr(time, "process_time", None)
            source = "time.process_time"
            if process_seconds is None:

                def process_seconds():
                    return time.process_time_ns() / 1_000_000_000

                source = "time.process_time_ns"
            return {
                "source": source,
                "scope": "worker_process_only",
                "children_supported": False,
                "worker_process_seconds": float(process_seconds()),
            }
        except (AttributeError, OSError, ValueError):
            return {
                "source": "unavailable",
                "scope": "unavailable",
                "children_supported": False,
            }


def _cpu_accounting_progress(
    current: dict, previous: dict | None, wall_interval: float | None
) -> dict:
    """Label counter scope and deltas; child totals cover only already-reaped children."""

    progress = {
        **current,
        "wall_interval_seconds": wall_interval,
        "accounted_cpu_core_equivalents": None,
    }
    counter_keys = [
        key
        for key, value in current.items()
        if key.endswith("_seconds") and isinstance(value, (int, float))
    ]
    deltas = []
    for key in counter_keys:
        delta_key = key.removesuffix("_seconds") + "_delta_seconds"
        previous_value = previous.get(key) if previous else None
        delta = (
            None if previous_value is None else max(0.0, current[key] - previous_value)
        )
        progress[delta_key] = delta
        if delta is not None:
            deltas.append(delta)
    if (
        wall_interval is not None
        and wall_interval > 0
        and len(deltas) == len(counter_keys)
    ):
        progress["accounted_cpu_core_equivalents"] = sum(deltas) / wall_interval
    return progress


@contextmanager
def _periodic_stack_dumps(
    path: Path | str | None,
    *,
    interval_seconds: float | None = None,
):
    """Sample this worker without signals or an unsafe native watchdog timer.

    Captures Python call sites throughout evaluation. A native call holding the
    GIL delays the sampler; that coverage limit is saved with every result.
    """
    from fedotllm.agents.evolve.evaluation.stack_sampler import periodic_stack_samples

    interval = (
        STACK_DUMP_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    )
    with periodic_stack_samples(
        path, interval_seconds=interval, scope="worker_process_threads"
    ) as status:
        yield status


class _StageTimings:
    """Publish atomic stage state and independent heartbeats during long fits."""

    def __init__(
        self,
        progress_path: Path | None = None,
        *,
        identity=None,
        log_buffer=None,
        heartbeat_interval_seconds: float = 30.0,
    ):
        if (
            not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat_interval_seconds must be finite and positive")
        self.records: dict[str, dict] = {}
        self.last_stage = ""
        self.failed_stage = ""
        self.progress_path = progress_path
        self.identity = dict(identity or {})
        self.log_buffer = log_buffer
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._lock = threading.RLock()
        self._active_stage: str | None = None
        self._active_started_monotonic: float | None = None
        self._active_started_at: str | None = None
        self._heartbeat_error: BaseException | None = None
        self._last_worker_log_size: int | None = None
        self._last_optimizer_event_at: str | None = None
        self._last_cpu_counters: dict | None = None
        self._last_cpu_sample_monotonic: float | None = None
        self.cpu_accounting: dict = {"source": "unavailable", "scope": "unavailable"}

    def _sample_cpu_accounting(self) -> dict:
        try:
            current = _read_cpu_counters()
        except Exception:
            # Diagnostic counters must never turn a successful model fit into a failure.
            current = {
                "source": "unavailable",
                "scope": "unavailable",
                "children_supported": False,
            }
        sampled_at = time.monotonic()
        wall_interval = (
            max(0.0, sampled_at - self._last_cpu_sample_monotonic)
            if self._last_cpu_sample_monotonic is not None
            else None
        )
        self.cpu_accounting = _cpu_accounting_progress(
            current, self._last_cpu_counters, wall_interval
        )
        self._last_cpu_counters = current
        self._last_cpu_sample_monotonic = sampled_at
        return dict(self.cpu_accounting)

    def _publish(self, event: str) -> None:
        if self.progress_path is None:
            return
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            elapsed = (
                time.monotonic() - self._active_started_monotonic
                if self._active_started_monotonic is not None
                else 0.0
            )
            progress = {
                **self.identity,
                "last_stage": self.last_stage,
                "active_stage": self._active_stage,
                "stage_started_at": self._active_started_at,
                "stage_elapsed_seconds": elapsed,
                "event": event,
                "updated_at": now,
                "failed_stage": self.failed_stage,
                "stage_timings": {
                    key: dict(value) for key, value in self.records.items()
                },
                "cpu_accounting": self._sample_cpu_accounting(),
            }
            if self.log_buffer is not None:
                progress["last_worker_output_line"] = self.log_buffer.last_log_line
                progress["last_worker_output_at"] = self.log_buffer.last_log_at
                progress["fedot_phase"] = self.log_buffer.fedot_phase or None
                progress["fedot_phase_at"] = self.log_buffer.fedot_phase_at or None
                progress["fedot_phase_line"] = self.log_buffer.fedot_phase_line or None
            worker_log_path = self.identity.get("worker_log_path")
            if worker_log_path:
                try:
                    log_path = Path(worker_log_path)
                    with log_path.open("rb") as handle:
                        handle.seek(0, os.SEEK_END)
                        size = handle.tell()
                        handle.seek(max(0, size - 64 * 1024))
                        tail = handle.read().decode("utf-8", errors="replace")
                    lines = [line.strip() for line in tail.splitlines() if line.strip()]
                    stat = log_path.stat()
                    progress["worker_log_size_bytes"] = stat.st_size
                    progress["last_worker_log_at"] = datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat()
                    if lines:
                        progress["last_worker_log_line"] = lines[-1][-500:]
                    tail_start = max(0, size - 64 * 1024)
                    previous_size = self._last_worker_log_size
                    if (
                        previous_size is None
                        or previous_size < tail_start
                        or previous_size > size
                    ):
                        new_text = tail
                    else:
                        new_text = tail[previous_size - tail_start :]
                    self._last_worker_log_size = size
                    optimizer_summary = _summarize_fedot_log(new_text)
                    if optimizer_summary:
                        # Only a new optimizer marker is evidence of recent
                        # FEDOT output. File mtime can change for wrapper lines.
                        self._last_optimizer_event_at = now
                        progress["last_optimizer_event_at"] = now
                        progress["last_fedot_log_at"] = now
                        optimizer_lines = [
                            line.strip()
                            for line in new_text.splitlines()
                            if re.search(
                                r"Generation num:|individuals out of .* evaluated successfully|"
                                r"LaggedTransformationImplementation.*Window size.*changed",
                                line,
                                re.IGNORECASE,
                            )
                        ]
                        if optimizer_lines:
                            progress["last_fedot_log_line"] = optimizer_lines[-1][-500:]
                        progress.update(optimizer_summary)
                    elif self._last_optimizer_event_at is not None:
                        progress["last_optimizer_event_at"] = (
                            self._last_optimizer_event_at
                        )
                        progress["last_fedot_log_at"] = self._last_optimizer_event_at
                except OSError:
                    # The bounded in-process log buffer remains a useful fallback
                    # when a custom runtime does not expose the redirected file.
                    pass
            temporary = self.progress_path.with_name(
                f".{self.progress_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, ensure_ascii=False, default=str))
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(self.progress_path)
                if hasattr(os, "O_DIRECTORY"):
                    directory_fd = os.open(
                        self.progress_path.parent, os.O_RDONLY | os.O_DIRECTORY
                    )
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_interval_seconds):
            try:
                self._publish("heartbeat")
            except BaseException as exc:
                self._heartbeat_error = exc
                return

    @contextmanager
    def measure(self, stage: str):
        with self._lock:
            self.last_stage = stage
            self._active_stage = stage
            self._active_started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        self._active_started_at = started_at
        started = time.monotonic()
        with self._lock:
            self.records[stage] = {"started_at": started_at, "status": "running"}
        print(f"worker stage={stage} event=start timestamp={started_at}", flush=True)
        self._publish("start")
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            daemon=True,
            name="quality-worker-heartbeat",
        )
        heartbeat_thread.start()
        status = "end"
        original = None
        try:
            yield
        except BaseException as exc:
            original = exc
            status = "failed"
            self.failed_stage = stage
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=max(1.0, self.heartbeat_interval_seconds))
            finished_at = datetime.now(timezone.utc).isoformat()
            duration = time.monotonic() - started
            with self._lock:
                self.records[stage] = {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": duration,
                    "status": status,
                }
                self._active_stage = None
                self._active_started_monotonic = None
                self._active_started_at = None
            print(
                f"worker stage={stage} event={status} timestamp={finished_at} "
                f"duration_seconds={duration:.6f}",
                flush=True,
            )
            try:
                self._publish(status)
                if self._heartbeat_error is not None:
                    raise OSError(
                        f"worker heartbeat could not be saved: {self._heartbeat_error}"
                    )
            except OSError as persistence_error:
                if original is None:
                    raise
                if hasattr(original, "add_note"):
                    original.add_note(
                        f"Stage progress save failed: {persistence_error}"
                    )


def _history_generations(history) -> int:
    if history is None:
        return 0
    for attr in ("generations", "individuals", "archive_history"):
        value = getattr(history, attr, None)
        if value is not None:
            try:
                return int(len(value))
            except TypeError:
                continue
    empty = getattr(history, "is_empty", None)
    if callable(empty) and empty():
        return 0
    return 1


def _search_ran(
    model,
    log_text: str,
    *,
    composition_started: bool = False,
    composition_skipped: bool = False,
) -> tuple[bool, str]:
    composer = getattr(model, "api_composer", None)
    was_optimised = bool(getattr(composer, "was_optimised", False))
    generations = _history_generations(getattr(model, "history", None))
    skipped = (
        composition_skipped or "timeout is too small for composing" in log_text.lower()
    )
    if skipped or not was_optimised:
        return False, "composing_did_not_start"
    if (
        generations <= 0
        and not composition_started
        and "pipeline composition started" not in log_text.lower()
    ):
        return False, "composing_did_not_start"
    return True, "composing_started"


def _positive_class_scores(prediction) -> "object":
    import numpy as np

    pred = np.asarray(getattr(prediction, "predict", prediction))
    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:, 0]
        if pred.shape[1] == 2:
            return pred[:, 1]
        return pred
    return pred.ravel()


def _score_classification(y_test, prediction, metric: str) -> float:
    """Holdout metric from probabilities. Avoids LabelEncoder.inverse_transform."""

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    y = np.asarray(y_test)
    if y.dtype.kind in "OUS":
        y = LabelEncoder().fit_transform(y)
    pred = np.asarray(getattr(prediction, "predict", prediction))
    name = str(metric or "roc_auc")
    if name == "roc_auc":
        scores = _positive_class_scores(pred)
        if getattr(scores, "ndim", 1) == 2:
            return float(roc_auc_score(y, scores, multi_class="ovr"))
        return float(roc_auc_score(y, scores))
    labels = (
        np.argmax(pred, axis=1)
        if pred.ndim == 2 and pred.shape[1] > 1
        else np.rint(pred.ravel())
    )
    if name == "f1":
        return float(f1_score(y, labels, average="macro"))
    return float(accuracy_score(y, labels))


def _score_regression(y_test, prediction, metric: str) -> float:
    """Compute a scale-aware regression metric from real-valued predictions."""

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    actual = np.asarray(y_test, dtype=float).reshape(-1)
    predicted = np.asarray(
        getattr(prediction, "predict", prediction), dtype=float
    ).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"regression prediction shape mismatch: target={actual.shape}, prediction={predicted.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("regression target or prediction contains non-finite values")
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(actual, predicted)))
    if metric == "mae":
        return float(mean_absolute_error(actual, predicted))
    raise ValueError(f"unsupported regression metric: {metric}")


def _predict_holdout(model, dataset, fold) -> str:
    """Official-fold features. Classification uses probs, not labels inverse."""

    if dataset.problem == "ts_forecasting":
        print("worker fit done; forecast", flush=True)
        if hasattr(model, "forecast"):
            model.forecast()
        else:
            model.predict(fold["X_test"])
        return "forecast"
    if dataset.problem == "classification":
        print("worker fit done; predict_proba", flush=True)
        model.predict_proba(fold["X_test"])
        return "predict_proba"
    print("worker fit done; predict", flush=True)
    model.predict(fold["X_test"])
    return "predict"


def _holdout_metrics(model, dataset, fold) -> tuple[float, dict]:
    metric_names = [dataset.metric]
    try:
        metrics = model.get_metrics(target=fold["y_test"], metric_names=metric_names)
        return float(metrics[dataset.metric]), dict(metrics)
    except Exception:
        if getattr(model, "prediction", None) is None:
            raise
        if dataset.problem == "classification":
            score = _score_classification(
                fold["y_test"], model.prediction, dataset.metric
            )
            return score, {
                dataset.metric: score,
                "fallback": "sklearn_from_predict_proba",
            }
        if dataset.problem == "regression":
            score = _score_regression(fold["y_test"], model.prediction, dataset.metric)
            return score, {dataset.metric: score, "fallback": "sklearn_from_prediction"}
        raise


def configure_worker_runtime(out_path: Path) -> dict:
    import os
    import tempfile
    import golem.core.paths as golem_paths

    runtime = Path(
        tempfile.mkdtemp(prefix=f"{out_path.stem}.runtime-", dir=str(out_path.parent))
    )
    temporary = runtime / "tmp"
    temporary.mkdir()
    golem = temporary / "GOLEM"
    golem.mkdir()
    os.environ["TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)
    # GOLEM hardcodes /tmp on Darwin. Bind its output path before importing
    # FEDOT/log/optimizer modules, which capture this function on import.
    # No installed dependency or FEDOT source file is changed.
    golem_paths.default_data_dir = lambda: str(golem)
    return {"temp_dir": str(temporary), "golem_data_dir": str(golem)}


def cache_policy_parameters(policy: str, problem: str) -> tuple[dict, tuple[str, ...]]:
    """Select public cache switches from the FEDOT runtime imported by this worker."""
    if policy == "isolated-worker-v1":
        return {}, ()
    if policy != "disabled-sqlite-v1":
        raise ValueError(f"unsupported FEDOT cache policy: {policy}")

    from fedot.api.api_utils.api_params_repository import ApiParamsRepository
    from fedot.core.repository.tasks import TaskTypesEnum

    defaults = ApiParamsRepository.default_params_for_task(TaskTypesEnum[problem])
    cache_keys = {
        key for key in defaults if key.startswith("use_") and key.endswith("_cache")
    }
    attributes = {
        "use_operations_cache": "operations_cache",
        "use_pipelines_cache": "pipelines_cache",
        "use_preprocessing_cache": "preprocessing_cache",
        "use_predictions_cache": "predictions_cache",
    }
    model_keys = cache_keys & {"use_operations_cache", "use_pipelines_cache"}
    if (
        len(model_keys) != 1
        or "use_preprocessing_cache" not in cache_keys
        or cache_keys - attributes.keys()
    ):
        raise ValueError(
            f"FEDOT cache policy has unsupported runtime switches: {sorted(cache_keys)}"
        )
    flags = {key: False for key in sorted(cache_keys)}
    return flags, tuple(attributes[key] for key in flags)


def assert_cache_policy(
    model, flags: dict, attributes: tuple[str, ...], *, phase: str
) -> None:
    """Fail closed if a runtime ignores switches or changes its cache API."""
    if not flags:
        return
    for key in flags:
        if model.params.get(key) is not False:
            raise ValueError(f"FEDOT cache policy switch {key} is not False {phase}")
    composer = model.api_composer
    for attribute in attributes:
        if not hasattr(composer, attribute):
            raise ValueError(
                f"FEDOT cache policy missing composer attribute {attribute} {phase}"
            )
        if getattr(composer, attribute) is not None:
            raise ValueError(f"FEDOT cache policy has active {attribute} {phase}")


def load_worker_fold(spec: dict):
    from fedotllm.agents.evolve.evaluation.openml_fold import load_registered_fold
    from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset, job_spec
    from fedotllm.agents.evolve.evaluation.smoke_data import (
        SMOKE_IDENTITY,
        load_smoke_fold,
        smoke_registry,
    )

    if spec.get("registry_identity") == SMOKE_IDENTITY:
        registry = smoke_registry()
        dataset = get_dataset(spec["source_dataset"], registry)
        expected = job_spec(dataset, registry=registry)
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("changed smoke worker profile")
        return dataset, load_smoke_fold(dataset)
    dataset = get_dataset(spec["source_dataset"])
    return dataset, load_registered_fold(dataset)


def main(argv: list[str] | None = None) -> int:
    from fedotllm.agents.evolve.evaluation.child_traces import traced_children
    from fedotllm.agents.evolve.evaluation.python_line_trace import (
        python_line_samples_from_environment,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec_json).read_text(encoding="utf-8"))
    with traced_children(
        spec.get("process_trace_dir"),
        attempt_id=str(spec.get("attempt_id", "unknown")),
        interval_seconds=float(spec.get("process_trace_interval_seconds", 60)),
    ) as child_diagnostics:
        with python_line_samples_from_environment(
            output_dir=spec.get("process_trace_dir"),
            interval_seconds=os.environ.get(
                "EVOLVE_TRACE_LINE_INTERVAL_SECONDS",
                spec.get("process_trace_interval_seconds", 60),
            ),
            context={
                key: spec[key]
                for key in (
                    "proposal_id",
                    "attempt_id",
                    "source_dataset",
                    "side",
                    "attempt",
                    "worker_invocation_id",
                )
                if spec.get(key) is not None
            },
        ) as line_diagnostics:
            with _periodic_stack_dumps(
                spec.get("stack_trace_path"),
                interval_seconds=spec.get("stack_dump_interval_seconds"),
            ) as diagnostics:
                payload, timings = _evaluate_worker(spec, Path(args.out))
                payload["metric_observations"]["stack_trace_diagnostics"] = {
                    **diagnostics,
                    "counter_snapshot_stage": "before_saved_payload",
                }
                payload["metric_observations"]["child_trace_diagnostics"] = dict(
                    child_diagnostics
                )
                payload["metric_observations"]["python_line_trace_diagnostics"] = dict(
                    line_diagnostics
                )
                payload["metric_observations"]["python_line_trace_path"] = (
                    line_diagnostics.get("output_path")
                )
                with timings.measure("saved_payload"):
                    write_json_atomic(Path(args.out), payload)
    return 0 if payload.get("status") != "crash" else 2


@contextmanager
def _worker_root_logging(log_stream: _BoundedLogBuffer):
    """Own only this evaluation's handlers; leave caller logging usable."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    handlers = [logging.StreamHandler(log_stream), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        for handler in handlers:
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.setLevel(previous_level)


def _evaluate_worker(spec: dict, out_path: Path) -> tuple[dict, _StageTimings]:
    log_stream = _BoundedLogBuffer()
    with _worker_root_logging(log_stream):
        return _evaluate_worker_with_logging(spec, out_path, log_stream)


def _evaluate_worker_with_logging(
    spec: dict,
    out_path: Path,
    log_stream: _BoundedLogBuffer,
) -> tuple[dict, _StageTimings]:
    """Evaluate with diagnostics active across import, data loading and scoring."""
    raw_composer_options = spec.get("composer_options", {})
    candidate_dispatcher = spec.get("candidate_dispatcher", "golem-joblib-v1")
    print(
        f"worker start {spec.get('source_dataset')} "
        f"side_timeout_s={spec.get('timeout_seconds')} "
        f"fedot_search_budget_s={spec.get('fedot_search_budget_seconds')} "
        f"fedot_timeout_min={spec.get('timeout_minutes')} "
        f"max_pipeline_fit_seconds={spec.get('max_pipeline_fit_time_seconds')} "
        f"max_pipeline_evaluation_seconds={spec.get('max_pipeline_evaluation_time_seconds')} "
        f"cv_folds={spec.get('cv_folds')} "
        f"n_jobs={spec.get('n_jobs')} preset={spec.get('preset')} "
        f"seed={spec.get('seed')} candidate_dispatcher={candidate_dispatcher} "
        f"composer_options={raw_composer_options}",
        flush=True,
    )
    timings = _StageTimings(
        out_path.with_suffix(".stages.json"),
        identity={
            key: spec.get(key)
            for key in (
                "proposal_id",
                "attempt_id",
                "source_dataset",
                "side",
                "attempt",
                "worker_invocation_id",
                "tuning_evaluation_policy",
                "registry_identity",
                "evaluation_identity",
                "preset",
                "n_jobs",
                "timeout_seconds",
                "fedot_search_budget_seconds",
                "fedot_timeout_policy",
                "final_fit_reserve_seconds",
                "timeout_minutes",
                "seed",
                "cv_folds",
                "max_pipeline_fit_time_seconds",
                "max_pipeline_fit_policy",
                "max_pipeline_evaluation_time_seconds",
                "max_pipeline_evaluation_policy",
                "candidate_dispatcher",
                "stack_trace_path",
                "fedot_trace_dir",
                "process_trace_dir",
                "process_trace_manifest_path",
                "composer_options",
                "worker_log_path",
            )
            if spec.get(key) is not None
        }
        | {"task_id": spec.get("source_dataset")},
        log_buffer=log_stream,
    )
    payload: dict
    worker_started = time.monotonic()
    cache_policy = spec.get("fedot_cache_policy", "isolated-worker-v1")
    cache_flags: dict = {}
    tuning_policy = spec.get("tuning_evaluation_policy", "legacy")
    tuning_isolation_observations = {"policy": tuning_policy, "enabled": False}
    cv_folds = None
    max_pipeline_fit_time_seconds = None
    max_pipeline_evaluation_time_seconds = None
    fit_policy = spec.get("max_pipeline_fit_policy", "per-cv-fold-v1")
    fedot_timeout_policy = spec.get("fedot_timeout_policy", "legacy")
    final_fit_reserve_seconds = int(spec.get("final_fit_reserve_seconds", 1500))
    evaluation_policy = spec.get(
        "max_pipeline_evaluation_policy", "per-candidate-evaluation-v1"
    )
    try:
        with timings.measure("validate_spec"):
            if (
                spec.get("tuning_evaluation_policy", "legacy")
                not in TUNING_EVALUATION_POLICIES
            ):
                raise ValueError("unsupported tuning evaluation policy")
            if not isinstance(raw_composer_options, dict):
                raise ValueError("composer_options must be an object")
            if (
                not isinstance(candidate_dispatcher, str)
                or candidate_dispatcher not in SUPPORTED_CANDIDATE_DISPATCHERS
            ):
                raise ValueError(
                    f"unsupported candidate_dispatcher: {candidate_dispatcher!r}"
                )
            if set(raw_composer_options) - {"pop_size", "num_of_generations"}:
                raise ValueError("unsupported FEDOT composer options")
            for key, value in raw_composer_options.items():
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"invalid FEDOT composer option {key}={value!r}")

        with timings.measure("import"):
            if (
                spec.get("runtime_paths_policy", "isolated-worker-v1")
                != "isolated-worker-v1"
            ):
                raise ValueError("unsupported worker runtime paths policy")
            runtime_paths = configure_worker_runtime(out_path)

        with timings.measure("fedot_import"):
            from fedot.api.main import Fedot

        with timings.measure("load_fold"):
            dataset, fold = load_worker_fold(spec)
        with timings.measure("configure_fit_budget"):
            cv_folds = int(
                spec.get("cv_folds")
                or (3 if dataset.problem == "ts_forecasting" else 5)
            )
            if cv_folds <= 0:
                raise ValueError("worker requires positive CV fold count")
            side_timeout_seconds = int(spec["timeout_seconds"])
            fedot_search_budget_seconds = int(spec["fedot_search_budget_seconds"])
            if not 0 < fedot_search_budget_seconds < side_timeout_seconds:
                raise ValueError(
                    "worker search budget must be positive and below the side deadline"
                )
            if fedot_timeout_policy not in {"legacy", "return_best"}:
                raise ValueError(
                    f"unsupported FEDOT timeout policy: {fedot_timeout_policy}"
                )
            if (
                final_fit_reserve_seconds < 0
                or final_fit_reserve_seconds >= fedot_search_budget_seconds
            ):
                raise ValueError(
                    "final fit reserve must be non-negative and below the FEDOT search budget"
                )
            expected_evaluation_limit = fedot_search_budget_seconds // cv_folds
            expected_fit_limit = (
                expected_evaluation_limit // cv_folds
                if fedot_timeout_policy == "return_best"
                else expected_evaluation_limit
            )
            configured_fit_limit = spec.get("max_pipeline_fit_time_seconds")
            max_pipeline_fit_time_seconds = (
                expected_fit_limit
                if configured_fit_limit is None
                else int(configured_fit_limit)
            )
            if max_pipeline_fit_time_seconds <= 0:
                raise ValueError(
                    "worker requires positive CV folds and per-fold fit limit"
                )
            if max_pipeline_fit_time_seconds != expected_fit_limit:
                raise ValueError(
                    "worker per-fold fit limit does not match the registered profile: "
                    f"expected {expected_fit_limit}, got {max_pipeline_fit_time_seconds}"
                )
            if fit_policy != "per-cv-fold-v1":
                raise ValueError(f"unsupported max_pipeline_fit_policy: {fit_policy}")
            configured_evaluation_limit = spec.get(
                "max_pipeline_evaluation_time_seconds"
            )
            max_pipeline_evaluation_time_seconds = (
                expected_evaluation_limit
                if configured_evaluation_limit is None
                else int(configured_evaluation_limit)
            )
            if max_pipeline_evaluation_time_seconds != expected_evaluation_limit:
                raise ValueError(
                    "worker total candidate evaluation limit does not match fedot_search_budget_seconds // cv_folds: "
                    f"expected {expected_evaluation_limit}, got {max_pipeline_evaluation_time_seconds}"
                )
            expected_evaluation_policy = (
                "per-cv-candidate-v1"
                if fedot_timeout_policy == "return_best"
                else "per-candidate-evaluation-v1"
            )
            if evaluation_policy != expected_evaluation_policy:
                raise ValueError(
                    f"unsupported total candidate evaluation policy: {evaluation_policy}"
                )
            timings.identity.update(
                {
                    "cv_folds": cv_folds,
                    "side_timeout_seconds": side_timeout_seconds,
                    "fedot_search_budget_seconds": fedot_search_budget_seconds,
                    "fedot_timeout_policy": fedot_timeout_policy,
                    "final_fit_reserve_seconds": final_fit_reserve_seconds,
                    "max_pipeline_fit_time_seconds": max_pipeline_fit_time_seconds,
                    "max_pipeline_fit_policy": fit_policy,
                    "max_pipeline_evaluation_time_seconds": max_pipeline_evaluation_time_seconds,
                    "max_pipeline_evaluation_policy": evaluation_policy,
                }
            )
            print(
                f"worker fit profile fedot_search_budget_seconds={fedot_search_budget_seconds} "
                f"planned_post_search_seconds={side_timeout_seconds - fedot_search_budget_seconds} "
                f"cv_folds={cv_folds} "
                f"max_pipeline_fit_time_seconds={max_pipeline_fit_time_seconds} "
                f"max_pipeline_evaluation_time_seconds={max_pipeline_evaluation_time_seconds} "
                f"fedot_timeout_policy={fedot_timeout_policy} "
                f"final_fit_reserve_seconds={final_fit_reserve_seconds} "
                f"evaluation_scope=per_candidate",
                flush=True,
            )
        print(
            f"worker fold loaded n_train={fold['n_train']} n_test={fold['n_test']} "
            f"loader={fold.get('loader')}; calling Fedot.fit",
            flush=True,
        )
        with timings.measure("fit"):
            cache_flags, cache_attributes = cache_policy_parameters(
                cache_policy, dataset.problem
            )
            # A unique attempt directory avoids FEDOT's default-directory
            # cleanup touching another worker's SQLite cache. Keep it for audit.
            cache_dir = out_path.with_suffix(".fedot-cache")
            cache_dir.mkdir(exist_ok=False)
            timeout_minutes = float(spec["timeout_minutes"])
            ts = dataset.problem == "ts_forecasting"
            extra = {}
            if ts:
                from fedot.core.repository.tasks import TsForecastingParams

                extra["task_params"] = TsForecastingParams(
                    forecast_length=int(
                        dataset.forecast_horizon or fold.get("forecast_horizon") or 12
                    )
                )
            composer_options = dict(raw_composer_options)
            if candidate_dispatcher == "isolated-candidate-process-v1":
                from fedot.core.optimisers.isolated_gp_optimizer import (
                    IsolatedEvoGraphOptimizer,
                )

                composer_options["optimizer"] = IsolatedEvoGraphOptimizer
            timeout_policy_options = (
                {
                    "max_pipeline_evaluation_time": float(
                        max_pipeline_evaluation_time_seconds
                    )
                    / 60.0
                }
                if fedot_timeout_policy == "return_best"
                else {}
            )
            model = Fedot(
                problem=dataset.problem,
                timeout=timeout_minutes,
                seed=int(spec["seed"]),
                n_jobs=int(spec["n_jobs"]),
                preset=str(spec["preset"]),
                with_tuning=bool(spec["with_tuning"]),
                timeout_policy=str(fedot_timeout_policy),
                final_fit_reserve_seconds=final_fit_reserve_seconds,
                cv_folds=cv_folds,
                max_pipeline_fit_time=float(max_pipeline_fit_time_seconds) / 60.0,
                metric=dataset.metric,
                safe_mode=False,
                logging_level=logging.INFO,
                cache_dir=str(cache_dir),
                **cache_flags,
                **timeout_policy_options,
                **extra,
                **composer_options,
            )
            assert_cache_policy(
                model, cache_flags, cache_attributes, phase="before fit"
            )
            fit_arguments = {"features": fold["X_train"]}
            if not ts:
                fit_arguments["target"] = fold["y_train"]
            if tuning_policy == "isolated-objective-process-v1" and spec.get(
                "with_tuning"
            ):
                from fedotllm.agents.evolve.evaluation.tuning_isolation import (
                    install_tuning_isolation,
                )

                isolation_spec = dict(
                    spec,
                    max_pipeline_fit_time_seconds=max_pipeline_fit_time_seconds,
                    max_pipeline_evaluation_time_seconds=max_pipeline_evaluation_time_seconds,
                )
                installation = install_tuning_isolation(isolation_spec, out_path)
                try:
                    with installation:
                        model.fit(**fit_arguments)
                finally:
                    tuning_isolation_observations = {
                        "enabled": True,
                        **installation.diagnostics(),
                    }
            else:
                model.fit(**fit_arguments)
            assert_cache_policy(model, cache_flags, cache_attributes, phase="after fit")
        with timings.measure("predict"):
            predict_path = _predict_holdout(model, dataset, fold)
        with timings.measure("metrics"):
            score, metrics = _holdout_metrics(model, dataset, fold)
            if not math.isfinite(float(score)):
                raise ValueError(
                    f"FEDOT returned a non-finite {dataset.metric} score: {score}"
                )
        log_text = log_stream.getvalue()
        search_ran, search_detail = _search_ran(
            model,
            log_text,
            composition_started=log_stream.composition_started,
            composition_skipped=log_stream.composition_skipped,
        )
        # A valid end-to-end prediction and metric is measurable quality even
        # when FEDOT skips composition (for example, its time-budget guard).
        # Keep optimizer activity as a separate diagnostic, not a validity gate.
        status = "ok"
        payload = {
            "task_id": dataset.task_id,
            "status": status,
            "score": score,
            "detail": search_detail if not search_ran else "",
            "n_train": fold["n_train"],
            "n_test": fold["n_test"],
            "seed": int(spec["seed"]),
            "metric_observations": {
                "runtime_paths": runtime_paths,
                "fedot_trace_enabled": bool(spec.get("fedot_trace_dir")),
                "fedot_trace_dir": spec.get("fedot_trace_dir"),
                "fedot_trace_interval_seconds": spec.get(
                    "fedot_trace_interval_seconds"
                ),
                "metric_valid": True,
                "search_ran": search_ran,
                "fedot_phase": log_stream.fedot_phase or None,
                "fedot_phase_at": log_stream.fedot_phase_at or None,
                "fedot_phase_line": log_stream.fedot_phase_line or None,
                "search_detail": search_detail,
                "was_optimised": bool(
                    getattr(
                        getattr(model, "api_composer", None), "was_optimised", False
                    )
                ),
                "was_tuned": bool(
                    getattr(getattr(model, "api_composer", None), "was_tuned", False)
                ),
                "history_generations": _history_generations(
                    getattr(model, "history", None)
                ),
                "metrics": metrics,
                "loader": fold.get("loader"),
                "preset": spec["preset"],
                "timeout_seconds": spec["timeout_seconds"],
                "fedot_search_budget_seconds": fedot_search_budget_seconds,
                "fedot_timeout_policy": fedot_timeout_policy,
                "final_fit_reserve_seconds": final_fit_reserve_seconds,
                "fedot_fit_budget_outcome": getattr(model, "fit_budget_outcome", None),
                "timeout_minutes": spec["timeout_minutes"],
                "cv_folds": cv_folds,
                "max_pipeline_fit_time_seconds": max_pipeline_fit_time_seconds,
                "max_pipeline_fit_policy": fit_policy,
                "max_pipeline_evaluation_time_seconds": max_pipeline_evaluation_time_seconds,
                "max_pipeline_evaluation_policy": evaluation_policy,
                "n_jobs": spec["n_jobs"],
                "candidate_dispatcher": candidate_dispatcher,
                "predict_path": predict_path,
                "fedot_cache_policy": cache_policy,
                "tuning_evaluation_policy": tuning_policy,
                "tuning_isolation": tuning_isolation_observations,
                "fedot_cache_flags": cache_flags,
                "fedot_cache_dir": str(cache_dir),
            },
            "log_tail": log_text[-4000:],
        }
        if not search_ran:
            payload["traceback"] = ""
    except Exception as exc:
        failed_stage = timings.failed_stage
        is_fedot_budget_timeout = _is_fedot_budget_timeout(exc)
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            failure_kind = "storage_error"
        elif is_fedot_budget_timeout:
            failure_kind = "fedot_budget_timeout"
        elif failed_stage == "fedot_import":
            failure_kind = "fedot_import_error"
        elif failed_stage in {"predict", "metrics"}:
            failure_kind = "scoring_error"
        elif failed_stage == "fit":
            failure_kind = "fedot_fit_error"
        else:
            failure_kind = "worker_error"
        payload = {
            "task_id": spec.get("source_dataset", ""),
            "status": "timeout" if is_fedot_budget_timeout else "crash",
            "failure_kind": failure_kind,
            "failed_stage": failed_stage,
            "last_stage": timings.last_stage,
            "score": None,
            "traceback": tb_mod.format_exc(),
            "detail": f"{type(exc).__name__}: {exc}",
            "n_train": 0,
            "seed": int(spec.get("seed") or 42),
            "metric_observations": {
                "search_ran": False,
                "fedot_trace_enabled": bool(spec.get("fedot_trace_dir")),
                "fedot_trace_dir": spec.get("fedot_trace_dir"),
                "fedot_trace_interval_seconds": spec.get(
                    "fedot_trace_interval_seconds"
                ),
                "fedot_phase": log_stream.fedot_phase or None,
                "fedot_phase_at": log_stream.fedot_phase_at or None,
                "fedot_phase_line": log_stream.fedot_phase_line or None,
                "fedot_cache_policy": cache_policy,
                "tuning_evaluation_policy": tuning_policy,
                "tuning_isolation": tuning_isolation_observations,
                "fedot_cache_flags": cache_flags,
                "cv_folds": cv_folds,
                "side_timeout_seconds": spec.get("timeout_seconds"),
                "fedot_search_budget_seconds": spec.get("fedot_search_budget_seconds"),
                "fedot_timeout_policy": fedot_timeout_policy,
                "final_fit_reserve_seconds": final_fit_reserve_seconds,
                "metric_outcome": "not_measured",
                "metric_valid": False,
                "fedot_budget_phase": getattr(exc, "phase", None)
                if is_fedot_budget_timeout
                else None,
                "fedot_budget_seconds": getattr(exc, "budget_seconds", None)
                if is_fedot_budget_timeout
                else None,
                "fedot_budget_elapsed_seconds": getattr(exc, "elapsed_seconds", None)
                if is_fedot_budget_timeout
                else None,
                "timeout_minutes": spec.get("timeout_minutes"),
                "max_pipeline_fit_time_seconds": max_pipeline_fit_time_seconds,
                "max_pipeline_fit_policy": fit_policy,
                "max_pipeline_evaluation_time_seconds": max_pipeline_evaluation_time_seconds,
                "max_pipeline_evaluation_policy": evaluation_policy,
            },
            "log_tail": log_stream.getvalue()[-4000:],
        }
    for key in (
        "proposal_id",
        "attempt_id",
        "source_dataset",
        "side",
        "attempt",
        "registry_identity",
        "evaluation_identity",
        "worker_invocation_id",
    ):
        if key in spec:
            payload[key] = spec[key]
    payload["duration_s"] = time.monotonic() - worker_started
    payload["cpu_accounting"] = timings._sample_cpu_accounting()
    payload["stage_timings"] = timings.records
    payload["last_stage"] = timings.last_stage
    payload.setdefault("failed_stage", timings.failed_stage)
    payload["stage_status_path"] = str(out_path.with_suffix(".stages.json"))
    return payload, timings


if __name__ == "__main__":
    raise SystemExit(main())
