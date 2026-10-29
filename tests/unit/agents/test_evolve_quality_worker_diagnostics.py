"""Short fixture checks for worker diagnostics; no FEDOT training or data access."""

import json
import logging
import errno
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker
from fedotllm.agents.evolve.execution import process


def _child_worker_blocked_in_native_call(stack_path, started):
    with worker._periodic_stack_dumps(stack_path, interval_seconds=0.05):
        started.set()
        time.sleep(30)


def test_diagnostic_buffer_keeps_bounded_tail_and_early_search_evidence():
    # Trimming an ordinary StringIO would retain unlimited memory and lose an
    # early search marker once large later messages replace its visible tail.
    buffer = worker._BoundedLogBuffer(max_bytes=1024)
    buffer.write("Pipeline composition started\n")
    for _ in range(300):
        buffer.write("long message " * 100)
    buffer.write("FINAL\n")
    assert len(buffer.getvalue().encode()) <= 1024
    assert buffer.getvalue().endswith("FINAL\n")
    assert buffer.composition_started is True
    buffer.write("timeout is too small for composing\n")
    assert buffer.composition_skipped is True


def test_diagnostic_buffer_tracks_last_fedot_fit_phase_without_refreshing_it_on_other_logs(
    tmp_path,
):
    buffer = worker._BoundedLogBuffer()
    timings = worker._StageTimings(tmp_path / "stages.json", log_buffer=buffer)
    phases = [
        ("Initial pipeline fitting started\n", "initial_assumption_fit"),
        ("Pipeline composition started.\n", "composition"),
        ("Hyperparameters tuning started with 20 min. timeout\n", "tuning"),
        ("Model generation finished\n", "post_search_fit"),
        ("Final pipeline was fitted\n", "fit_complete"),
    ]

    for message, expected_phase in phases:
        buffer.write(message)
        assert buffer.fedot_phase == expected_phase
        assert buffer.fedot_phase_line == message.strip()
        assert buffer.fedot_phase_at
        last_phase_at = buffer.fedot_phase_at
        timings._publish("heartbeat")
        sidecar = json.loads((tmp_path / "stages.json").read_text())
        assert sidecar["fedot_phase"] == expected_phase
        assert sidecar["fedot_phase_at"] == last_phase_at
        assert sidecar["fedot_phase_line"] == message.strip()
        buffer.write("worker heartbeat only\n")
        assert buffer.fedot_phase == expected_phase
        assert buffer.fedot_phase_at == last_phase_at
        timings._publish("heartbeat")
        sidecar = json.loads((tmp_path / "stages.json").read_text())
        assert sidecar["fedot_phase_at"] == last_phase_at

    buffer.write("Timeout is too small for composing; initial assumption is used\n")
    assert buffer.fedot_phase == "initial_assumption_fallback"


def test_stage_heartbeat_continues_during_long_fit(tmp_path):
    # A stage-start record alone cannot distinguish a long live fit from a
    # dead worker. The independent heartbeat must carry task identity and time.
    progress_path = tmp_path / "quality-out-fixture.stages.json"
    timings = worker._StageTimings(
        progress_path,
        identity={
            "proposal_id": "proposal-7",
            "task_id": "traffic",
            "side": "patch",
            "attempt": 2,
            "preset": "best_quality",
            "n_jobs": 4,
        },
        heartbeat_interval_seconds=0.01,
    )
    with timings.measure("fit"):
        time.sleep(0.05)
        heartbeat = json.loads(progress_path.read_text())
        assert heartbeat["event"] == "heartbeat"
        assert heartbeat["active_stage"] == "fit"
        assert heartbeat["stage_elapsed_seconds"] > 0
        assert heartbeat["task_id"] == "traffic"
        assert heartbeat["proposal_id"] == "proposal-7"
        assert heartbeat["side"] == "patch"
        assert heartbeat["attempt"] == 2
    final = json.loads(progress_path.read_text())
    assert final["event"] == "end"
    assert final["active_stage"] is None


def test_stage_heartbeat_reports_worker_and_reaped_child_cpu_deltas(
    tmp_path, monkeypatch
):
    progress_path = tmp_path / "quality-out-cpu.stages.json"
    samples = iter(
        [
            {
                "source": "resource.getrusage",
                "scope": "worker_and_reaped_children",
                "children_supported": True,
                "worker_user_seconds": 1.0,
                "worker_system_seconds": 0.2,
                "reaped_children_user_seconds": 4.0,
                "reaped_children_system_seconds": 0.5,
            },
            {
                "source": "resource.getrusage",
                "scope": "worker_and_reaped_children",
                "children_supported": True,
                "worker_user_seconds": 1.4,
                "worker_system_seconds": 0.3,
                "reaped_children_user_seconds": 6.5,
                "reaped_children_system_seconds": 0.7,
            },
        ]
    )
    monkeypatch.setattr(worker, "_read_cpu_counters", lambda: next(samples))
    timings = worker._StageTimings(progress_path, heartbeat_interval_seconds=10)

    timings._publish("start")
    timings._publish("heartbeat")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    cpu = progress["cpu_accounting"]
    assert cpu["source"] == "resource.getrusage"
    assert cpu["scope"] == "worker_and_reaped_children"
    assert cpu["children_supported"] is True
    assert cpu["worker_user_seconds"] == 1.4
    assert cpu["worker_user_delta_seconds"] == pytest.approx(0.4)
    assert cpu["worker_system_delta_seconds"] == pytest.approx(0.1)
    assert cpu["reaped_children_user_delta_seconds"] == pytest.approx(2.5)
    assert cpu["reaped_children_system_delta_seconds"] == pytest.approx(0.2)
    assert cpu["wall_interval_seconds"] >= 0


def test_stage_heartbeat_marks_self_only_cpu_fallback(tmp_path, monkeypatch):
    progress_path = tmp_path / "quality-out-cpu-fallback.stages.json"
    monkeypatch.setattr(
        worker,
        "_read_cpu_counters",
        lambda: {
            "source": "time.process_time_ns",
            "scope": "worker_process_only",
            "children_supported": False,
            "worker_process_seconds": 2.5,
        },
    )
    timings = worker._StageTimings(progress_path, heartbeat_interval_seconds=10)

    timings._publish("heartbeat")
    cpu = json.loads(progress_path.read_text(encoding="utf-8"))["cpu_accounting"]

    assert cpu["scope"] == "worker_process_only"
    assert cpu["children_supported"] is False
    assert cpu["worker_process_seconds"] == 2.5


def test_unavailable_cpu_telemetry_does_not_break_stage_heartbeat(
    tmp_path, monkeypatch
):
    progress_path = tmp_path / "quality-out-cpu-unavailable.stages.json"

    def unavailable():
        raise RuntimeError("fixture counter error")

    monkeypatch.setattr(worker, "_read_cpu_counters", unavailable)
    timings = worker._StageTimings(progress_path, heartbeat_interval_seconds=10)

    timings._publish("heartbeat")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    assert progress["event"] == "heartbeat"
    assert progress["cpu_accounting"]["scope"] == "unavailable"
    assert progress["cpu_accounting"]["accounted_cpu_core_equivalents"] is None


def test_cpu_time_ns_fallback_reports_its_actual_source(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setattr(worker.time, "process_time", None)
    monkeypatch.setattr(worker.time, "process_time_ns", lambda: 2_500_000_000)

    counters = worker._read_cpu_counters()

    assert counters["source"] == "time.process_time_ns"
    assert counters["worker_process_seconds"] == 2.5


def test_stage_heartbeat_reads_latest_line_from_shared_worker_log(tmp_path):
    # FEDOT's process-pool workers write directly to the supervisor-owned log
    # file; their output does not pass through the parent worker's log buffer.
    progress_path = tmp_path / "quality-out-fixture.stages.json"
    worker_log = tmp_path / "quality-log-fixture.log"
    worker_log.write_text("Generation num: 1 size: 4\n", encoding="utf-8")
    timings = worker._StageTimings(
        progress_path,
        identity={"task_id": "traffic", "worker_log_path": str(worker_log)},
        heartbeat_interval_seconds=10,
    )
    with timings.measure("fit"):
        worker_log.write_text(
            "MultiprocessingDispatcher - 4 individuals out of 4 in previous population were evaluated successfully.\n"
            "EvoGraphOptimizer - Generation num: 1 size: 4\n"
            "SparseLaggedTransformationImplementation - Window size of lagged transformation was changed from 0 to 356\n",
            encoding="utf-8",
        )
        timings._publish("heartbeat")
        heartbeat = json.loads(progress_path.read_text(encoding="utf-8"))
        assert heartbeat["last_fedot_log_line"].endswith("from 0 to 356")
        assert heartbeat["last_fedot_log_at"]
        assert heartbeat["last_observed_generation"] == 1
        assert heartbeat["last_observed_population_size"] == 4
        assert heartbeat["last_population_evaluated"] == 4
        assert heartbeat["last_population_size"] == 4
        assert (
            heartbeat["last_observed_operation"]
            == "SparseLaggedTransformationImplementation"
        )
        assert heartbeat["last_observed_window_size"] == 356


def test_non_fedot_log_append_does_not_refresh_last_optimizer_event(tmp_path):
    progress_path = tmp_path / "quality-out-fixture.stages.json"
    worker_log = tmp_path / "quality-log-fixture.log"
    worker_log.write_text(
        "EvoGraphOptimizer - Generation num: 1 size: 4\n", encoding="utf-8"
    )
    timings = worker._StageTimings(
        progress_path,
        identity={"task_id": "traffic", "worker_log_path": str(worker_log)},
    )

    timings._publish("heartbeat")
    first = json.loads(progress_path.read_text(encoding="utf-8"))
    first_optimizer_time = first.get("last_optimizer_event_at")
    assert first_optimizer_time

    with worker_log.open("a", encoding="utf-8") as handle:
        handle.write("worker stage=fit event=heartbeat\n")
    old_time = worker_log.stat().st_mtime
    os.utime(worker_log, (old_time + 60, old_time + 60))
    timings._publish("heartbeat")
    second = json.loads(progress_path.read_text(encoding="utf-8"))

    assert second["last_optimizer_event_at"] == first_optimizer_time
    assert second["last_worker_log_at"] != first.get("last_worker_log_at")


def test_periodic_fit_stack_dump_is_saved_and_timer_is_cancelled(tmp_path):
    import time

    stack_path = tmp_path / "quality-fit.stacks.txt"
    with worker._periodic_stack_dumps(stack_path, interval_seconds=0.04) as status:
        time.sleep(0.12)
        assert status["enabled"] is True
    content = stack_path.read_text(encoding="utf-8")
    assert "--- stack dump at " in content
    assert "test_periodic_fit_stack_dump_is_saved_and_timer_is_cancelled" in content
    assert status["dump_count"] >= 1
    size = stack_path.stat().st_size
    time.sleep(0.08)
    assert stack_path.stat().st_size == size, "stack timer remained armed after fit"


def test_periodic_fit_stack_dump_preserves_native_duration_and_declares_gil_limit(
    tmp_path,
):
    import ctypes

    libc = ctypes.PyDLL(None)
    blocking_sleep = libc.sleep
    blocking_sleep.argtypes = [ctypes.c_uint]
    blocking_sleep.restype = ctypes.c_uint
    stack_path = tmp_path / "quality-fit-gil-blocked.stacks.txt"

    with worker._periodic_stack_dumps(stack_path, interval_seconds=0.05) as status:
        time.sleep(0.1)
        # PyDLL deliberately keeps the GIL while this native call blocks. A
        # normal Python sampler thread cannot reliably run until it returns.
        started = time.monotonic()
        blocking_sleep(1)
        assert time.monotonic() - started >= 0.9
        assert status["enabled"] is True

    content = stack_path.read_text(encoding="utf-8")
    assert "--- stack dump at " in content
    assert "GIL" in status["limitations"]
    assert status["capture_method"] == "python_thread"
    assert status["dump_count"] >= 1


def test_stack_artifact_survives_worker_termination_after_dump(tmp_path):
    import multiprocessing
    import time

    context = multiprocessing.get_context("spawn")
    started = context.Event()
    stack_path = tmp_path / "quality-fit-killed.stacks.txt"
    child = context.Process(
        target=_child_worker_blocked_in_native_call,
        args=(str(stack_path), started),
    )
    child.start()
    try:
        assert started.wait(10), "test worker did not reach the fit context"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if stack_path.is_file() and "--- stack dump at " in stack_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                break
            if child.exitcode is not None:
                break
            time.sleep(0.02)
        assert stack_path.is_file()
        assert "--- stack dump at " in stack_path.read_text(
            encoding="utf-8", errors="replace"
        )
    finally:
        if child.is_alive():
            child.terminate()
        child.join(5)
    assert not child.is_alive()
    assert "--- stack dump at " in stack_path.read_text(
        encoding="utf-8", errors="replace"
    )


def test_periodic_fit_stack_dump_failure_does_not_interrupt_fit(tmp_path, monkeypatch):
    blocker = tmp_path / "file-not-directory"
    blocker.write_text("keep", encoding="utf-8")
    stack_path = blocker / "quality-fit.stacks.txt"

    with worker._periodic_stack_dumps(stack_path, interval_seconds=0.04) as status:
        assert status["enabled"] is False
        assert "FileExistsError" in status["error"]
        assert sum((1, 2, 3)) == 6


def test_expected_artifact_paths_are_stable_per_attempt_and_unique_across_retries(
    tmp_path,
):
    from fedotllm.agents.evolve.evaluation.fedot_quality import worker_artifact_paths

    first = worker_artifact_paths(
        "fedot-ts-traffic", "stock", "proposal-revision-2", 1, root=tmp_path
    )
    same_attempt = worker_artifact_paths(
        "fedot-ts-traffic", "stock", "proposal-revision-2", 1, root=tmp_path
    )
    retry = worker_artifact_paths(
        "fedot-ts-traffic", "stock", "proposal-revision-2", 2, root=tmp_path
    )

    assert first == same_attempt
    assert first["attempt_id"] == "proposal-revision-2:fedot-ts-traffic:stock:1"
    assert first["worker_log_path"] != retry["worker_log_path"]
    assert first["stage_status_path"].endswith(".stages.json")
    assert first["stack_trace_path"].endswith(".stacks.txt")
    assert first["stack_trace_path"] != retry["stack_trace_path"]


def _install_fixture_dependencies(
    monkeypatch,
    *,
    fail_stage=None,
    problem="regression",
    verbose_log=True,
    composer_options=None,
    fit_delay_seconds=0.0,
    expected_timeout_policy="legacy",
    expected_final_fit_reserve_seconds=1500,
    expected_max_pipeline_fit_minutes=None,
    expected_max_pipeline_evaluation_minutes=None,
    expected_optimizer=None,
):
    from fedotllm.agents.evolve.evaluation import openml_fold, quality_registry

    dataset = SimpleNamespace(
        task_id="fixture", problem=problem, metric="rmse", forecast_horizon=12
    )
    fold = {
        "X_train": [1],
        "y_train": [1],
        "X_test": [1],
        "y_test": [1],
        "n_train": 1,
        "n_test": 1,
        "loader": "fixture",
    }
    monkeypatch.setattr(quality_registry, "get_dataset", lambda _: dataset)
    monkeypatch.setattr(openml_fold, "load_registered_fold", lambda _: fold)

    class FedotBudgetExceeded(TimeoutError):
        phase = "final_refit"
        budget_seconds = 1500.0
        elapsed_seconds = 1500.2

    timeout_module = ModuleType("fedot.api.api_utils.timeout")
    timeout_module.FedotBudgetExceeded = FedotBudgetExceeded
    monkeypatch.setitem(sys.modules, "fedot.api.api_utils.timeout", timeout_module)

    class FixtureFedot:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 45
            assert kwargs["n_jobs"] == 4
            expected_fit = expected_max_pipeline_fit_minutes
            if expected_fit is None:
                expected_fit = 15 if problem == "ts_forecasting" else 9
            assert kwargs["max_pipeline_fit_time"] == expected_fit
            if expected_max_pipeline_evaluation_minutes is not None:
                assert (
                    kwargs["max_pipeline_evaluation_time"]
                    == expected_max_pipeline_evaluation_minutes
                )
            elif "max_pipeline_evaluation_time" in kwargs:
                assert kwargs["max_pipeline_evaluation_time"] == (
                    15 if problem == "ts_forecasting" else 9
                )
            assert kwargs["timeout_policy"] == expected_timeout_policy
            assert (
                kwargs["final_fit_reserve_seconds"]
                == expected_final_fit_reserve_seconds
            )
            for key, value in (composer_options or {}).items():
                assert kwargs[key] == value
            if expected_optimizer is not None:
                assert kwargs["optimizer"] is expected_optimizer
            cache_dir = Path(kwargs["cache_dir"])
            assert cache_dir.is_dir()
            assert cache_dir.name.endswith(".fedot-cache")
            self.api_composer = SimpleNamespace(was_optimised=True)
            self.history = None

        def fit(self, **kwargs):
            if fit_delay_seconds:
                time.sleep(fit_delay_seconds)
            logging.info("Pipeline composition started")
            if verbose_log:
                logging.info("x" * (2 * 1024 * 1024))
            if fail_stage == "fit":
                raise ValueError("fixture fit failure")
            if fail_stage == "budget":
                raise FedotBudgetExceeded("final refit reserve exhausted")

        def predict(self, values):
            if fail_stage == "predict":
                raise ValueError("fixture prediction failure")
            self.prediction = values

        def forecast(self):
            self.prediction = None

        def get_metrics(self, **kwargs):
            if fail_stage == "metrics":
                raise ValueError("fixture metric failure")
            if fail_stage == "storage":
                raise OSError(errno.ENOSPC, "No space left on device")
            return {"rmse": 0.125}

    fixture_module = ModuleType("fedot.api.main")
    fixture_module.Fedot = FixtureFedot
    monkeypatch.setitem(sys.modules, "fedot.api.main", fixture_module)


@pytest.mark.parametrize("fail_fit", [False, True])
def test_worker_reports_stage_timing_and_saves_bounded_diagnostics(
    tmp_path, monkeypatch, capsys, fail_fit
):
    # Losing stage records hides a stall/failure; retaining full logs grows the
    # payload with a long search. Dependency fixtures avoid any ML execution.
    _install_fixture_dependencies(monkeypatch, fail_stage="fit" if fail_fit else None)
    from fedotllm.agents.evolve.storage.journal import (
        write_json_atomic as atomic_writer,
    )

    atomic_outputs = []

    def record_atomic_write(path, payload):
        atomic_outputs.append(Path(path))
        atomic_writer(path, payload)

    monkeypatch.setattr(worker, "write_json_atomic", record_atomic_write, raising=False)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:2",
                "side": "patch",
                "attempt": 2,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
                "evaluation_identity": {"source_sha256": "source-v1", "n_jobs": 4},
            }
        )
    )
    out = tmp_path / "payload.json"
    before_handlers = list(logging.getLogger().handlers)
    before_level = logging.getLogger().level
    logging.getLogger().handlers.clear()
    try:
        rc = worker.main(["--spec-json", str(spec), "--out", str(out)])
    finally:
        logging.getLogger().handlers[:] = before_handlers
        logging.getLogger().setLevel(before_level)
    payload = json.loads(out.read_text())
    assert atomic_outputs == [out]
    output = capsys.readouterr().out
    assert rc == (2 if fail_fit else 0), payload.get("detail")
    assert payload["duration_s"] > 0
    assert payload["cpu_accounting"]["scope"] in {
        "worker_and_reaped_children",
        "worker_process_only",
        "unavailable",
    }
    expected = (
        [
            "validate_spec",
            "import",
            "fedot_import",
            "load_fold",
            "configure_fit_budget",
            "fit",
        ]
        if fail_fit
        else [
            "validate_spec",
            "import",
            "fedot_import",
            "load_fold",
            "configure_fit_budget",
            "fit",
            "predict",
            "metrics",
        ]
    )
    for stage in expected:
        timing = payload["stage_timings"][stage]
        assert timing["started_at"]
        assert timing["finished_at"]
        assert timing["duration_seconds"] >= 0
        assert f"worker stage={stage} event=start" in output
    assert "worker stage=saved_payload event=end" in output
    assert payload["last_stage"] == ("fit" if fail_fit else "metrics")
    progress = json.loads(out.with_suffix(".stages.json").read_text())
    assert progress["last_stage"] == "saved_payload"
    assert progress["event"] == "end"
    assert progress["failed_stage"] == ("fit" if fail_fit else "")
    assert progress["proposal_id"] == "proposal-7"
    assert progress["attempt_id"] == "proposal-7:fixture:patch:2"
    assert progress["task_id"] == "fixture"
    assert progress["side"] == "patch"
    assert progress["attempt"] == 2
    assert progress["registry_identity"] == "registry-v1"
    assert progress["evaluation_identity"] == {
        "source_sha256": "source-v1",
        "n_jobs": 4,
    }
    assert progress["worker_log_path"] == "/logs/worker.log"
    assert "cpu_accounting" in progress
    assert progress["stage_timings"]["fit"]["status"] == (
        "failed" if fail_fit else "end"
    )
    assert progress["stage_timings"]["saved_payload"]["duration_seconds"] >= 0
    assert out.with_suffix(".stages.json").stat().st_size < 10_000
    assert len(payload["log_tail"]) <= 4000
    assert payload["proposal_id"] == "proposal-7"
    assert payload["attempt_id"] == "proposal-7:fixture:patch:2"
    assert payload["source_dataset"] == "fixture"
    assert payload["side"] == "patch"
    assert payload["attempt"] == 2
    assert payload["registry_identity"] == "registry-v1"
    assert payload["evaluation_identity"] == {"source_sha256": "source-v1", "n_jobs": 4}
    assert payload["metric_observations"]["fedot_phase"] == "composition"
    assert (
        payload["metric_observations"]["fedot_phase_line"]
        == "Pipeline composition started"
    )
    if not fail_fit:
        assert payload["metric_observations"]["search_ran"] is True
        assert payload["metric_observations"]["fedot_search_budget_seconds"] == 2700
        assert payload["metric_observations"]["max_pipeline_fit_time_seconds"] == 540
        assert (
            payload["metric_observations"]["max_pipeline_evaluation_time_seconds"]
            == 540
        )


def test_worker_rejects_search_budget_that_consumes_entire_side_deadline(
    tmp_path, monkeypatch, capsys
):
    _install_fixture_dependencies(monkeypatch, verbose_log=False)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 60,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 3600,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": 5,
                "max_pipeline_fit_time_seconds": 720,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
            }
        )
    )
    out = tmp_path / "payload.json"

    assert worker.main(["--spec-json", str(spec), "--out", str(out)]) == 2

    payload = json.loads(out.read_text())
    assert payload["failed_stage"] == "configure_fit_budget"
    assert "below the side deadline" in payload["detail"]


def test_worker_records_periodic_stack_dumps_with_the_fit_attempt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(worker, "STACK_DUMP_INTERVAL_SECONDS", 0.04)
    _install_fixture_dependencies(
        monkeypatch, verbose_log=False, fit_delay_seconds=0.12
    )
    spec_path = tmp_path / "spec.json"
    stack_path = tmp_path / "quality-fit.stacks.txt"
    spec_path.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": 5,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "stack_trace_path": str(stack_path),
                "registry_identity": "registry-v1",
            }
        )
    )
    out = tmp_path / "payload.json"
    before_handlers = list(logging.getLogger().handlers)
    before_level = logging.getLogger().level
    logging.getLogger().handlers.clear()
    try:
        assert worker.main(["--spec-json", str(spec_path), "--out", str(out)]) == 0
    finally:
        logging.getLogger().handlers[:] = before_handlers
        logging.getLogger().setLevel(before_level)

    payload = json.loads(out.read_text(encoding="utf-8"))
    stages = json.loads(out.with_suffix(".stages.json").read_text(encoding="utf-8"))
    stack_text = stack_path.read_text(encoding="utf-8")
    stack_observations = payload["metric_observations"]["stack_trace_diagnostics"]
    assert stack_observations["enabled"] is True
    assert stack_observations["dump_count"] >= 1
    assert stack_observations["path"] == str(stack_path)
    assert "--- stack dump at " in stack_text
    assert "test_worker_records_periodic_stack_dumps_with_the_fit_attempt" in stack_text
    assert stages["stack_trace_path"] == str(stack_path)


@pytest.mark.parametrize(
    "operation",
    ["load_worker_fold", "_predict_holdout", "_holdout_metrics", "write_json_atomic"],
)
def test_worker_traces_silent_operations_outside_fit(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(worker, "STACK_DUMP_INTERVAL_SECONDS", 0.04)
    _install_fixture_dependencies(monkeypatch, verbose_log=False)
    original = getattr(worker, operation)

    def slow_operation(*args, **kwargs):
        time.sleep(0.16)
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, operation, slow_operation)
    spec = tmp_path / "spec.json"
    stack = tmp_path / "lifecycle.stacks.txt"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": 5,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "side": "patch",
                "attempt_id": "p:fixture:patch:1",
                "stack_trace_path": str(stack),
            }
        )
    )
    output = tmp_path / "payload.json"
    handlers = list(logging.getLogger().handlers)
    level = logging.getLogger().level
    try:
        assert worker.main(["--spec-json", str(spec), "--out", str(output)]) == 0
    finally:
        logging.getLogger().handlers[:] = handlers
        logging.getLogger().setLevel(level)

    assert "in slow_operation" in stack.read_text()
    payload = json.loads(output.read_text())
    diagnostics = payload["metric_observations"]["stack_trace_diagnostics"]
    if operation != "write_json_atomic":
        assert diagnostics["dump_count"] >= 1
    else:
        assert diagnostics["counter_snapshot_stage"] == "before_saved_payload"


def test_worker_passes_registered_composer_options_to_fedot(
    tmp_path, monkeypatch, capsys
):
    options = {"pop_size": 4, "num_of_generations": 2}
    _install_fixture_dependencies(monkeypatch, composer_options=options)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
                "composer_options": options,
                "evaluation_identity": {"source_sha256": "source-v1", "n_jobs": 4},
            }
        )
    )
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])

    payload = json.loads(out.read_text())
    assert rc == 0, payload.get("detail")
    assert payload["status"] == "ok"


def test_worker_selects_isolated_candidate_optimizer_only_when_registered(
    tmp_path, monkeypatch, capsys
):
    class IsolatedOptimizer:
        pass

    module = ModuleType("fedot.core.optimisers.isolated_gp_optimizer")
    module.IsolatedEvoGraphOptimizer = IsolatedOptimizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _install_fixture_dependencies(monkeypatch, expected_optimizer=IsolatedOptimizer)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "candidate_dispatcher": "isolated-candidate-process-v1",
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v2",
                "composer_options": {},
                "evaluation_identity": {"source_sha256": "source-v1", "n_jobs": 4},
            }
        )
    )
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])

    payload = json.loads(out.read_text())
    assert rc == 0, payload.get("detail")
    assert payload["status"] == "ok"


def test_worker_passes_registered_return_best_budget_policy_to_fedot(
    tmp_path, monkeypatch, capsys
):
    _install_fixture_dependencies(
        monkeypatch,
        expected_timeout_policy="return_best",
        expected_final_fit_reserve_seconds=1500,
        expected_max_pipeline_fit_minutes=1.8,
        expected_max_pipeline_evaluation_minutes=9,
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "fedot_timeout_policy": "return_best",
                "final_fit_reserve_seconds": 1500,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": 5,
                "max_pipeline_fit_time_seconds": 108,
                "max_pipeline_evaluation_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v2-return-best",
                "max_pipeline_evaluation_policy": "per-cv-candidate-v1",
            }
        )
    )
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])
    payload = json.loads(out.read_text())

    assert rc == 0, payload.get("detail")
    assert payload["status"] == "ok"
    assert payload["metric_observations"]["fedot_timeout_policy"] == "return_best"
    assert payload["metric_observations"]["final_fit_reserve_seconds"] == 1500


def test_worker_records_internal_fedot_budget_timeout_without_inventing_metric(
    tmp_path, monkeypatch, capsys
):
    _install_fixture_dependencies(
        monkeypatch,
        fail_stage="budget",
        expected_timeout_policy="return_best",
        expected_final_fit_reserve_seconds=1500,
        expected_max_pipeline_fit_minutes=1.8,
        expected_max_pipeline_evaluation_minutes=9,
        verbose_log=False,
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "fedot_timeout_policy": "return_best",
                "final_fit_reserve_seconds": 1500,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": 5,
                "max_pipeline_fit_time_seconds": 108,
                "max_pipeline_evaluation_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v2-return-best",
                "max_pipeline_evaluation_policy": "per-cv-candidate-v1",
            }
        )
    )
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])
    payload = json.loads(out.read_text())

    observations = payload["metric_observations"]
    assert rc == 0, payload.get("detail")
    assert payload["status"] == "timeout"
    assert payload["failure_kind"] == "fedot_budget_timeout"
    assert observations["metric_outcome"] == "not_measured"
    assert observations["metric_valid"] is False
    assert observations["fedot_budget_phase"] == "final_refit"
    assert observations["fedot_budget_seconds"] == 1500.0


@pytest.mark.parametrize("composer_options", [[], "", 0, None])
def test_worker_rejects_falsey_non_object_composer_options(
    tmp_path, monkeypatch, capsys, composer_options
):
    _install_fixture_dependencies(monkeypatch, verbose_log=False)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
                "composer_options": composer_options,
            }
        )
    )
    out = tmp_path / "payload.json"

    assert worker.main(["--spec-json", str(spec), "--out", str(out)]) == 2

    payload = json.loads(out.read_text())
    assert payload["failure_kind"] == "worker_error"
    assert payload["failed_stage"] == "validate_spec"


def test_worker_classifies_fedot_import_failure_separately(
    tmp_path, monkeypatch, capsys
):
    import builtins

    _install_fixture_dependencies(monkeypatch)
    monkeypatch.delitem(sys.modules, "fedot.api.main", raising=False)
    real_import = builtins.__import__

    def fail_fedot_import(name, *args, **kwargs):
        if name == "fedot.api.main":
            raise RuntimeError("fixture FEDOT import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_fedot_import)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
                "evaluation_identity": {"source_sha256": "source-v1", "n_jobs": 4},
            }
        )
    )
    out = tmp_path / "payload.json"

    rc = worker.main(["--spec-json", str(spec), "--out", str(out)])

    payload = json.loads(out.read_text())
    assert rc == 2
    assert payload["failure_kind"] == "fedot_import_error"
    assert payload["failed_stage"] == "fedot_import"


def test_worker_does_not_blame_fedot_for_runtime_setup_failure(
    tmp_path, monkeypatch, capsys
):
    _install_fixture_dependencies(monkeypatch)

    def fail_runtime_setup(_):
        raise RuntimeError("fixture runtime setup failure")

    monkeypatch.setattr(worker, "configure_worker_runtime", fail_runtime_setup)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "max_pipeline_fit_time_seconds": 540,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:1",
                "side": "patch",
                "attempt": 1,
                "worker_log_path": "/logs/worker.log",
                "registry_identity": "registry-v1",
                "evaluation_identity": {"source_sha256": "source-v1", "n_jobs": 4},
            }
        )
    )
    out = tmp_path / "payload.json"

    assert worker.main(["--spec-json", str(spec), "--out", str(out)]) == 2

    payload = json.loads(out.read_text())
    assert payload["failure_kind"] == "worker_error"
    assert payload["failed_stage"] == "import"


@pytest.mark.parametrize(
    "fail_stage,expected_failure_kind",
    [
        ("fit", "fedot_fit_error"),
        ("predict", "scoring_error"),
        ("metrics", "scoring_error"),
        ("storage", "storage_error"),
    ],
)
def test_worker_payload_identifies_failure_stage_and_kind(
    tmp_path, monkeypatch, capsys, fail_stage, expected_failure_kind
):
    _install_fixture_dependencies(
        monkeypatch,
        fail_stage=fail_stage,
        problem="ts_forecasting"
        if fail_stage in {"metrics", "storage"}
        else "regression",
        verbose_log=False,
    )
    spec = tmp_path / "spec.json"
    cv_folds = 3 if fail_stage in {"metrics", "storage"} else 5
    spec.write_text(
        json.dumps(
            {
                "source_dataset": "fixture",
                "timeout_minutes": 45,
                "timeout_seconds": 3600,
                "fedot_search_budget_seconds": 2700,
                "seed": 42,
                "n_jobs": 4,
                "cv_folds": cv_folds,
                "max_pipeline_fit_time_seconds": 2700 // cv_folds,
                "preset": "best_quality",
                "with_tuning": False,
                "proposal_id": "proposal-7",
                "attempt_id": "proposal-7:fixture:patch:2",
                "side": "patch",
                "attempt": 2,
                "worker_log_path": "/logs/worker.log",
            }
        )
    )
    out = tmp_path / "payload.json"
    assert worker.main(["--spec-json", str(spec), "--out", str(out)]) == 2
    payload = json.loads(out.read_text())
    expected_stage = "metrics" if fail_stage == "storage" else fail_stage
    assert payload["failed_stage"] == expected_stage
    assert payload["failure_kind"] == expected_failure_kind
    assert payload["last_stage"] == expected_stage
    assert payload["stage_timings"][expected_stage]["status"] == "failed"


def test_interpreter_identity_runs_named_probe_with_safe_environment(
    tmp_path, monkeypatch
):
    # A selected fixture interpreter accepts named scripts and rejects -c/-;
    # the old inline-code probe falls back to a file fingerprint instead of JSON.
    launcher = tmp_path / "fixture_interpreter"
    distribution = tmp_path / "thegolem-99.5.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: thegolem\nVersion: 99.5\n"
    )
    gph_distribution = tmp_path / "giotto_ph-0.2.4.dist-info"
    gph_distribution.mkdir()
    (gph_distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: giotto-ph\nVersion: 0.2.4\n"
    )
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "if any(arg in ('-c', '-') for arg in sys.argv[1:]): sys.exit(91)\n"
        "if os.environ.get('OPENAI_API_KEY'): sys.exit(92)\n"
        "if os.environ.get('PYTHONNOUSERSITE') != '1': sys.exit(93)\n"
        f"os.environ['PYTHONPATH'] = {str(tmp_path)!r}\n"
        "raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
    )
    launcher.chmod(0o700)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    identity = json.loads(process._interpreter_identity(str(launcher)))
    assert identity["implementation"] == "CPython"
    assert identity["executable"] == sys.executable
    assert identity["version"].startswith("3.")
    assert identity["packages"]["golem"] == "99.5"
    assert set(identity["packages"]) == {
        "fedot",
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "golem",
        "thegolem",
        "catboost",
        "xgboost",
        "lightgbm",
        "giotto-ph",
    }
    assert identity["packages"]["thegolem"] == "99.5"
    assert identity["packages"]["giotto-ph"] == "0.2.4"
    assert identity["platform"]["machine"]
    assert identity["platform"]["system"]
