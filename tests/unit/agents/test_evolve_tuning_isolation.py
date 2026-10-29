"""Real subprocess checks for the tuning objective's process boundary."""

import ctypes
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = "fedotllm.agents.evolve.evaluation.tuning_isolation"


def isolation_module():
    import importlib

    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            pytest.fail("controller-owned tuning process isolation is not implemented")
        raise


def evaluator(mode="ok", folds=3, marker=None):
    # Local classes serialize by value: the child must not import the pytest module.
    class Evaluator:
        def __init__(self):
            self._data_producer = lambda: iter(range(folds))
            self._objective = self.metric

        def metric(self, graph, **kwargs):
            if mode == "runtime":
                import tempfile
                import golem.core.paths as paths
                import golem.core.log as log

                Path(marker).write_text(
                    json.dumps(
                        {
                            "cwd": os.getcwd(),
                            "temp": tempfile.gettempdir(),
                            "golem": paths.default_data_dir(),
                            "log": log.default_data_dir(),
                        }
                    )
                )
            if mode == "rng":
                import random
                import numpy as np

                return SimpleNamespace(
                    valid=True, values=(random.random() + float(np.random.random()),)
                )
            return SimpleNamespace(
                valid=mode != "invalid",
                values=(float("nan") if mode == "nan" else 2.0,),
            )

        def evaluate(self, graph):
            results = []
            for fold in self._data_producer():
                if marker:
                    Path(marker).write_text(str(os.getpid()))
                if mode == "short_slow":
                    time.sleep(0.004)
                if mode == "sleep":
                    time.sleep(30)
                elif mode == "native":
                    # PyDLL keeps the GIL while libc sleeps, like an uninterruptible extension.
                    ctypes.PyDLL(None).sleep(30)
                elif mode == "crash":
                    os._exit(7)
                elif mode == "kernel_probe":
                    outcomes = {}
                    for name, arguments in [
                        ("setpgid", (0, 0)),
                        ("setsid", ()),
                        ("unshare", (0,)),
                        ("setns", (-1, 0)),
                    ]:
                        probe_path = Path(marker).with_suffix("." + name)
                        probe_pid = os.fork()
                        if probe_pid == 0:
                            library = ctypes.CDLL(None, use_errno=True)
                            result = getattr(library, name)(*arguments)
                            probe_path.write_text(
                                json.dumps(
                                    {"returncode": result, "errno": ctypes.get_errno()}
                                )
                            )
                            os._exit(0)
                        os.waitpid(probe_pid, 0)
                        outcomes[name] = json.loads(probe_path.read_text())
                    Path(marker).write_text(json.dumps(outcomes))
                elif mode == "escape":
                    import subprocess

                    child = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        start_new_session=True,
                    )
                    Path(marker).with_suffix(".escaped").write_text(str(child.pid))
                    time.sleep(30)
                elif mode == "kill_supervisor":
                    import signal

                    os.kill(os.getppid(), signal.SIGKILL)
                    time.sleep(30)
                elif mode == "nested":
                    import subprocess

                    subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(30)"]
                    )
                    time.sleep(30)
                elif mode == "thread":
                    import threading

                    threading.Thread(
                        target=lambda: time.sleep(30), daemon=False
                    ).start()
                result = self._objective(graph, fold_id=fold)
                results.append(result.values[0])
                if mode == "partial":
                    break
            return SimpleNamespace(valid=True, values=(sum(results) / len(results),))

    return Evaluator()


def runner(tmp_path, **kwargs):
    return isolation_module().ObjectiveProcessRunner(
        tmp_path,
        fold_seconds=kwargs.get("fold_seconds", 0.5),
        candidate_seconds=kwargs.get("candidate_seconds", 3.0),
    )


def test_success_requires_complete_cv_and_retains_fitness(tmp_path):
    result = runner(tmp_path).evaluate(
        evaluator(), None, tuning_deadline=time.monotonic() + 10
    )
    assert result.status == "complete"
    assert result.values == (2.0,)
    assert result.completed_folds == 3
    assert result.cleanup_verified is True
    assert result.worker_pid != os.getpid()
    with pytest.raises(ProcessLookupError):
        os.kill(result.worker_pid, 0)


@pytest.mark.parametrize("mode", ["sleep", "native", "nested"])
def test_fold_deadline_kills_and_reaps_computation(tmp_path, mode):
    started = time.monotonic()
    result = runner(tmp_path).evaluate(
        evaluator(mode), None, tuning_deadline=started + 10
    )
    assert time.monotonic() - started < 5
    assert result.status == "fold_timeout"
    assert result.values is None
    assert result.cleanup_verified
    with pytest.raises(ProcessLookupError):
        os.killpg(result.worker_pid, 0)


@pytest.mark.parametrize(
    "mode,status",
    [
        ("partial", "incomplete_cv"),
        ("nan", "invalid_fitness"),
        ("invalid", "invalid_fitness"),
        ("crash", "child_crash"),
    ],
)
def test_partial_invalid_and_crashed_results_never_become_fitness(
    tmp_path, mode, status
):
    result = runner(tmp_path).evaluate(
        evaluator(mode), None, tuning_deadline=time.monotonic() + 10
    )
    assert result.status == status
    assert result.values is None
    assert result.cleanup_verified


def test_remaining_tuning_budget_caps_candidate(tmp_path):
    started = time.monotonic()
    result = runner(tmp_path, fold_seconds=10, candidate_seconds=20).evaluate(
        evaluator("sleep"), None, tuning_deadline=started + 0.7
    )
    assert result.status == "tuning_timeout"
    assert time.monotonic() - started < 4
    assert result.cleanup_verified


def test_exhausted_budget_does_not_start_worker(tmp_path):
    result = runner(tmp_path).evaluate(
        evaluator(), None, tuning_deadline=time.monotonic() - 1
    )
    assert result.status == "tuning_timeout"
    assert result.worker_pid is None
    assert result.values is None


def test_candidate_budget_caps_without_fold_expiry(tmp_path):
    result = runner(tmp_path, fold_seconds=10, candidate_seconds=0.7).evaluate(
        evaluator("sleep"), None, tuning_deadline=time.monotonic() + 10
    )
    assert result.status == "candidate_timeout"
    assert result.cleanup_verified


def test_serialization_failure_has_no_worker_and_no_fitness(tmp_path):
    class Unpickleable:
        def __reduce__(self):
            raise ValueError("deliberate serialization failure")

    result = runner(tmp_path).evaluate(
        evaluator(), Unpickleable(), tuning_deadline=time.monotonic() + 10
    )
    assert result.status == "serialization_error"
    assert result.worker_pid is None
    assert result.values is None


def test_lingering_thread_is_cleaned_after_valid_result(tmp_path):
    result = runner(tmp_path).evaluate(
        evaluator("thread"), None, tuning_deadline=time.monotonic() + 10
    )
    assert result.status == "complete"
    assert result.cleanup_verified
    assert result.values == (2.0,)


def test_events_record_every_fold_and_cleanup(tmp_path):
    runner(tmp_path).evaluate(evaluator(), None, tuning_deadline=time.monotonic() + 10)
    events = [
        json.loads(line)
        for file in tmp_path.glob("*/events.jsonl")
        for line in file.read_text().splitlines()
    ]
    assert sum(event["event"] == "fold_started" for event in events) == 3
    assert sum(event["event"] == "fold_complete" for event in events) == 3
    assert any(event["event"] == "cleanup" and event["verified"] for event in events)


def test_installation_restores_runtime_and_never_edits_fedot_sources(tmp_path):
    import hashlib
    import fedot.core.pipelines.tuning.tuner_builder as builder

    before = hashlib.sha256(Path(builder.__file__).read_bytes()).digest()
    original = builder.TunerBuilder.build
    spec = {
        "tuning_evaluation_policy": "isolated-objective-process-v1",
        "max_pipeline_fit_time_seconds": 500,
        "max_pipeline_evaluation_time_seconds": 1500,
    }
    with isolation_module().install_tuning_isolation(
        spec, tmp_path / "result.json"
    ) as installation:
        assert builder.TunerBuilder.build is not original
        assert installation.diagnostics()["policy"] == "isolated-objective-process-v1"
    assert builder.TunerBuilder.build is original
    assert hashlib.sha256(Path(builder.__file__).read_bytes()).digest() == before


def test_best_completed_candidate_survives_later_timeout(tmp_path):
    module = isolation_module()
    session = module.TuningSession(runner(tmp_path), budget_seconds=10)
    session.start()
    assert session.evaluate(evaluator(), "best").values == (2.0,)
    assert session.evaluate(evaluator("sleep"), "blocked").values is None
    assert session.finish("initial", deviation=0.05) == "best"
    assert session.best_values == (2.0,)


def test_best_completed_candidate_survives_supervisor_error_after_verified_cleanup(
    tmp_path,
):
    module = isolation_module()

    class IntermittentRunner:
        directory = tmp_path

        def __init__(self):
            self.results = []

        def evaluate(self, evaluator, graph, *, tuning_deadline):
            result = (
                module.EvaluationResult("complete", values=(1.0,), completed_folds=3)
                if not self.results
                else module.EvaluationResult(
                    "supervisor_error",
                    cleanup_verified=True,
                    error="OSError: transient supervisor read failure",
                )
            )
            self.results.append(result)
            return result

    session = module.TuningSession(IntermittentRunner(), budget_seconds=10)
    session.start()
    assert session.evaluate(evaluator(), "best").values == (1.0,)
    with pytest.raises(RuntimeError, match="transient supervisor read failure"):
        session.evaluate(evaluator(), "unverified")

    recovered = session.recover_after_failure("initial", deviation=0.05)

    assert recovered == "best"
    assert session.restored_best
    assert session.recovered_failure


def test_supervisor_error_without_verified_cleanup_cannot_restore_candidate(tmp_path):
    module = isolation_module()

    class UnsafeRunner:
        directory = tmp_path

        def __init__(self):
            self.results = []

        def evaluate(self, evaluator, graph, *, tuning_deadline):
            result = (
                module.EvaluationResult("complete", values=(1.0,), completed_folds=3)
                if not self.results
                else module.EvaluationResult(
                    "supervisor_error",
                    cleanup_verified=False,
                    error="supervisor cleanup could not be verified",
                )
            )
            self.results.append(result)
            return result

    session = module.TuningSession(UnsafeRunner(), budget_seconds=10)
    session.start()
    session.evaluate(evaluator(), "best")
    with pytest.raises(RuntimeError, match="cleanup could not be verified"):
        session.evaluate(evaluator(), "unsafe")

    assert session.recover_after_failure("initial", deviation=0.05) is None
    assert not session.recovered_failure


def test_session_uses_shared_deadline_across_candidates(tmp_path):
    session = isolation_module().TuningSession(
        runner(tmp_path, fold_seconds=10, candidate_seconds=20), budget_seconds=0.7
    )
    session.start()
    assert session.evaluate(evaluator("sleep"), None).status == "tuning_timeout"
    result = session.evaluate(evaluator(), None)
    assert result.status == "tuning_timeout"
    assert result.worker_pid is None


@pytest.mark.parametrize("problem", ["classification", "regression", "ts_forecasting"])
def test_real_fedot_tuner_uses_isolated_objective(tmp_path, problem):
    # Worker protocol fixtures intentionally substitute FEDOT modules. Run the
    # real-library check in a fresh interpreter, as production workers do, so
    # those fixtures cannot leave imported enum/repository references behind.
    if os.environ.get("FEDOTLLM_REAL_TUNER_TEST") != problem:
        import subprocess

        environment = os.environ.copy()
        environment["FEDOTLLM_REAL_TUNER_TEST"] = problem
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).resolve()}::test_real_fedot_tuner_uses_isolated_objective[{problem}]",
                "-q",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return
    import numpy as np
    from datetime import timedelta
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.metrics_repository import (
        ClassificationMetricsEnum,
        RegressionMetricsEnum,
    )

    task = Task(
        TaskTypesEnum[problem],
        TsForecastingParams(forecast_length=4) if problem == "ts_forecasting" else None,
    )
    index = np.arange(80)
    if problem == "classification":
        target = index % 2
        features = np.column_stack([target, index / 80])
        pipeline = Pipeline(PrimaryNode("logit"))
        metric = ClassificationMetricsEnum.ROCAUC
    elif problem == "regression":
        features, target = np.column_stack([index, index**2]), index * 0.2 + 1
        pipeline = Pipeline(PrimaryNode("ridge"))
        metric = RegressionMetricsEnum.RMSE
    else:
        features = target = np.sin(index / 5) + index * 0.01
        lagged = PrimaryNode("lagged")
        lagged.parameters = {"window_size": 8}
        pipeline = Pipeline(SecondaryNode("ridge", nodes_from=[lagged]))
        metric = RegressionMetricsEnum.MAE
    data = InputData(
        idx=index,
        features=features,
        target=target,
        task=task,
        data_type=DataTypesEnum.ts
        if problem == "ts_forecasting"
        else DataTypesEnum.table,
    )
    spec = {
        "tuning_evaluation_policy": "isolated-objective-process-v1",
        "max_pipeline_fit_time_seconds": 10,
        "max_pipeline_evaluation_time_seconds": 40,
    }
    with isolation_module().install_tuning_isolation(
        spec, tmp_path / "result.json"
    ) as installation:
        tuner = (
            TunerBuilder(task)
            .with_cv_folds(2)
            .with_validation_blocks(2 if problem == "ts_forecasting" else None)
            .with_metric(metric)
            .with_n_jobs(1)
            .with_iterations(1)
            .with_timeout(timedelta(seconds=45))
            .build(data)
        )
        result = tuner.tune(pipeline, show_progress=False)
        assert result.nodes
        diagnostics = installation.diagnostics()
        assert diagnostics["evaluations"] >= 2
        assert diagnostics["completed"] >= 2
        assert diagnostics["cleanup_verified"]
        completed = [
            outcome
            for outcome in installation.runner.results
            if outcome.status == "complete"
        ]
        assert all(outcome.completed_folds == 2 for outcome in completed)
        assert all(np.isfinite(outcome.values[0]) for outcome in completed)


def test_parent_death_guard_cleans_child_group(tmp_path):
    import cloudpickle
    import subprocess

    payload = tmp_path / "parent-input.pkl"
    payload.write_bytes(cloudpickle.dumps(evaluator("nested")))
    script = tmp_path / "owner.py"
    script.write_text("""import cloudpickle, pathlib, sys, time
from fedotllm.agents.evolve.evaluation.tuning_isolation import ObjectiveProcessRunner
root = pathlib.Path(sys.argv[1])
with (root / "parent-input.pkl").open("rb") as stream:
    objective = cloudpickle.load(stream)
ObjectiveProcessRunner(root / "evaluations", fold_seconds=30, candidate_seconds=40).evaluate(objective, None, tuning_deadline=time.monotonic()+50)
""")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    parent = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)], env=env, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 5
        progress = []
        while time.monotonic() < deadline:
            progress = list((tmp_path / "evaluations").glob("*/progress.json"))
            if progress:
                break
            time.sleep(0.02)
        assert progress, "compute child did not reach a fold"
        directory = progress[0].parent
        pid = json.loads((directory / "worker.json").read_text())["worker_pid"]
        parent.kill()
        parent.wait(timeout=2)
        deadline = time.monotonic() + 5
        while not (directory / "status.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        status = json.loads((directory / "status.json").read_text())
        assert status["status"] == "parent_exited"
        assert status["cleanup_verified"]
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)


def test_unreviewed_runtime_source_is_rejected(tmp_path, monkeypatch):
    import fedot.core.pipelines.tuning.tuner_builder as builder

    original = builder.PipelineObjectiveEvaluate

    class UnreviewedEvaluator(original):
        pass

    monkeypatch.setattr(builder, "PipelineObjectiveEvaluate", UnreviewedEvaluator)
    spec = {
        "tuning_evaluation_policy": "isolated-objective-process-v1",
        "max_pipeline_fit_time_seconds": 500,
        "max_pipeline_evaluation_time_seconds": 1500,
    }
    with pytest.raises(RuntimeError, match="unsupported.*source"):
        with isolation_module().install_tuning_isolation(
            spec, tmp_path / "result.json"
        ):
            pass


def test_fold_completion_cannot_hide_an_overrun_between_supervisor_ticks(tmp_path):
    result = runner(tmp_path, fold_seconds=0.001).evaluate(
        evaluator("short_slow"), None, tuning_deadline=time.monotonic() + 10
    )
    assert result.status == "fold_timeout"
    assert result.values is None


def test_successful_child_preserves_native_random_streams(tmp_path):
    import random
    import numpy as np

    python_before, numpy_before = random.getstate(), np.random.get_state()
    try:
        random.seed(713)
        np.random.seed(713)
        python_reference, numpy_reference = (
            random.Random(713),
            np.random.RandomState(713),
        )
        expected = (
            sum(python_reference.random() + numpy_reference.random() for _ in range(3))
            / 3
        )
        result = runner(tmp_path, candidate_seconds=10).evaluate(
            evaluator("rng"), None, tuning_deadline=time.monotonic() + 15
        )
        assert result.status == "complete"
        assert result.values[0] == pytest.approx(expected)
        assert random.random() == python_reference.random()
        assert np.random.random() == numpy_reference.random()
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)


def test_compute_child_uses_private_runtime_before_fedot_import(tmp_path):
    marker = tmp_path / "runtime-observed.json"
    result = runner(tmp_path / "isolated", candidate_seconds=10).evaluate(
        evaluator("runtime", marker=marker), None, tuning_deadline=time.monotonic() + 15
    )
    assert result.status == "complete"
    observed = json.loads(marker.read_text())
    attempt = next((tmp_path / "isolated").glob("*/events.jsonl")).parent
    assert Path(observed["cwd"]) == attempt
    for key in ("temp", "golem", "log"):
        assert Path(observed[key]).is_relative_to(attempt)


def test_zero_remaining_tuning_budget_returns_without_spawning(tmp_path):
    session = isolation_module().TuningSession(runner(tmp_path), budget_seconds=0)
    session.start()
    result = session.evaluate(evaluator(), None)
    assert result.status == "tuning_timeout"
    assert result.worker_pid is None


def test_compute_cannot_create_an_escaping_python_subprocess(tmp_path):
    import signal

    marker = tmp_path / "pid.txt"
    escaped = marker.with_suffix(".escaped")
    try:
        result = runner(tmp_path / "runs").evaluate(
            evaluator("escape", marker=marker),
            None,
            tuning_deadline=time.monotonic() + 10,
        )
        assert not escaped.exists(), "compute created a process outside the owned group"
        assert result.status == "evaluation_error"
        assert result.cleanup_verified
    finally:
        if escaped.exists():
            try:
                os.kill(int(escaped.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_death_cannot_leave_native_compute_alive(tmp_path):
    import signal

    marker = tmp_path / "compute.pid"
    result = runner(tmp_path / "runs").evaluate(
        evaluator("kill_supervisor", marker=marker),
        None,
        tuning_deadline=time.monotonic() + 10,
    )
    assert result.values is None
    pid = int(marker.read_text())
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                # Darwin can transiently deny probing an orphan zombie group.
                # Wait for ESRCH; EPERM is never proof of successful cleanup.
                pass
            time.sleep(0.02)
        else:
            pytest.fail("compute group survived its supervisor")
    finally:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_containment_failure_latches_before_another_candidate_can_start(tmp_path):
    class Unpickleable:
        def __reduce__(self):
            raise ValueError("deliberate serialization failure")

    session = isolation_module().TuningSession(runner(tmp_path), budget_seconds=10)
    session.start()
    with pytest.raises(RuntimeError, match="isolation"):
        session.evaluate(evaluator(), Unpickleable())
    with pytest.raises(RuntimeError, match="isolation"):
        session.evaluate(evaluator(), None)
    assert len(session.runner.results) == 1


def test_supervisor_failure_keeps_root_cause_in_exception_and_diagnostics(tmp_path):
    module = isolation_module()

    class FailedRunner:
        directory = tmp_path
        results = []

        def evaluate(self, evaluator, graph, *, tuning_deadline):
            result = module.EvaluationResult(
                "supervisor_error",
                error="OSError: injected guardian pipe failure",
                cleanup_verified=True,
                completed_folds=2,
            )
            self.results.append(result)
            return result

    session = module.TuningSession(FailedRunner(), budget_seconds=10)
    session.start()
    with pytest.raises(RuntimeError, match="OSError: injected guardian pipe failure"):
        session.evaluate(evaluator(), None)

    installation = module.TuningIsolationInstallation.__new__(
        module.TuningIsolationInstallation
    )
    installation.runner = FailedRunner()
    installation.runner.results = [
        module.EvaluationResult(
            "supervisor_error",
            error="OSError: injected guardian pipe failure",
            cleanup_verified=True,
            completed_folds=2,
        )
    ]
    installation.sessions = [session]
    installation.runtime = {}
    diagnostics = installation.diagnostics()
    assert diagnostics["failures"] == [
        {
            "status": "supervisor_error",
            "error": "OSError: injected guardian pipe failure",
            "cleanup_verified": True,
            "completed_folds": 2,
            "evaluation_directory": None,
            "recovered_by_completed_candidate": False,
        }
    ]


def test_installation_accepts_only_verified_recovery_and_records_failure(tmp_path):
    module = isolation_module()
    result = module.EvaluationResult(
        "supervisor_error",
        error="OSError: isolated final-check failure",
        cleanup_verified=True,
        completed_folds=3,
    )
    session = module.TuningSession(runner(tmp_path), budget_seconds=10)
    session.last_result = result
    session.recovered_failure = True
    installation = module.TuningIsolationInstallation.__new__(
        module.TuningIsolationInstallation
    )
    installation.runner = runner(tmp_path)
    installation.runner.results = [result]
    installation.sessions = [session]
    installation.runtime = {}
    installation._original_build = None

    assert installation.__exit__(None, None, None) is False
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["failures"][0]["recovered_by_completed_candidate"] is True
    assert summary["cleanup_verified"] is True


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="requires Linux kernel seccomp"
)
def test_linux_filter_denies_native_group_and_namespace_changes_after_fork(tmp_path):
    import errno

    marker = tmp_path / "native-probes.json"
    result = runner(tmp_path / "runs", fold_seconds=2, candidate_seconds=10).evaluate(
        evaluator("kernel_probe", folds=1, marker=marker),
        None,
        tuning_deadline=time.monotonic() + 15,
    )
    assert result.status == "complete"
    assert result.containment["kernel_group_containment"]
    for probe in json.loads(marker.read_text()).values():
        assert probe == {"returncode": -1, "errno": errno.EPERM}
