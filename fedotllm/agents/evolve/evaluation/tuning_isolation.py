"""Controller-owned, fresh-process containment for FEDOT tuning objectives.

No FEDOT files are changed. Each candidate is supervised by a stdlib-only
process outside the worker session; a lifetime pipe covers abrupt worker death.
Fitness leaves the child only as bounded JSON, after every CV fold completed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

POLICY = "isolated-objective-process-v1"
_FATAL_STATUSES = {
    "serialization_error",
    "supervisor_error",
    "invalid_result",
    "bootstrap_error",
    "guardian_error",
}
# Inspected upstream 0.7.5 and candidate-parent-guard-v3 distributions. A new
# runtime must be reviewed and explicitly admitted rather than silently wrapped.
_REVIEWED_EVALUATOR_SOURCES = {
    "34524e0f98155b48e4c451cbcf3195052b9989ac97150e7744a511685a307a3f",
    "0cafbbb5c11e4a7bbe160778e571b4eeb00ee2fc99e65eec61886a87651f248d",
}
_REVIEWED_BUILDER_SOURCES = {
    "533eceb4d930372ffff30ef75c4eec756c4d9f76b2bfe756a3ee3968f66cc03d",
    "0d29353fde450498292562da1ca879769d8ab93b8d3f65bdb19e55b72b54b269",
}


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    values: tuple[float, ...] | None = None
    completed_folds: int = 0
    worker_pid: int | None = None
    cleanup_verified: bool = True
    error: str | None = None
    containment: dict | None = None
    evaluation_directory: str | None = None


class ObjectiveProcessRunner:
    def __init__(self, directory, *, fold_seconds, candidate_seconds):
        if os.name != "posix":
            raise RuntimeError("tuning isolation requires POSIX process groups")
        for value in (fold_seconds, candidate_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("isolation budgets must be finite and positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.fold_seconds = float(fold_seconds)
        self.candidate_seconds = float(candidate_seconds)
        self.results = []

    def evaluate(self, evaluator, graph, *, tuning_deadline):
        import cloudpickle
        from ._tuning_isolation_process import event

        started = time.monotonic()
        if not math.isfinite(tuning_deadline):
            raise ValueError("a finite shared tuning deadline is required")
        directory = self.directory / uuid.uuid4().hex
        directory.mkdir(mode=0o700)
        deadline = min(tuning_deadline, started + self.candidate_seconds)
        kind = (
            "tuning_timeout"
            if tuning_deadline <= started + self.candidate_seconds
            else "candidate_timeout"
        )
        event(
            directory,
            "evaluation_started",
            deadline=deadline,
            fold_seconds=self.fold_seconds,
            candidate_seconds=self.candidate_seconds,
            policy=POLICY,
        )
        if deadline <= started:
            result = EvaluationResult("tuning_timeout")
        else:
            try:
                with (directory / "input.pkl").open("xb") as stream:
                    import random

                    numpy = sys.modules.get("numpy")
                    rng = {
                        "python": random.getstate(),
                        "numpy": numpy.random.get_state()
                        if numpy is not None
                        else None,
                    }
                    cloudpickle.dump((evaluator, graph, rng), stream)
            except Exception as exc:
                result = EvaluationResult(
                    "serialization_error", error=f"{type(exc).__name__}: {exc}"
                )
            else:
                result = self._run(directory, deadline, kind)
        result = replace(result, evaluation_directory=str(directory))
        self.results.append(result)
        # Payload contains dataset folds and is temporary controller-owned state;
        # cleanup happens only after confirmed evaluation termination.
        if result.cleanup_verified:
            (directory / "input.pkl").unlink(missing_ok=True)
        event(
            directory,
            "parent_received",
            status=result.status,
            cleanup_verified=result.cleanup_verified,
        )
        return result

    def _run(self, directory, deadline, kind):
        from ._tuning_isolation_process import MAX_RESULT_BYTES

        if time.monotonic() >= deadline:
            return EvaluationResult(kind)
        parent_read, parent_write = os.pipe()
        helper = Path(__file__).with_name("_tuning_isolation_process.py")
        env = os.environ.copy()
        env["FEDOTLLM_TUNING_ISOLATION_CHILD"] = "1"
        # Match the worker's explicitly configured Python import environment.
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(p or os.getcwd()).resolve()) for p in sys.path
        )
        process = None
        try:
            with (directory / "supervisor.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(helper),
                        "supervise",
                        str(directory),
                        "--parent-fd",
                        str(parent_read),
                        "--deadline",
                        str(deadline),
                        "--deadline-kind",
                        kind,
                        "--fold-seconds",
                        str(self.fold_seconds),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    env=env,
                    cwd=str(directory),
                    pass_fds=(parent_read,),
                    close_fds=True,
                )
            os.close(parent_read)
            parent_read = None
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()) + 6)
            except subprocess.TimeoutExpired:
                # EOF requests guardian cleanup; never silently accept a result.
                os.close(parent_write)
                parent_write = None
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    return EvaluationResult(
                        "supervisor_unresponsive", cleanup_verified=False
                    )
            status = directory / "status.json"
            if not status.exists() or status.stat().st_size > MAX_RESULT_BYTES:
                return EvaluationResult("supervisor_error", cleanup_verified=False)
            data = json.loads(status.read_text())
            rng = data.pop("rng_state", None)
            if rng is not None and data.get("cleanup_verified"):
                import random

                state = rng["python"]
                random.setstate((state[0], tuple(state[1]), state[2]))
                if rng.get("numpy") is not None and "numpy" in sys.modules:
                    numpy = sys.modules["numpy"]
                    state = rng["numpy"]
                    numpy.random.set_state(
                        (state[0], numpy.asarray(state[1], dtype="uint32"), *state[2:])
                    )
            values = data.get("values")
            if data["status"] == "complete":
                if (
                    not values
                    or not all(math.isfinite(float(v)) for v in values)
                    or not data.get("completed_folds")
                ):
                    data.update(status="invalid_result", values=None)
                else:
                    data["values"] = tuple(float(v) for v in values)
            else:
                data["values"] = None
            return EvaluationResult(**data)
        finally:
            for descriptor in (parent_read, parent_write):
                if descriptor is not None:
                    os.close(descriptor)


class TuningSession:
    """One shared native tuning allowance, including initial/final validation."""

    def __init__(self, runner, *, budget_seconds):
        if not math.isfinite(budget_seconds) or budget_seconds < 0:
            raise ValueError("native tuner must supply a nonnegative finite timeout")
        self.runner = runner
        self.budget_seconds = budget_seconds
        self.deadline = None
        self.best_graph = None
        self.best_values = None
        self.initial_values = None
        self.last_result = None
        self.restored_best = False
        self.recovered_failure = False
        self.calls = 0
        self.failure = None
        self.failure_error = None

    def start(self):
        self.deadline = time.monotonic() + self.budget_seconds

    def evaluate(self, evaluator, graph):
        from copy import deepcopy

        if self.failure is not None:
            detail = f": {self.failure_error}" if self.failure_error else ""
            raise RuntimeError(
                f"tuning isolation failure is latched: {self.failure}{detail}"
            )
        if self.deadline is None:
            raise RuntimeError("objective called outside its tuning session")
        result = self.runner.evaluate(evaluator, graph, tuning_deadline=self.deadline)
        self.last_result = result
        if not result.cleanup_verified or result.status in _FATAL_STATUSES:
            self.failure = result.status
            self.failure_error = result.error
            detail = f": {result.error}" if result.error else ""
            raise RuntimeError(f"tuning isolation failed: {result.status}{detail}")
        if self.calls == 0:
            self.initial_values = result.values
        self.calls += 1
        if result.values is not None and (
            self.best_values is None or result.values[0] < self.best_values[0]
        ):
            self.best_values = result.values
            self.best_graph = deepcopy(graph)
        return result

    def finish(self, returned_graph, *, deviation):
        # Preserve an already fully validated candidate when the final repeated
        # check cannot finish. Keep BaseTuner's original improvement threshold.
        if (
            self.last_result is None
            or self.last_result.values is not None
            or self.best_values is None
        ):
            return returned_graph
        initial = self.initial_values[0] if self.initial_values is not None else None
        threshold = (
            initial - abs(initial) * deviation / 100 if initial is not None else None
        )
        # The original valid initial graph also remains eligible as fallback.
        if (
            threshold is None
            or self.best_values[0] <= threshold
            or self.best_values == self.initial_values
        ):
            self.restored_best = True
            return self.best_graph
        return returned_graph

    def recover_after_failure(self, fallback_graph, *, deviation):
        """Keep a validated CV candidate after a contained final-check failure.

        A supervisor error is recoverable only when cleanup was verified and
        an earlier candidate completed every CV fold with a finite score.
        Otherwise the caller must preserve the original fail-closed behavior.
        """
        result = self.last_result
        if (
            self.failure != "supervisor_error"
            or result is None
            or result.status != "supervisor_error"
            or not result.cleanup_verified
            or self.best_graph is None
            or self.best_values is None
        ):
            return None
        recovered = self.finish(fallback_graph, deviation=deviation)
        if recovered is None:
            return None
        self.recovered_failure = True
        return recovered


class TuningIsolationInstallation:
    def __init__(self, spec, out_path):
        if spec.get("tuning_evaluation_policy") != POLICY:
            raise ValueError("unsupported tuning evaluation isolation policy")
        if os.environ.get("FEDOTLLM_TUNING_ISOLATION_CHILD") == "1":
            raise RuntimeError("recursive tuning isolation installation refused")
        self.runner = ObjectiveProcessRunner(
            Path(out_path).with_suffix(".tuning-isolation"),
            fold_seconds=float(spec["max_pipeline_fit_time_seconds"]),
            candidate_seconds=float(spec["max_pipeline_evaluation_time_seconds"]),
        )
        self.sessions = []
        self._original_build = None
        self.runtime = {}

    def __enter__(self):
        import hashlib
        import inspect
        from fedot.core.pipelines.tuning import tuner_builder
        from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
        from golem.core.optimisers.objective.objective import to_fitness
        from golem.core.tuning.simultaneous import SimultaneousTuner

        evaluator_type = tuner_builder.PipelineObjectiveEvaluate
        required = {"objective", "data_producer", "time_constraint", "eval_n_jobs"}
        if not required.issubset(inspect.signature(evaluator_type.__init__).parameters):
            raise RuntimeError("unsupported FEDOT PipelineObjectiveEvaluate interface")
        if getattr(
            tuner_builder.TunerBuilder.build, "_controller_tuning_isolation", False
        ):
            raise RuntimeError("tuning isolation is already installed")
        self.runtime = {
            "evaluator_file_sha256": hashlib.sha256(
                Path(inspect.getfile(evaluator_type)).read_bytes()
            ).hexdigest(),
            "builder_file_sha256": hashlib.sha256(
                Path(tuner_builder.__file__).read_bytes()
            ).hexdigest(),
        }
        if (
            self.runtime["evaluator_file_sha256"] not in _REVIEWED_EVALUATOR_SOURCES
            or self.runtime["builder_file_sha256"] not in _REVIEWED_BUILDER_SOURCES
        ):
            raise RuntimeError("unsupported FEDOT evaluator or tuner builder source")
        self._module = tuner_builder
        self._original_build = tuner_builder.TunerBuilder.build
        installation = self
        original_build = self._original_build

        def build(builder, data):
            if builder.tuner_class is not SimultaneousTuner:
                raise RuntimeError(
                    "isolated tuning supports the validated SimultaneousTuner only"
                )
            if builder.timeout is None:
                raise RuntimeError(
                    "isolated tuning requires a finite native tuning timeout"
                )
            session = TuningSession(
                installation.runner, budget_seconds=builder.timeout.total_seconds()
            )
            installation.sessions.append(session)
            captured = []

            class IsolatedObjective(ObjectiveEvaluate):
                def __init__(self, *args, **kwargs):
                    delegate = evaluator_type(*args, **kwargs)
                    if not all(
                        hasattr(delegate, field)
                        for field in ("_data_producer", "_objective", "evaluate")
                    ):
                        raise RuntimeError("unsupported FEDOT objective instance")
                    if delegate._objective.is_multi_objective:
                        raise RuntimeError(
                            "isolated SimultaneousTuner requires a single objective"
                        )
                    self.delegate = delegate
                    super().__init__(
                        delegate._objective, eval_n_jobs=delegate._eval_n_jobs
                    )
                    captured.append(self)

                def evaluate(self, graph):
                    self.delegate._eval_n_jobs = self._eval_n_jobs
                    result = session.evaluate(self.delegate, graph)
                    return to_fitness(result.values, False)

            # TunerBuilder resolves this module-global only during construction.
            # The original evaluator stays unmodified and is serialized to child.
            old_evaluator = tuner_builder.PipelineObjectiveEvaluate
            tuner_builder.PipelineObjectiveEvaluate = IsolatedObjective
            try:
                tuner = original_build(builder, data)
            finally:
                tuner_builder.PipelineObjectiveEvaluate = old_evaluator
            if len(captured) != 1 or not hasattr(tuner, "deviation"):
                raise RuntimeError("unsupported FEDOT tuner construction")
            original_tune = tuner.tune

            def tune(graph, **kwargs):
                session.start()
                try:
                    result = original_tune(graph, **kwargs)
                except RuntimeError:
                    result = session.recover_after_failure(
                        graph, deviation=tuner.deviation
                    )
                    if result is None:
                        raise
                if session.restored_best:
                    tuner.obtained_metric = session.best_values[0]
                    from ._tuning_isolation_process import event

                    event(
                        installation.runner.directory,
                        "retained_completed_candidate",
                        values=list(session.best_values),
                        recovered_failure=session.recovered_failure,
                        failure_status=session.failure,
                    )
                else:
                    result = session.finish(result, deviation=tuner.deviation)
                return result

            tuner.tune = tune
            return tuner

        build._controller_tuning_isolation = True
        tuner_builder.TunerBuilder.build = build
        return self

    def diagnostics(self):
        from collections import Counter

        counts = dict(Counter(result.status for result in self.runner.results))

        def recovered(result):
            return any(
                session.recovered_failure and session.last_result is result
                for session in self.sessions
            )

        failures = [
            {
                "status": result.status,
                "error": result.error,
                "cleanup_verified": result.cleanup_verified,
                "completed_folds": result.completed_folds,
                "evaluation_directory": result.evaluation_directory,
                "recovered_by_completed_candidate": recovered(result),
            }
            for result in self.runner.results
            if not result.cleanup_verified or result.status in _FATAL_STATUSES
        ]
        observed = [
            result for result in self.runner.results if result.worker_pid is not None
        ]
        kernel_containment = (
            all(
                result.containment is not None
                and result.containment.get("kernel_group_containment") is True
                for result in observed
            )
            if observed
            else None
        )
        return {
            "policy": POLICY,
            "runtime": self.runtime,
            "events_directory": str(self.runner.directory),
            "containment_modes": sorted(
                {
                    result.containment["process_group_guard"]
                    for result in self.runner.results
                    if result.containment is not None
                }
            ),
            "kernel_group_containment": kernel_containment,
            "platform": sys.platform,
            "evaluations": len(self.runner.results),
            "completed": counts.get("complete", 0),
            "statuses": counts,
            "failures": failures,
            "sessions": len(self.sessions),
            "retained_best": sum(session.restored_best for session in self.sessions),
            "cleanup_verified": all(
                result.cleanup_verified for result in self.runner.results
            ),
        }

    def __exit__(self, exc_type, exc, traceback):
        if self._original_build is not None:
            self._module.TunerBuilder.build = self._original_build
        from ._tuning_isolation_process import atomic_json

        atomic_json(self.runner.directory / "summary.json", self.diagnostics())
        failures = [
            result
            for result in self.runner.results
            if (not result.cleanup_verified or result.status in _FATAL_STATUSES)
            and not any(
                session.recovered_failure and session.last_result is result
                for session in self.sessions
            )
        ]
        if failures:
            details = "; ".join(
                f"{result.status}: {str(result.error or 'no supervisor detail')[:500]}"
                for result in failures[:5]
            )
            raise RuntimeError(
                "tuning isolation failed closed; see tuning-isolation/summary.json"
                + (f" (cause: {details})" if details else "")
            )
        return False


def install_tuning_isolation(spec, out_path):
    """Return a reversible installation; use it around ``Fedot.fit``.

    ``spec`` supplies tuning_evaluation_policy and the already-validated
    max_pipeline_fit_time_seconds / max_pipeline_evaluation_time_seconds.
    Native TunerBuilder.timeout supplies the shared remaining tuning allowance.
    """
    return TuningIsolationInstallation(spec, out_path)
