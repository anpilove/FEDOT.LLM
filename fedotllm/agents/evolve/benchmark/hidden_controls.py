"""Independent controls for production acceptance on localized public contracts.

These exposed fixtures are regression cases, not fresh hidden evidence. Private
mutation/probe details never participate in the production acceptance decision.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fedotllm.agents.evolve.agents.failures import classify_model_failure
from fedotllm.agents.evolve.benchmark.micro_agent import (
    Architecture,
    _Calls,
    _usage_requests,
)
from fedotllm.agents.evolve.benchmark.micro_discovery import FileSelection
from fedotllm.agents.evolve.discovery.navigation import (
    architecture_cards,
    inspect_shortlist,
)
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.types import (
    EvolveRunPolicy, PatchCandidate, PatchEdit, MatchSite, SnippetResult,
)


@dataclass(frozen=True)
class HiddenControl:
    control_id: str
    component: str
    symptom: str
    public_contract: str
    file_path: str
    old_code: str
    defective_code: str
    responsible_symbol: str
    probe: str


class FileShortlist(BaseModel):
    selected_indices: list[int] = Field(default_factory=list, min_length=1, max_length=3)
    rationale: str = ""


_COMMON_IMPORTS = """import json
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
"""


_CONTROLS: tuple[HiddenControl, ...] = (
    HiddenControl(
        control_id="heldout-a17",
        component="data operation",
        symptom=(
            "The public scaling operation no longer centers continuous columns at "
            "zero or gives them unit variance."
        ),
        public_contract=(
            "After fitting, the public scaling operation centers each continuous "
            "column at zero and scales it to unit variance."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/sklearn_transformations.py"
        ),
        old_code="self.operation = StandardScaler(**self.params.to_dict())",
        defective_code="self.operation = MinMaxScaler(**self.params.to_dict())",
        responsible_symbol="ScalingImplementation",
        probe=_COMMON_IMPORTS
        + """
rng = np.random.default_rng(17)
x = rng.normal(loc=[10., 100.], scale=[2., 20.], size=(40, 2))
data = InputData(idx=np.arange(len(x)), features=x,
                 target=(x[:, 0] > 10).astype(int).reshape(-1, 1),
                 task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
pipeline = PipelineBuilder().add_node('scaling').build()
pipeline.fit(data)
values = np.asarray(pipeline.predict(data).predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'mean': np.round(values.mean(axis=0), 6).tolist(),
    'std': np.round(values.std(axis=0), 6).tolist(),
    'shape': list(values.shape)}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="heldout-b42",
        component="classification evaluation",
        symptom=(
            "A binary classifier requested in probability mode crashes while "
            "extracting the positive-class probability."
        ),
        public_contract=(
            "Probability mode returns one finite positive-class probability per "
            "input row without raising an exception."
        ),
        file_path="fedot/core/operations/evaluation/evaluation_interfaces.py",
        old_code="prediction = trained_operation.predict_proba(features)",
        defective_code="prediction = trained_operation.predict(features)",
        responsible_symbol="SkLearnEvaluationStrategy._sklearn_compatible_prediction",
        probe=_COMMON_IMPORTS
        + """
x = np.array([[-2.], [-1.], [1.], [2.]])
data = InputData(idx=np.arange(4), features=x,
                 target=np.array([[0], [0], [1], [1]]),
                 task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
pipeline = PipelineBuilder().add_node('logit').build()
pipeline.fit(data)
values = np.asarray(pipeline.predict(data, output_mode='probs').predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'values': np.round(values, 6).tolist(), 'shape': list(values.shape)}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="heldout-c93",
        component="time-series transformation",
        symptom=(
            "A lagged time-series pipeline returns one extra forecast value compared "
            "with the public forecast_length contract."
        ),
        public_contract=(
            "A fitted time-series pipeline returns exactly forecast_length finite "
            "forecast values."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/ts_transformations.py"
        ),
        old_code="""        new_input_data = copy(input_data)
        forecast_length = new_input_data.task.task_params.forecast_length

        # Correct window size parameter
        self._check_and_correct_window_size(new_input_data.features, forecast_length)

        target = np.array(new_input_data.target)""",
        defective_code="""        new_input_data = copy(input_data)
        forecast_length = new_input_data.task.task_params.forecast_length + 1

        # Correct window size parameter
        self._check_and_correct_window_size(new_input_data.features, forecast_length)

        target = np.array(new_input_data.target)""",
        responsible_symbol="LaggedImplementation.transform_for_fit",
        probe=_COMMON_IMPORTS
        + """
t = np.arange(30)
series = np.sin(t / 3) + t * 0.1
data = InputData(idx=t, features=series, target=series,
                 task=Task(TaskTypesEnum.ts_forecasting,
                           TsForecastingParams(forecast_length=3)),
                 data_type=DataTypesEnum.ts)
pipeline = (PipelineBuilder().add_node('lagged', params={'window_size': 4})
            .add_node('ridge').build())
pipeline.fit(data)
values = np.asarray(pipeline.predict(data).predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'finite': bool(np.isfinite(values).all()), 'shape': list(values.shape)}, sort_keys=True))
""",
    ),
)


_FRESH_CONTROLS: tuple[HiddenControl, ...] = (
    HiddenControl(
        control_id="prospective-d14",
        component="normalization data operation",
        symptom=(
            "The public normalization operation no longer maps each fitted "
            "continuous feature to the closed interval from zero to one."
        ),
        public_contract=(
            "On its fitted training table, normalization returns finite columns "
            "whose minimum is zero and maximum is one."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/sklearn_transformations.py"
        ),
        old_code="self.operation = MinMaxScaler(**self.params.to_dict())",
        defective_code="self.operation = StandardScaler(**self.params.to_dict())",
        responsible_symbol="NormalizationImplementation",
        probe=_COMMON_IMPORTS
        + """
rng = np.random.default_rng(17)
x = rng.normal(loc=[10., 100.], scale=[2., 20.], size=(40, 2))
data = InputData(idx=np.arange(len(x)), features=x,
                 target=(x[:, 0] > 10).astype(int).reshape(-1, 1),
                 task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
pipeline = PipelineBuilder().add_node('normalization').build()
pipeline.fit(data)
values = np.asarray(pipeline.predict(data).predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'min': np.round(values.min(axis=0), 6).tolist(),
    'max': np.round(values.max(axis=0), 6).tolist(),
    'shape': list(values.shape)}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="prospective-e28",
        component="categorical data operation",
        symptom=(
            "A fitted public one-hot encoder crashes when prediction data contains "
            "a category that was absent from its training data."
        ),
        public_contract=(
            "The default fitted one-hot operation transforms unseen prediction "
            "categories into a finite matrix with one output row per input row."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/categorical_encoders.py"
        ),
        old_code="'handle_unknown': 'ignore'",
        defective_code="'handle_unknown': 'error'",
        responsible_symbol="OneHotEncodingImplementation",
        probe=_COMMON_IMPORTS
        + """
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import OneHotEncodingImplementation
train = InputData(
    idx=np.arange(4),
    features=np.array([['red', 1.], ['blue', 2.], ['red', 3.], ['blue', 4.]], dtype=object),
    target=np.array([[0], [1], [0], [1]]),
    task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table,
    categorical_idx=np.array([0]))
prediction_features = np.array([['green', 5.], ['red', 6.]], dtype=object)
operation = OneHotEncodingImplementation()
operation.fit(train)
values = np.asarray(operation._apply_one_hot_encoding(prediction_features), dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'finite': bool(np.isfinite(values).all()), 'rows': len(values),
    'shape': list(values.shape)}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="prospective-f61",
        component="time-series model",
        symptom=(
            "The public naive-average forecast uses the oldest configured history "
            "window instead of the most recent window."
        ),
        public_contract=(
            "The naive-average operation forecasts the arithmetic mean of the most "
            "recent configured fraction of the observed series."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "models/ts_implementations/naive.py"
        ),
        old_code="mean_value = np.nanmean(input_data.features[-window:])",
        defective_code="mean_value = np.nanmean(input_data.features[:window])",
        responsible_symbol="NaiveAverageForecastImplementation.predict",
        probe=_COMMON_IMPORTS
        + """
series = np.array([100., 100., 100., 100., 1., 2., 3., 4.])
data = InputData(
    idx=np.arange(len(series)), features=series, target=series,
    task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=3)),
    data_type=DataTypesEnum.ts)
pipeline = (PipelineBuilder()
            .add_node('ts_naive_average', params={'part_for_averaging': 0.5})
            .build())
pipeline.fit(data)
values = np.asarray(pipeline.predict(data).predict, dtype=float).reshape(-1)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'values': np.round(values, 6).tolist(), 'shape': list(values.shape)},
    sort_keys=True))
""",
    ),
)


_FRESH_V2_CONTROLS: tuple[HiddenControl, ...] = (
    HiddenControl(
        control_id="prospective-g07",
        component="numeric missing-value imputation",
        symptom=(
            "The default public simple-imputation operation fills missing numeric "
            "values with zero instead of the fitted column statistic."
        ),
        public_contract=(
            "With default parameters, simple numeric imputation replaces each "
            "missing value with the mean learned from that feature column."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/sklearn_transformations.py"
        ),
        old_code="self.params_num = self.params.to_dict()",
        defective_code=(
            "self.params_num = {**self.params.to_dict(), 'strategy': 'constant', "
            "'fill_value': 0}"
        ),
        responsible_symbol="ImputationImplementation",
        probe=_COMMON_IMPORTS
        + """
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import ImputationImplementation
x = np.array([[1., 10.], [np.nan, 20.], [5., np.nan], [7., 40.]])
data = InputData(
    idx=np.arange(len(x)), features=x,
    target=np.array([[0], [1], [0], [1]]),
    task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table,
    numerical_idx=np.array([0, 1]))
operation = ImputationImplementation()
operation.fit(data)
values = np.asarray(operation.transform(data).predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'values': np.round(values, 6).tolist(),
    'finite': bool(np.isfinite(values).all())}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="prospective-h31",
        component="full probability classification output",
        symptom=(
            "For a binary classifier, public full-probability mode returns only "
            "one class column instead of the complete two-class matrix."
        ),
        public_contract=(
            "Binary classification in full_probs mode returns one finite "
            "probability column per class and every row sums to one."
        ),
        file_path="fedot/core/operations/evaluation/classification.py",
        old_code=(
            "elif n_classes == 2 and self.output_mode != 'full_probs' and "
            "len(prediction.shape) > 1:"
        ),
        defective_code=(
            "elif n_classes == 2 and self.output_mode == 'full_probs' and "
            "len(prediction.shape) > 1:"
        ),
        responsible_symbol="SkLearnClassificationStrategy.predict",
        probe=_COMMON_IMPORTS
        + """
x = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
data = InputData(
    idx=np.arange(len(x)), features=x,
    target=np.array([[0], [0], [0], [1], [1], [1]]),
    task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
pipeline = PipelineBuilder().add_node('knn', params={'n_neighbors': 3}).build()
pipeline.fit(data)
values = np.asarray(pipeline.predict(data, output_mode='full_probs').predict, dtype=float)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'shape': list(values.shape), 'finite': bool(np.isfinite(values).all()),
    'row_sums': np.round(values.sum(axis=1), 6).tolist()
    if values.ndim == 2 else []}, sort_keys=True))
""",
    ),
    HiddenControl(
        control_id="prospective-i52",
        component="configured Gaussian time-series filter",
        symptom=(
            "The public Gaussian-filter operation ignores its positive sigma and "
            "returns an impulse time series unchanged instead of smoothing it."
        ),
        public_contract=(
            "A Gaussian filter configured with positive sigma spreads an isolated "
            "finite impulse over neighboring time steps while preserving shape."
        ),
        file_path=(
            "fedot/core/operations/evaluation/operation_implementations/"
            "data_operations/ts_transformations.py"
        ),
        old_code="smoothed_ts = gaussian_filter(source_ts, sigma=sigma)",
        defective_code="smoothed_ts = gaussian_filter(source_ts, sigma=0)",
        responsible_symbol="GaussianFilterImplementation.transform",
        probe=_COMMON_IMPORTS
        + """
series = np.zeros(21, dtype=float)
series[10] = 1.0
data = InputData(
    idx=np.arange(len(series)), features=series, target=series.copy(),
    task=Task(TaskTypesEnum.ts_forecasting,
              TsForecastingParams(forecast_length=3)),
    data_type=DataTypesEnum.ts)
pipeline = PipelineBuilder().add_node('gaussian_filter', params={'sigma': 2}).build()
pipeline.fit(data)
values = np.asarray(pipeline.predict(data).predict, dtype=float).reshape(-1)
print('EVOLVE_OBSERVATION=' + json.dumps({
    'shape': list(values.shape), 'center': round(float(values[10]), 6),
    'neighbor_mass': round(float(values[8:13].sum() - values[10]), 6),
    'unchanged': bool(np.array_equal(values, series))}, sort_keys=True))
""",
    ),
)


_CATALOG: tuple[str, ...] = (
    "fedot/preprocessing/preprocessing.py",
    "fedot/core/data/merge/data_merger.py",
    "fedot/core/pipelines/node.py",
    "fedot/core/operations/operation.py",
    "fedot/core/operations/evaluation/common_preprocessing.py",
    "fedot/core/operations/evaluation/classification.py",
    "fedot/core/operations/evaluation/evaluation_interfaces.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py",
    "fedot/core/operations/evaluation/operation_implementations/data_operations/ts_transformations.py",
    "fedot/core/operations/evaluation/time_series.py",
    "fedot/core/data/data.py",
    "fedot/core/operations/operation_parameters.py",
)

_FRESH_CATALOG: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py",
            "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/naive.py",
            *_CATALOG,
        )
    )
)

_FRESH_V2_CATALOG: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "fedot/core/operations/evaluation/classification.py",
            *_FRESH_CATALOG,
        )
    )
)


def _observation(result: SnippetResult) -> str | None:
    if result.status != "ok":
        return None
    values = [
        line.partition("=")[2].strip()
        for line in result.stdout.splitlines()
        if line.startswith("EVOLVE_OBSERVATION=")
    ]
    return values[0] if len(values) == 1 else None


def _cards(source: Path, catalog: tuple[str, ...] = _CATALOG) -> str:
    return architecture_cards(source, catalog, max_chars_per_file=1_500)


class ControllerLeadSelection(FileSelection):
    line: int = Field(ge=1)


def _public_lead(
    source: Path, calls: _Calls, contract: str, observed: SnippetResult,
    catalog: tuple[str, ...],
) -> MatchSite:
    """The same localization protocol for healthy and mutated checkouts."""
    evidence = json.dumps({
        "status": observed.status,
        "stdout_tail": observed.stdout[-2_000:],
        "stderr_tail": observed.stderr[-4_000:],
    }, ensure_ascii=False)
    context = (
        f"Investigate whether this public contract is satisfied:\n{contract}\n"
        f"Observed execution (not proof of a defect):\n{evidence}\n"
        "Select the source owning this behavior even if it appears healthy. "
        "Do not assume a repair is necessary. The verifier will decide.\n"
    )
    shortlist = calls.create(
        context + "Choose up to three zero-based catalog indices.\n" + _cards(source, catalog),
        FileShortlist, stage="hidden_file_localization", metadata={},
    )
    indices = list(dict.fromkeys(
        index for index in shortlist.selected_indices if 0 <= index < len(catalog)
    ))
    if not indices:
        raise ValueError("localizer returned no valid catalog entries")
    selected = calls.create(
        context + "\nInspected source:\n"
        + inspect_shortlist(source, catalog, indices)
        + "\nChoose an original catalog index and an exact source line inside "
        "the relevant class/function. Do not propose a patch.\n"
        + _cards(source, catalog),
        ControllerLeadSelection, stage="hidden_file_navigation", metadata={},
    )
    if not 0 <= selected.selected_index < len(catalog):
        raise ValueError("localizer returned an invalid catalog index")
    return MatchSite(
        "execution", catalog[selected.selected_index], selected.line,
        why=f"Check the public contract; it may already hold: {contract}",
        evidence=(f"Observed public execution: {evidence}",),
        mechanism=selected.mechanism,
        hypothesis_kind="correctness",
    )


def _accepted_decisions(workspace: Path) -> list[dict]:
    """Read decisions BEFORE the independent oracle sees a proposed repair."""
    path = workspace / "journal.jsonl"
    if not path.is_file():
        raise ValueError("main controller did not persist a decision journal")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted = []
    for row in rows:
        # DEV keeps are provisional. FINAL may reject or ablate that patch.
        # Secondary correctness/maintenance/signal tracks end before FINAL.
        secondary = row.get("event") == "decision" and any(
            row.get(flag) for flag in (
                "correctness_keep", "maintenance_keep", "metric_signal_keep", "metric_goal_keep"
            )
        )
        final = row.get("event") == "final" and row.get("keep_final")
        if secondary or final:
            if not row.get("candidate") or not row.get("edits"):
                raise ValueError("accepted decision is missing its replayable patch")
            accepted.append(row)
    return accepted


def _independent_assessment(
    source: Path, workspace: Path, decisions: list[dict], *,
    probe: str, healthy_observation: str, healthy_case: bool,
) -> list[dict]:
    """Oracle results cannot alter the already recorded controller decisions."""
    assessed = []
    for index, decision in enumerate(decisions):
        tree = create_experiment_checkout(
            source, workspace, run_id="independent", candidate_id=f"patch-{index}",
        )
        row = {
            "candidate": decision["candidate"],
            "controller_accepted": True,
            "confirmed_fix": False,
            "false_accept": None,
            "execution_ok": False,
        }
        try:
            candidate = PatchCandidate(
                decision["candidate"],
                edits=[PatchEdit(**edit) for edit in decision["edits"]],
            )
            if not apply_patch(tree, candidate):
                row["reason"] = "accepted patch could not be replayed"
            else:
                checked = run_fedot_snippet(tree, probe)
                observation = _observation(checked)
                evaluable = observation is not None or checked.status == "runtime_error"
                row.update(
                    execution_ok=evaluable,
                    patched_status=checked.status,
                    confirmed_fix=evaluable and not healthy_case
                    and observation == healthy_observation,
                    false_accept=(healthy_case or observation != healthy_observation)
                    if evaluable else None,
                    reason="" if evaluable else "independent probe unavailable",
                )
        finally:
            discard_experiment_checkout(tree, workspace=workspace, source=source)
        assessed.append(row)
    return assessed


def run_hidden_control_benchmark(
    source: Path,
    workspace: Path,
    *,
    inference: Any,
    architecture: Architecture = "staged",
    committee_size: int = 3,
    controls: tuple[HiddenControl, ...] = _CONTROLS,
    catalog: tuple[str, ...] = _CATALOG,
    component_name: str = "hidden-controls",
) -> dict[str, Any]:
    """Exercise production verification/repair/acceptance, then judge privately.

    Localization supplies a source lead, not a verdict, patch, or oracle.
    This measures acceptance on supplied public contracts, not blind discovery.
    The immutable-source manifest check alone is disabled because fixtures are
    intentionally mutated. All behavioral, metric and protection gates remain.
    """
    from fedotllm.agents.evolve.controller.campaign import run_once

    if architecture != "staged":
        raise ValueError("main-controller hidden controls support only staged architecture")
    source, workspace = source.resolve(), workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    before = source_fingerprint(source)
    requests_before = _usage_requests(inference)
    calls = _Calls(inference)
    rows: list[dict] = []
    healthy_rows: list[dict] = []
    run_id = f"hidden-{uuid.uuid4().hex[:8]}"
    for control in controls:
        healthy = run_fedot_snippet(source, control.probe)
        reference = _observation(healthy)
        for healthy_case in (False, True):
            case_id = f"{'healthy' if healthy_case else 'mutation'}-{control.control_id}"
            case_workspace = workspace / run_id / case_id
            tree = create_experiment_checkout(
                source, workspace, run_id=run_id, candidate_id=case_id,
            )
            row = {
                "control_id": case_id, "component": control.component,
                "controller_accepted": False, "accepted": False,
                "confirmed_fix": False, "execution_ok": False,
                "status": "not_evaluated", "assessments": [],
            }
            try:
                if reference is None:
                    raise ValueError("healthy reference probe is not evaluable")
                if not healthy_case:
                    mutation = PatchCandidate(
                        f"private-{control.control_id}",
                        edits=[PatchEdit(control.file_path, control.old_code, control.defective_code)],
                    )
                    if not apply_patch(tree, mutation):
                        raise ValueError("private mutation could not be applied")
                observed = run_fedot_snippet(tree, control.probe)
                if observed.status not in {"ok", "runtime_error"}:
                    raise ValueError(f"fixture probe unavailable: {observed.status}")
                if not healthy_case and _observation(observed) == reference:
                    raise ValueError("private mutation does not reproduce the defect")
                lead = _public_lead(tree, calls, control.public_contract, observed, catalog)
                decision = run_once(
                    checkout=tree, workspace=case_workspace,
                    findings_path=case_workspace / "findings.jsonl",
                    inference=inference, verifier_inference=inference,
                    resume_lead=lead, max_leads=1,
                    policy=EvolveRunPolicy(verify_manifest=False, fedot_quality_jobs=False),
                )
                accepted = _accepted_decisions(case_workspace)
                row.update(
                    controller_accepted=bool(accepted), accepted=bool(accepted),
                    controller_reason=decision.reason,
                    controller_infrastructure_error=decision.infrastructure_error,
                )
                # Nothing from this point is sent back to run_once/inference.
                assessments = _independent_assessment(
                    tree, case_workspace, accepted, probe=control.probe,
                    healthy_observation=reference, healthy_case=healthy_case,
                )
                execution_ok = (
                    not decision.infrastructure_error
                    and all(item["execution_ok"] for item in assessments)
                    and not decision.reason.startswith("verification_inconclusive")
                )
                row.update(
                    assessments=assessments,
                    execution_ok=execution_ok,
                    confirmed_fix=any(item["confirmed_fix"] for item in assessments),
                    status="evaluated" if execution_ok else "inconclusive",
                )
            except Exception as exc:
                failure = classify_model_failure(exc)
                row.update(
                    status="evaluation_error", reason=str(failure)[:1000],
                    infrastructure=failure.infrastructure,
                )
            finally:
                discard_experiment_checkout(tree, workspace=workspace, source=source)
            (healthy_rows if healthy_case else rows).append(row)
    assessed = [item for row in [*rows, *healthy_rows] for item in row["assessments"]]
    confirmed = sum(row["confirmed_fix"] for row in rows)
    false_accepts = sum(item["false_accept"] is True for item in assessed)
    healthy_rejections = sum(
        row["execution_ok"] and not row["accepted"] for row in healthy_rows
    )
    losses = sum(not row["execution_ok"] for row in [*rows, *healthy_rows])
    unchanged = source_fingerprint(source) == before
    target_met = bool(controls) and (
        confirmed == len(controls) and healthy_rejections == len(controls)
        and false_accepts == 0 and losses == 0 and unchanged
    )
    payload = {
        "schema_version": 2, "component": component_name,
        "architecture": "staged",
        "validation_scope": "production_controller_with_public_contract_localization",
        "oracle_used_for_acceptance": False,
        "manifest_check_disabled_for_mutated_fixtures": True,
        "controls_are_regression_cases": True,
        "ok": target_met, "execution_ok": losses == 0 and unchanged,
        "quality_target_met": target_met, "source_unchanged": unchanged,
        "metrics": {
            "autonomous_confirmed_fixes": confirmed,
            "false_accepts": false_accepts,
            "unassessed_accepts": sum(item["false_accept"] is None for item in assessed),
            "healthy_rejections": healthy_rejections,
            "evaluation_losses": losses,
            "infrastructure_failures": sum(
                bool(row.get("infrastructure") or row.get("controller_infrastructure_error"))
                for row in [*rows, *healthy_rows]
            ),
            "provider_requests": max(0, _usage_requests(inference) - requests_before),
        },
        "cases": rows, "healthy_cases": healthy_rows,
    }
    (workspace / f"hidden_controls_{architecture}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return payload


def run_fresh_hidden_control_benchmark(
    source: Path,
    workspace: Path,
    *,
    inference: Any,
    architecture: Architecture = "staged",
    committee_size: int = 3,
) -> dict[str, Any]:
    """Replay the first formerly hidden set as regression controls."""

    return run_hidden_control_benchmark(
        source,
        workspace,
        inference=inference,
        architecture=architecture,
        committee_size=committee_size,
        controls=_FRESH_CONTROLS,
        catalog=_FRESH_CATALOG,
        component_name="hidden-controls-fresh",
    )


def run_fresh_v2_hidden_control_benchmark(
    source: Path,
    workspace: Path,
    *,
    inference: Any,
    architecture: Architecture = "staged",
    committee_size: int = 3,
) -> dict[str, Any]:
    """Replay the second formerly hidden set as regression controls."""

    return run_hidden_control_benchmark(
        source,
        workspace,
        inference=inference,
        architecture=architecture,
        committee_size=committee_size,
        controls=_FRESH_V2_CONTROLS,
        catalog=_FRESH_V2_CATALOG,
        component_name="hidden-controls-fresh2",
    )
