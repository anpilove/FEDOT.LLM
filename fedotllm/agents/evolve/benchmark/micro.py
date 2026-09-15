"""Fast, private correctness controls for EvolveAgent development.

The model-facing view contains a failing public contract and a single source
file.  Oracle symbols and controller-owned probes remain in this module and
must never be appended to Scout, Verifier, or Fixer prompts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fedotllm.agents.evolve.execution.checkout import source_fingerprint
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.types import SnippetResult

StageName = Literal[
    "stock_probe", "localization", "verification", "fixer", "behavior_probe"
]
StageStatus = Literal["passed", "failed", "not_run"]

_STAGE_ORDER: tuple[StageName, ...] = (
    "stock_probe",
    "localization",
    "verification",
    "fixer",
    "behavior_probe",
)
_OBSERVATION_PREFIX = "EVOLVE_OBSERVATION="

_COMMON_DATA = """import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

rng = np.random.default_rng(42)
n = 120
x = rng.normal(size=(n, 6))
train_data = InputData(
    idx=np.arange(n),
    features=x,
    target=(x[:, 0] + 0.5 * x[:, 1] > 0).astype(int).reshape(-1, 1),
    task=Task(TaskTypesEnum.classification),
    data_type=DataTypesEnum.table,
)
t = np.arange(80)
series = np.sin(t / 7.0) * 10 + t * 0.05 + rng.normal(size=80) * 0.2
ts_data = InputData(
    idx=t,
    features=series,
    target=series,
    task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=5)),
    data_type=DataTypesEnum.ts,
)
"""


@dataclass(frozen=True)
class MicroCasePrompt:
    """The complete case payload that may be shown to a model."""

    case_id: str
    file_path: str
    symptom: str


@dataclass(frozen=True)
class MicroCaseOracle:
    """Private evaluator data. Never serialize this into model context."""

    symbols: tuple[str, ...]
    stock_observation: dict[str, Any]
    patched_observation: dict[str, Any]


@dataclass(frozen=True)
class MicroCase:
    prompt: MicroCasePrompt
    oracle: MicroCaseOracle
    behavior_probe: str


@dataclass(frozen=True)
class MicroStageResult:
    stage: StageName
    status: StageStatus
    reason: str = ""
    observation: dict[str, Any] | None = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass
class MicroCaseResult:
    case_id: str
    stages: list[MicroStageResult] = field(default_factory=list)

    def add(self, result: MicroStageResult) -> None:
        expected_index = len(self.stages)
        if expected_index >= len(_STAGE_ORDER):
            raise ValueError("all microbenchmark stages are already recorded")
        expected = _STAGE_ORDER[expected_index]
        if result.stage != expected:
            raise ValueError(f"expected stage {expected!r}, got {result.stage!r}")
        self.stages.append(result)

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(stage.passed for stage in self.stages)


@dataclass(frozen=True)
class MicroFastResult:
    source_hash: str
    cases: tuple[MicroCaseResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(case.ok for case in self.cases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": "micro-fast",
            "ok": self.ok,
            "source_hash": self.source_hash,
            "cases": {
                case.case_id: {
                    "ok": case.ok,
                    "stages": [
                        {
                            "stage": stage.stage,
                            "status": stage.status,
                            "reason": stage.reason,
                            "observation": stage.observation,
                            "duration_s": stage.duration_s,
                        }
                        for stage in case.stages
                    ],
                }
                for case in self.cases
            },
        }


def _probe(body: str) -> str:
    return _COMMON_DATA + "\n" + body.strip() + "\n"


_CASES: tuple[MicroCase, ...] = (
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="partial_poly_params",
            file_path=(
                "fedot/core/operations/evaluation/operation_implementations/"
                "data_operations/sklearn_transformations.py"
            ),
            symptom=(
                "A public PipelineBuilder pipeline with poly_features(degree=3) and logit "
                "must fit when only the declared degree parameter is supplied."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("PolyFeaturesImplementation.__init__",),
            stock_observation={"outcome": "InvalidParameterError"},
            patched_observation={"outcome": "fit_ok"},
        ),
        behavior_probe=_probe(
            """from fedot.core.pipelines.pipeline_builder import PipelineBuilder

try:
    pipeline = (PipelineBuilder()
                .add_node("poly_features", params={"degree": 3})
                .add_node("logit")
                .build())
    pipeline.fit(train_data)
except Exception as exc:
    outcome = type(exc).__name__
else:
    outcome = "fit_ok"
print("EVOLVE_OBSERVATION=" + json.dumps({"outcome": outcome}, sort_keys=True))""".replace(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder",
                "import json\nfrom fedot.core.pipelines.pipeline_builder import PipelineBuilder",
            )
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="lda_effective_solver",
            file_path=(
                "fedot/core/operations/evaluation/operation_implementations/"
                "models/discriminant_analysis.py"
            ),
            symptom=(
                "A public PipelineBuilder lda operation with shrinkage=0.5 must fit "
                "when the caller leaves the solver at its default."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("LDAImplementation.check_and_correct_params",),
            stock_observation={"outcome": "NotImplementedError"},
            patched_observation={"outcome": "fit_ok"},
        ),
        behavior_probe=_probe(
            """import json
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

try:
    pipeline = PipelineBuilder().add_node("lda", params={"shrinkage": 0.5}).build()
    pipeline.fit(train_data)
except Exception as exc:
    outcome = type(exc).__name__
else:
    outcome = "fit_ok"
print("EVOLVE_OBSERVATION=" + json.dumps({"outcome": outcome}, sort_keys=True))"""
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="lagged_reproducibility",
            file_path=(
                "fedot/core/operations/evaluation/operation_implementations/"
                "data_operations/ts_transformations.py"
            ),
            symptom=(
                "Repeated identical lagged(window_size=252) fits on a short time series "
                "must choose one legal effective window deterministically."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("LaggedImplementation._check_and_correct_window_size",),
            stock_observation={"deterministic": False, "legal": True},
            patched_observation={"deterministic": True, "legal": True},
        ),
        behavior_probe=_probe(
            """import json
import fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations as ts_impl
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

draws = iter((0.10, 0.25, 0.40, 0.55, 0.70, 0.85))
ts_impl.random = lambda: next(draws)
windows = []
for _ in range(6):
    pipeline = (PipelineBuilder()
                .add_node("lagged", params={"window_size": 252})
                .add_node("ridge")
                .build())
    pipeline.fit(ts_data)
    windows.append(next(node for node in pipeline.nodes if node.name == "lagged")
                   .parameters["window_size"])
max_allowed = len(ts_data.features) - ts_data.task.task_params.forecast_length - 1
observation = {
    "deterministic": len(set(windows)) == 1,
    "legal": all(1 <= value <= max_allowed for value in windows),
}
print("EVOLVE_OBSERVATION=" + json.dumps(observation, sort_keys=True))"""
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="polyfit_parameter_identity",
            file_path=(
                "fedot/core/operations/evaluation/operation_implementations/"
                "models/ts_implementations/poly.py"
            ),
            symptom=(
                "Fitting polyfit(degree=99) may correct its effective degree, but must "
                "not change PipelineNode.descriptive_id used by OperationsCache."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=(
                "PolyfitImplementation._correct_degree",
                "PolyfitImplementation.degree",
            ),
            stock_observation={"effective_degree": 3, "id_stable": False},
            patched_observation={"effective_degree": 3, "id_stable": True},
        ),
        behavior_probe=_probe(
            """import json
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

pipeline = PipelineBuilder().add_node("polyfit", params={"degree": 99}).build()
before = pipeline.root_node.descriptive_id
pipeline.fit(ts_data)
after = pipeline.root_node.descriptive_id
observation = {
    "effective_degree": pipeline.root_node.fitted_operation.degree,
    "id_stable": before == after,
}
print("EVOLVE_OBSERVATION=" + json.dumps(observation, sort_keys=True))"""
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="nonfinite_target_preprocessing",
            file_path="fedot/preprocessing/preprocessing.py",
            symptom=(
                "Fit-time obligatory preprocessing must handle numeric +inf/-inf "
                "targets consistently with NaN targets instead of passing them to "
                "an estimator."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("DataPreprocessor._prepare_obligatory_unimodal",),
            stock_observation={"outcome": "ValueError"},
            patched_observation={"outcome": "fit_ok", "finite_prediction": True},
        ),
        behavior_probe=_probe(
            """import json
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

x_local = rng.normal(size=(60, 3))
y_local = x_local[:, 0] * 2 + 1
y_local[3] = np.inf
data = InputData(
    idx=np.arange(60),
    features=x_local,
    target=y_local.reshape(-1, 1),
    task=Task(TaskTypesEnum.regression),
    data_type=DataTypesEnum.table,
)
try:
    pipeline = PipelineBuilder().add_node("scaling").add_node("lasso").build()
    pipeline.fit(data)
    prediction = pipeline.predict(data)
except Exception as exc:
    observation = {"outcome": type(exc).__name__}
else:
    observation = {
        "outcome": "fit_ok",
        "finite_prediction": bool(np.all(np.isfinite(prediction.predict))),
    }
print("EVOLVE_OBSERVATION=" + json.dumps(observation, sort_keys=True))"""
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="merge_parent_index_alignment",
            file_path="fedot/core/data/merge/data_merger.py",
            symptom=(
                "Merging table outputs with the same unique sample ids in different "
                "row orders must join every parent prediction by sample id."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("DataMerger.find_common_predicts",),
            stock_observation={"aligned": False},
            patched_observation={"aligned": True},
        ),
        behavior_probe=_probe(
            """import json
from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData

def output(idx, values, main):
    idx = np.asarray(idx)
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return OutputData(
        idx=idx,
        features=values,
        predict=values,
        target=idx.reshape(-1, 1),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(is_main_target=main),
    )

left = output([0, 1, 2], [10, 11, 12], True)
right = output([2, 1, 0], [102, 101, 100], False)
merged = DataMerger.get([left, right]).merge()
aligned = np.array_equal(merged.features[:, 1], 100 + merged.idx)
print("EVOLVE_OBSERVATION=" + json.dumps({"aligned": bool(aligned)}, sort_keys=True))"""
        ),
    ),
    MicroCase(
        prompt=MicroCasePrompt(
            case_id="single_column_multits_lagged",
            file_path=(
                "fedot/core/operations/evaluation/operation_implementations/"
                "data_operations/ts_transformations.py"
            ),
            symptom=(
                "Lagged transformation of a valid one-column multi_ts input must "
                "produce a two-dimensional trajectory table."
            ),
        ),
        oracle=MicroCaseOracle(
            symbols=("LaggedImplementation._apply_transformation_for_fit",),
            stock_observation={"outcome": "ValueError"},
            patched_observation={"outcome": "fit_ok", "shape": [7, 2]},
        ),
        behavior_probe=_probe(
            """import json
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import LaggedTransformationImplementation
from fedot.core.operations.operation_parameters import OperationParameters

values = np.arange(10, dtype=float).reshape(-1, 1)
data = InputData(
    idx=np.arange(10),
    features=values,
    target=values,
    task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
    data_type=DataTypesEnum.multi_ts,
)
operation = LaggedTransformationImplementation(OperationParameters(window_size=2))
try:
    output = operation.transform_for_fit(data)
except Exception as exc:
    observation = {"outcome": type(exc).__name__}
else:
    observation = {"outcome": "fit_ok", "shape": list(output.predict.shape)}
print("EVOLVE_OBSERVATION=" + json.dumps(observation, sort_keys=True))"""
        ),
    ),
)


def micro_cases() -> tuple[MicroCase, ...]:
    """Return immutable private case specifications."""

    return _CASES


def model_facing_context(case: MicroCase) -> str:
    """Render the only case metadata that an LLM stage may receive."""

    return (
        f"Correctness case: {case.prompt.case_id}\n"
        f"Allowed FEDOT source file: {case.prompt.file_path}\n"
        f"Observed public contract: {case.prompt.symptom}\n"
        "Locate and explain the responsible source mechanism in this file."
    )


def parse_observation(result: SnippetResult) -> tuple[dict[str, Any] | None, str]:
    """Parse exactly one JSON observation from a controller-owned probe."""

    if result.status != "ok":
        return None, f"probe status is {result.status}: {result.detail}"
    rows = [
        line[len(_OBSERVATION_PREFIX) :]
        for line in result.stdout.splitlines()
        if line.startswith(_OBSERVATION_PREFIX)
    ]
    if len(rows) != 1:
        return None, f"expected one observation, got {len(rows)}"
    try:
        payload = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        return None, f"observation is not JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "observation must be a JSON object"
    return payload, ""


def validate_observation(
    case: MicroCase,
    result: SnippetResult,
    *,
    patched: bool,
) -> MicroStageResult:
    """Validate probe output against the private semantic postcondition."""

    observation, error = parse_observation(result)
    expected = (
        case.oracle.patched_observation if patched else case.oracle.stock_observation
    )
    passed = observation == expected and not error
    phase = "patched" if patched else "stock"
    return MicroStageResult(
        stage="behavior_probe" if patched else "stock_probe",
        status="passed" if passed else "failed",
        reason=error
        or ("" if passed else f"{phase} observation did not match contract"),
        observation=observation,
        duration_s=result.duration_s,
    )


def run_stock_microbenchmark(source: Path) -> MicroFastResult:
    """Run the deterministic, zero-LLM preflight for all correctness cases."""

    source = source.resolve()
    before = source_fingerprint(source)
    rows: list[MicroCaseResult] = []
    for case in micro_cases():
        row = MicroCaseResult(case.prompt.case_id)
        row.add(
            validate_observation(
                case,
                run_fedot_snippet(source, case.behavior_probe),
                patched=False,
            )
        )
        rows.append(row)
    if source_fingerprint(source) != before:
        raise RuntimeError("stock microbenchmark mutated the frozen source")
    return MicroFastResult(source_hash=before, cases=tuple(rows))
