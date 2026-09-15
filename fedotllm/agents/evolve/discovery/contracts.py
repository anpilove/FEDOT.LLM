"""Small public FEDOT contract checks used before speculative metric search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.types import PatchSite, SnippetResult, VerificationResult


CONTRACT_PROBE_MARKER = "controller public contract probe:\n"
CONTRACT_EVIDENCE = "controller_public_contract_probe"


@dataclass(frozen=True)
class PublicContract:
    contract_id: str
    claim: str
    probe: str
    candidate_files: tuple[str, ...]


_COMMON = """import json
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
rng = np.random.default_rng(17)
train_data = InputData(
    idx=np.arange(40), features=rng.normal(size=(40, 3)),
    target=(rng.normal(size=40) > 0).astype(int).reshape(-1, 1),
    task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
ts_values = np.sin(np.arange(30) / 3) + np.arange(30) * 0.1
ts_data = InputData(
    idx=np.arange(30), features=ts_values, target=ts_values,
    task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=3)),
    data_type=DataTypesEnum.ts)
"""


def _probe(body: str) -> str:
    return _COMMON + "\n" + body.strip() + "\n"


_CONTRACTS: tuple[PublicContract, ...] = (
    PublicContract(
        "partial_operation_parameters",
        "A public operation must preserve its declared defaults when a caller supplies only one optional parameter.",
        _probe("""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
try:
    pipeline = (PipelineBuilder().add_node('poly_features', params={'degree': 3})
                .add_node('logit').build())
    pipeline.fit(train_data)
except Exception as exc:
    observation = {'outcome': type(exc).__name__}
else:
    observation = {'outcome': 'fit_ok'}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert observation == {'outcome': 'fit_ok'}, observation"""),
        (
            "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py",
            "fedot/core/operations/operation_parameters.py",
            "fedot/core/operations/hyperparameters_preprocessing.py",
        ),
    ),
    PublicContract(
        "parameter_identity_after_fit",
        "Fit may correct an effective parameter but must not mutate a node identity used by caches.",
        _probe("""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
pipeline = PipelineBuilder().add_node('polyfit', params={'degree': 99}).build()
before = pipeline.root_node.descriptive_id
pipeline.fit(ts_data)
observation = {'id_stable': before == pipeline.root_node.descriptive_id}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert observation['id_stable'], observation"""),
        (
            "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/poly.py",
            "fedot/core/pipelines/node.py",
            "fedot/core/operations/operation_parameters.py",
        ),
    ),
    PublicContract(
        "finite_numeric_target",
        "Obligatory fit preprocessing must not pass non-finite numeric target rows to an estimator.",
        _probe("""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
x = rng.normal(size=(60, 3)); y = x[:, 0] * 2 + 1; y[3] = np.inf
data = InputData(idx=np.arange(60), features=x, target=y.reshape(-1, 1),
                 task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table)
try:
    pipeline = PipelineBuilder().add_node('scaling').add_node('lasso').build()
    pipeline.fit(data); values = np.asarray(pipeline.predict(data).predict)
    observation = {'fit_ok': True, 'finite': bool(np.isfinite(values).all())}
except Exception as exc:
    observation = {'fit_ok': False, 'error': type(exc).__name__}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert observation.get('fit_ok') and observation.get('finite'), observation"""),
        (
            "fedot/preprocessing/preprocessing.py",
            "fedot/core/data/data_preprocessing.py",
            "fedot/preprocessing/data_types.py",
        ),
    ),
    PublicContract(
        "merge_row_identity",
        "Multi-parent table merge must align every parent prediction by sample id.",
        _probe("""from fedot.core.data.data import OutputData
from fedot.core.data.merge.data_merger import DataMerger
from fedot.core.data.supplementary_data import SupplementaryData
def output(idx, values, main):
    idx = np.asarray(idx); values = np.asarray(values, dtype=float).reshape(-1, 1)
    return OutputData(idx=idx, features=values, predict=values,
        target=idx.reshape(-1, 1), task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(is_main_target=main))
merged = DataMerger.get([
    output([0, 1, 2], [10, 11, 12], True),
    output([2, 1, 0], [102, 101, 100], False)]).merge()
observation = {'aligned': bool(np.array_equal(merged.features[:, 1], 100 + merged.idx))}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert observation['aligned'], observation"""),
        (
            "fedot/core/data/merge/data_merger.py",
            "fedot/core/data/merge/supplementary_data_merger.py",
            "fedot/core/data/array_utilities.py",
        ),
    ),
    PublicContract(
        "single_column_multits_shape",
        "A valid one-column multi_ts lagged transform must produce a two-dimensional table.",
        _probe("""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
values = np.arange(10, dtype=float).reshape(-1, 1)
data = InputData(idx=np.arange(10), features=values, target=values,
    task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
    data_type=DataTypesEnum.multi_ts)
try:
    pipeline = (PipelineBuilder().add_node('lagged', params={'window_size': 2})
                .add_node('ridge').build())
    pipeline.fit(data)
    result = pipeline.predict(data)
    prediction = np.asarray(result.predict)
    observation = {'fit_ok': True, 'ndim': int(prediction.ndim),
                   'length': int(prediction.shape[0])}
except Exception as exc:
    observation = {'fit_ok': False, 'error': type(exc).__name__}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert (observation.get('fit_ok') and observation.get('ndim') == 2
        and observation.get('length') == 2), observation"""),
        (
            "fedot/core/operations/evaluation/operation_implementations/data_operations/ts_transformations.py",
            "fedot/core/operations/evaluation/time_series.py",
            "fedot/core/data/multi_modal.py",
        ),
    ),
    PublicContract(
        "predict_state_stability",
        "Repeated predict calls on identical input must not mutate fitted behavior.",
        _probe("""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
pipeline = PipelineBuilder().add_node('scaling').add_node('logit').build()
pipeline.fit(train_data)
first = np.asarray(pipeline.predict(train_data, output_mode='probs').predict)
second = np.asarray(pipeline.predict(train_data, output_mode='probs').predict)
observation = {'stable': bool(np.array_equal(first, second)), 'shape': list(first.shape)}
print('EVOLVE_OBSERVATION=' + json.dumps(observation, sort_keys=True))
assert observation['stable'], observation"""),
        (
            "fedot/core/pipelines/node.py",
            "fedot/core/operations/operation.py",
            "fedot/core/operations/evaluation/evaluation_interfaces.py",
        ),
    ),
)


def public_contracts() -> tuple[PublicContract, ...]:
    return _CONTRACTS


def supports_public_contracts(checkout: Path) -> bool:
    """Return whether *checkout* looks like a complete FEDOT source tree.

    Campaign unit tests and partial source snapshots often contain only the
    file under investigation.  Running public API probes there produces only
    import errors and, more importantly, can make a mocked snippet runner look
    like six real contract failures.  The standalone contract function stays
    usable with an injected runner; the controller calls it only for a source
    tree that can actually execute the public API.
    """

    required = (
        "fedot/core/data/data.py",
        "fedot/core/pipelines/pipeline_builder.py",
        "fedot/core/repository/tasks.py",
    )
    return all((checkout / path).is_file() for path in required)


def _observation(result: SnippetResult) -> str:
    rows = [
        line.partition("=")[2].strip()
        for line in (result.stdout or "").splitlines()
        if line.startswith("EVOLVE_OBSERVATION=")
    ]
    return rows[0] if len(rows) == 1 else ""


def discover_contract_violations(
    checkout: Path,
    *,
    run_fn: Callable[[Path, str], SnippetResult] = run_fedot_snippet,
) -> tuple[list[PatchSite], list[dict]]:
    """Return only observed assertion failures from stable public contracts."""

    leads: list[PatchSite] = []
    rows: list[dict] = []
    for contract in _CONTRACTS:
        result = run_fn(checkout, contract.probe)
        violated = result.status == "runtime_error" and "AssertionError" in (
            result.stderr or ""
        )
        row = {
            "contract_id": contract.contract_id,
            "status": "violated" if violated else "passed" if result.status == "ok" else "invalid",
            "observation": _observation(result),
            "runtime_status": result.status,
            "candidate_files": list(contract.candidate_files),
        }
        rows.append(row)
        if not violated:
            continue
        existing = tuple(path for path in contract.candidate_files if (checkout / path).is_file())
        if not existing:
            continue
        leads.append(
            PatchSite(
                channel="public_contract",
                file_path=existing[0],
                line=1,
                why=contract.claim,
                evidence=(
                    f"observed contract id: {contract.contract_id}",
                    f"observed stock result: {row['observation']}",
                    "candidate contract owners: " + ", ".join(existing[:3]),
                    CONTRACT_PROBE_MARKER + contract.probe,
                ),
                signals=("observed_contract_failure", "public_api"),
                mechanism="The controller reproduced a public functional contract failure.",
                proposed_change="Trace the producer/consumer path and restore the asserted contract.",
                hypothesis_kind="correctness",
            )
        )
    return leads, rows


def verification_from_contract_lead(lead: PatchSite) -> VerificationResult | None:
    """Recover the exact controller-owned stock-failing probe from a lead."""

    code = next(
        (
            item[len(CONTRACT_PROBE_MARKER) :]
            for item in lead.evidence
            if item.startswith(CONTRACT_PROBE_MARKER)
        ),
        "",
    )
    if lead.channel != "public_contract" or not code:
        return None
    return VerificationResult(
        "verified_bug",
        claim=lead.why,
        expected="the public contract assertion passes after the source correction",
        observed=next(
            (item for item in lead.evidence if item.startswith("observed stock result:")),
            "",
        ),
        reproduction_code=code,
        evidence=(CONTRACT_EVIDENCE,),
        detail="Controller-owned public contract probe failed on immutable stock FEDOT.",
    )
