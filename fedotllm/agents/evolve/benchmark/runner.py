from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.evaluation.affected_eval import evaluate_affected_metric
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.evaluation.compare import compare_pack
from fedotllm.agents.evolve.discovery.discover import default_parameter_leads, leads_from_scores
from fedotllm.agents.evolve.agents.fixer import fix_lead
from fedotllm.agents.evolve.evaluation.judge import (
    measure_fedot_tests,
    measure_patched,
    measure_stock,
    normalize_test_result,
    tests_regressed,
    verdict,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.commands.recall import measure_localization
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.execution.smoke import import_error
from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    PatchEdit,
    PatchSite,
    ScoreResult,
    VerificationResult,
)
from fedotllm.agents.evolve.agents.verifier import (
    ContractSupportAudit,
    VerificationProposal,
    _audit_contract_support,
    _grounded_public_exception,
    replay_reproduction,
    verification_context,
    verify_lead,
)

_PCA_FILE = (
    "fedot/core/operations/evaluation/operation_implementations/"
    "data_operations/sklearn_transformations.py"
)
_PCA_OLD = """        self.pca = PCA(**self.params.to_dict())
        self.number_of_features = None
"""
_PCA_NEW = _PCA_OLD + """
    def fit(self, input_data: InputData) -> PCA:
        features = input_data.features
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = np.asarray(features)
        self.number_of_samples, self.number_of_features = features.shape
        cat_idx = input_data.categorical_idx
        if cat_idx is None or np.size(cat_idx) == 0:
            ids = np.arange(self.number_of_features)
        else:
            cat_idx = np.asarray(cat_idx)
            cat_idx = cat_idx[(cat_idx >= 0) & (cat_idx < self.number_of_features)]
            ids = np.setdiff1d(np.arange(self.number_of_features), cat_idx)
            if ids.size == 0:
                ids = np.arange(self.number_of_features)
        self._ids_to_project = ids
        projected = np.nan_to_num(features[:, ids].astype(float), nan=0.0)
        if projected.shape[1] > 1:
            self.check_and_correct_params(is_ts_data=input_data.data_type is DataTypesEnum.ts)
            try:
                self.pca.fit(projected)
            except Exception as exc:
                self.log.info(
                    f"Switched from {type(self.pca).__name__} to default PCA on fit stage due to {exc}"
                )
                self.pca = PCA()
                self.pca.fit(projected)
        else:
            self.pca = None
        return self.pca

    def transform(self, input_data: InputData) -> OutputData:
        features = input_data.features
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        features = np.asarray(features)
        ids = getattr(self, "_ids_to_project", None)
        if ids is None:
            n_cols = features.shape[1]
            cat_idx = input_data.categorical_idx
            if cat_idx is None or np.size(cat_idx) == 0:
                ids = np.arange(n_cols)
            else:
                cat_idx = np.asarray(cat_idx)
                cat_idx = cat_idx[(cat_idx >= 0) & (cat_idx < n_cols)]
                ids = np.setdiff1d(np.arange(n_cols), cat_idx)
                if ids.size == 0:
                    ids = np.arange(n_cols)
        if self.pca is not None:
            projected = np.nan_to_num(features[:, ids].astype(float), nan=0.0)
            transformed = self.pca.transform(projected)
        else:
            transformed = np.nan_to_num(features[:, ids].astype(float), nan=0.0)
        output_data = self._convert_to_output(input_data, transformed)
        n_out = int(np.asarray(transformed).shape[1])
        output_data.categorical_idx = np.array([], dtype=int)
        output_data.numerical_idx = np.arange(n_out)
        output_data.encoded_idx = np.array([], dtype=int)
        output_data.categorical_features = None
        output_data.features_names = None
        return output_data
"""

_RF_DEFAULTS_FILE = "fedot/core/repository/data/default_operation_params.json"
_RF_DEFAULTS_OLD = '''  "rf": {
    "n_jobs": 1
  },
'''
_RF_DEFAULTS_NEW = '''  "rf": {
    "n_jobs": 1,
    "min_samples_leaf": 2
  },
'''

_LAGGED_FILE = (
    "fedot/core/operations/evaluation/operation_implementations/"
    "data_operations/ts_transformations.py"
)
_LAGGED_RANDOM = """        if self.window_size > max_allowed_window_size:
            new = int(random() * max_allowed_window_size)
            new = min(new, max_allowed_window_size)"""
_LAGGED_HAC = """        if self.window_size > max_allowed_window_size:
            selector = WindowSizeSelector(method=WindowSizeSelectorMethodsEnum.HAC, window_range=(5, 60))
            new = int(selector.apply(time_series) * time_series.shape[0] * 0.01)
            new = min(new, max_allowed_window_size)"""


def pca_case() -> PatchCandidate:
    return PatchCandidate(
        candidate_id="pca_numeric_projection",
        edits=[PatchEdit(_PCA_FILE, _PCA_OLD, _PCA_NEW)],
        rationale="Project numeric columns and reset output metadata coherently.",
        contract="fit and transform use the same projection; output metadata matches transformed width",
    )


def rf_leaf2_case() -> PatchCandidate:
    return PatchCandidate(
        candidate_id="rf_default_min_samples_leaf_2",
        edits=[PatchEdit(_RF_DEFAULTS_FILE, _RF_DEFAULTS_OLD, _RF_DEFAULTS_NEW)],
        rationale=(
            "Use a two-sample default leaf for RF to reduce single-row leaf variance; "
            "explicit user parameters still override the repository default."
        ),
        contract="only the default rf path changes; explicit operation parameters remain authoritative",
    )


def lagged_hac_negative_case() -> PatchCandidate:
    """Known invariant fix that the affected metric must reject as unsafe."""

    return PatchCandidate(
        candidate_id="lagged_hac_deterministic_negative_control",
        edits=[PatchEdit(_LAGGED_FILE, _LAGGED_RANDOM, _LAGGED_HAC)],
        rationale="Replace a random oversize-window fallback with deterministic HAC.",
        contract="identical data and parameters choose an identical corrected window",
    )


def benchmark_affected(source: Path, workspace: Path) -> dict:
    """Prove that branch reachability can overturn a broad metric-neutral fix."""

    run_id = f"affected-{uuid.uuid4().hex[:8]}"
    candidate = lagged_hac_negative_case()
    experiment = create_experiment_checkout(
        source,
        workspace,
        run_id=run_id,
        candidate_id=candidate.candidate_id,
    )
    before = source_fingerprint(source)
    try:
        applied = apply_patch(experiment, candidate)
        if not applied:
            return {
                "ok": False,
                "component": "affected",
                "reason": "negative_control_patch_apply_failed",
            }
        verification = VerificationResult(
            "verified_bug",
            claim="oversize lagged correction must be deterministic",
            reproduction_code="""from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import TsForecastingParams
pipeline = PipelineBuilder().add_node('lagged', params={'window_size': 100}).build()
assert pipeline is not None
""",
        )
        lead = PatchSite("benchmark", _LAGGED_FILE, 132)
        result = evaluate_affected_metric(
            source,
            experiment,
            verification,
            lead,
            seed=42,
            max_tasks=2,
        )
        checks = {
            "literal_overlay": result.get("params") == {"window_size": 100},
            "operation_selected": result.get("operation") == "lagged",
            "two_real_workloads": len(result.get("rows") or ()) == 2,
            "lead_reached": all(
                bool(row.get("lead_reached")) for row in result.get("rows") or ()
            ),
            "unsafe_fix_rejected": result.get("status") == "regressed",
            "source_immutable": source_fingerprint(source) == before,
        }
        return {
            "ok": all(checks.values()),
            "component": "affected",
            "checks": checks,
            "candidate": candidate.candidate_id,
            "result": result,
        }
    finally:
        discard_experiment_checkout(
            experiment,
            workspace=workspace,
            source=source,
        )


def benchmark_verification(source: Path, workspace: Path) -> dict:
    """Real crash control: public runtime failure is evidence, API misuse is not."""

    reproduction = """import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5))
y = X[:, 0] * X[:, 1] + X[:, 2]
data = InputData(idx=np.arange(60), task=Task(TaskTypesEnum.regression),
                 data_type=DataTypesEnum.table, features=X, target=y.reshape(-1, 1))
pipeline = (PipelineBuilder()
            .add_node('poly_features', params={'degree': 3})
            .add_node('linear').build())
pipeline.fit(data)
prediction = pipeline.predict(data)
assert np.isfinite(prediction.predict).all(), 'valid pipeline must produce finite output'
"""
    proposal = VerificationProposal(
        action="verify_bug",
        claim="poly_features must accept its documented degree parameter",
        expected="the public pipeline fits with degree=3",
        file_path=_PCA_FILE,
        reproduction_code=reproduction,
        why="degree is in the registered runtime parameter surface",
    )
    before = source_fingerprint(source)
    stock = run_fedot_snippet(source, reproduction)
    grounded, terminal = _grounded_public_exception(proposal, stock)

    direct = proposal.model_copy(
        update={
            "reproduction_code": reproduction.replace(
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder",
                "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
                "from fedot.core.operations.evaluation.operation_implementations."
                "data_operations.sklearn_transformations import PolyFeaturesImplementation",
            )
        }
    )
    direct_grounded, _ = _grounded_public_exception(direct, stock)

    misuse_code = """import numpy as np
from fedot.core.operations.operation import Operation
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
data = InputData(idx=np.arange(2), features=np.ones((2, 1)), target=np.ones((2, 1)),
                 task=Task(TaskTypesEnum.regression), data_type=DataTypesEnum.table)
operation = Operation(operation_type='poly_features', params={'degree': 3})
operation.fit(data)
assert operation is not None
"""
    misuse = proposal.model_copy(update={"reproduction_code": misuse_code})
    misuse_result = run_fedot_snippet(source, misuse_code)
    misuse_grounded, _ = _grounded_public_exception(misuse, misuse_result)

    checks = {
        "public_probe_crashed": stock.status == "runtime_error",
        "claimed_runtime_reached": grounded,
        "real_failure_preserved": "InvalidParameterError" in terminal,
        "direct_implementation_rejected": not direct_grounded,
        "public_api_misuse_rejected": not misuse_grounded,
        "source_immutable": source_fingerprint(source) == before,
    }
    return {
        "ok": all(checks.values()),
        "component": "verification",
        "checks": checks,
        "terminal_exception": terminal,
    }


def benchmark_contract_support(source: Path, workspace: Path, *, inference) -> dict:
    """Regression for a reproduced assertion outside a documented precondition."""

    reproduction = """from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.preprocessing.structure import PipelineStructureExplorer

primary = PipelineNode('scaling')
pipeline = Pipeline(PipelineNode('linear', nodes_from=[primary]))
observed = PipelineStructureExplorer.check_structure_by_tag(
    pipeline, 'imputation', source_name='source-not-present-in-pipeline')
assert observed is False
"""
    lead = PatchSite(
        "regression",
        "fedot/preprocessing/structure.py",
        57,
        (
            "A missing primary-source label makes all(empty paths) true, but the "
            "source_name docstring requires the label of a primary node."
        ),
        hypothesis_kind="correctness",
    )
    proposal = VerificationProposal(
        action="verify_bug",
        claim="an absent primary-source label must be treated as a deficient branch",
        expected="the absent source is reported as requiring preprocessing",
        file_path=lead.file_path,
        reproduction_code=reproduction,
    )
    audit: ContractSupportAudit = _audit_contract_support(
        source,
        lead,
        proposal,
        inference=inference,
    )
    payload = {
        "ok": audit.verdict == "unsupported",
        "component": "contract-support",
        "execution_ok": True,
        "false_accepts": int(audit.verdict == "supported"),
        "healthy_rejections": int(audit.verdict == "unsupported"),
        "audit": audit.model_dump(),
    }
    (workspace / "contract_support_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _score(task: str, status: str, value: float) -> ScoreResult:
    return ScoreResult(task_id=task, status=status, score=value)


def negative_controls() -> dict:
    lift = ("lift",)
    protect = ("guard",)
    stock = {"lift": _score("lift", "ok", 0.80), "guard": _score("guard", "ok", 0.80)}
    packs = {
        "no_op": {"lift": _score("lift", "ok", 0.80), "guard": _score("guard", "ok", 0.80)},
        "inverse": {"lift": _score("lift", "ok", 0.70), "guard": _score("guard", "ok", 0.80)},
        "neutral": {"lift": _score("lift", "ok", 0.805), "guard": _score("guard", "ok", 0.80)},
        "regression": {"lift": _score("lift", "ok", 0.82), "guard": _score("guard", "ok", 0.70)},
        "crash": {"lift": _score("lift", "crash", 0.5), "guard": _score("guard", "ok", 0.80)},
        "timeout": {"lift": _score("lift", "timeout", float("nan")), "guard": _score("guard", "ok", 0.80)},
        "invalid": {"lift": _score("lift", "invalid", float("nan")), "guard": _score("guard", "ok", 0.80)},
    }
    rows = {}
    for name, patched in packs.items():
        decision = compare_pack(
            stock,
            patched,
            lift_ids=lift,
            protect_ids=protect,
            min_delta=0.01,
            sentinel=0.5,
        )
        rows[name] = {"kept": decision.keep, "reason": decision.reason}
    return rows


def _judge_known_case(
    source: Path,
    workspace: Path,
    *,
    run_id: str,
    candidate: PatchCandidate,
    tasks: tuple[str, ...],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
) -> dict:
    tree = create_experiment_checkout(
        source, workspace, run_id=run_id, candidate_id=candidate.candidate_id
    )
    try:
        applied = apply_patch(tree, candidate)
        if not applied:
            return {
                "ok": False,
                "case_id": candidate.candidate_id,
                "reason": "canonical_patch_apply_failed",
            }
        dev_stock = measure_stock(tasks, checkout=source, split="dev", seed=42)
        dev_patched = measure_patched(tasks, checkout=tree, split="dev", seed=42)
        dev = verdict(
            dev_stock,
            dev_patched,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
        )
        final_stock = measure_stock(tasks, checkout=source, split="final", seed=42)
        final_patched = measure_patched(tasks, checkout=tree, split="final", seed=42)
        final = verdict(
            final_stock,
            final_patched,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
        )
        return {
            "ok": bool(dev.keep and final.keep),
            "case_id": candidate.candidate_id,
            "dev": {"keep": dev.keep, "reason": dev.reason, "delta": dev.target_delta},
            "final": {"keep": final.keep, "reason": final.reason, "delta": final.target_delta},
        }
    finally:
        discard_experiment_checkout(tree, workspace=workspace, source=source)


def benchmark_judge(source: Path, workspace: Path) -> dict:
    run_id = f"benchmark-{uuid.uuid4().hex[:8]}"
    cases = [
        _judge_known_case(
            source,
            workspace,
            run_id=run_id,
            candidate=pca_case(),
            tasks=("pca->catboost", "catboost", "fast_ica->lgbm"),
            lift_ids=("pca->catboost",),
            protect_ids=("catboost", "fast_ica->lgbm"),
        ),
        _judge_known_case(
            source,
            workspace,
            run_id=run_id,
            candidate=rf_leaf2_case(),
            tasks=("rf", "cancer", "kc2"),
            lift_ids=("rf", "cancer", "kc2"),
            protect_ids=("rf", "cancer", "kc2"),
        ),
    ]
    controls = negative_controls()
    negative_kept = sum(int(row["kept"]) for row in controls.values())
    dev_kept = sum(int(bool(row.get("dev", {}).get("keep"))) for row in cases)
    final_kept = sum(int(bool(row.get("final", {}).get("keep"))) for row in cases)
    pca = cases[0]
    return {
        "ok": all(row["ok"] for row in cases) and negative_kept == 0,
        "component": "judge",
        "confirmed_cases": len(cases),
        "known_patches_kept_by_dev": dev_kept,
        "known_patches_kept_by_final": final_kept,
        "negative_controls_kept": negative_kept,
        # Compatibility: keep the original PCA summary at the top level.
        "dev": pca["dev"],
        "final": pca["final"],
        "cases": {row["case_id"]: row for row in cases},
        "controls": controls,
    }


def benchmark_localization(source: Path, workspace: Path) -> dict:
    payload = measure_localization(source, workspace=workspace)
    ranked = payload["by_split"]
    hits = 0
    for split in ranked.values():
        ranks = split["structural_invariant"]["ranks"]
        hits += sum(rank is not None and rank <= 20 for rank in ranks.values())
    return {
        "ok": hits >= 4,
        "component": "localization",
        "localization_top20_hits": hits,
        "details": payload,
    }


def benchmark_configuration(source: Path, workspace: Path) -> dict:
    """Exercise the complete non-LLM path for an executed operation default.

    The harness supplies only the operations executed by the frozen workloads.
    The canonical patch remains benchmark-only and is never exposed to Scout or
    Fixer.  This component isolates infrastructure failures from LLM failures.
    """

    operation_hints = {
        "rf": ("rf",),
        "cancer": ("rf",),
        "kc2": ("rf",),
    }
    runtime_stock = measure_stock(
        tuple(operation_hints),
        checkout=source,
        split="dev",
        seed=42,
        collect_coverage=True,
    )
    leads = default_parameter_leads(
        source,
        operation_hints,
        scores=runtime_stock,
    )
    first = leads[0] if leads else None
    localization_ok = bool(
        first
        and first.channel == "configuration"
        and first.file_path == _RF_DEFAULTS_FILE
        and "operation rf" in first.why
    )
    parameter_surface_ok = bool(
        first
        and any(
            "runtime implementation: RandomForestClassifier" in item
            and "min_samples_leaf" in item
            and "estimator defaults" in item
            for item in first.evidence
        )
    )

    run_id = f"configuration-{uuid.uuid4().hex[:8]}"
    experiment = create_experiment_checkout(
        source,
        workspace,
        run_id=run_id,
        candidate_id="rf-default-chain",
    )
    try:
        stock_hash = source_fingerprint(source)
        applied = apply_patch(experiment, rf_leaf2_case())
        patched_hash = source_fingerprint(experiment) if applied else stock_hash
        if applied:
            from fedotllm.agents.evolve.controller.campaign import compare_behavior_probe

            probe = compare_behavior_probe(
                source,
                experiment,
                "\n".join(
                    (
                        "from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository",
                        'params = DefaultOperationParamsRepository().get_default_params_for_operation("rf")',
                        'print("EVOLVE_OBSERVATION=" + str(params.get("min_samples_leaf", 1)))',
                    )
                ),
            )
        else:
            probe = {"status": "not_run"}

        judged = (
            _judge_known_case(
                source,
                workspace,
                run_id=run_id,
                candidate=rf_leaf2_case(),
                tasks=("rf", "cancer", "kc2"),
                lift_ids=("rf", "cancer", "kc2"),
                protect_ids=("rf", "cancer", "kc2"),
            )
            if applied
            else {"ok": False, "reason": "canonical_patch_apply_failed"}
        )
        checks = {
            "localization_top1": localization_ok,
            "parameter_surface_observed": parameter_surface_ok,
            "patch_applied": applied,
            "behavior_changed": probe.get("status") == "changed",
            "fingerprint_changed": stock_hash != patched_hash,
            "dev_keep": bool(judged.get("dev", {}).get("keep")),
            "final_keep": bool(judged.get("final", {}).get("keep")),
        }
        return {
            "ok": all(checks.values()),
            "component": "configuration",
            "checks": checks,
            "lead": asdict(first) if first else None,
            "behavior_probe": probe,
            "fingerprints": {"stock": stock_hash, "patched": patched_hash},
            "judge": judged,
        }
    finally:
        discard_experiment_checkout(
            experiment,
            workspace=workspace,
            source=source,
        )


def benchmark_configuration_search(source: Path, workspace: Path) -> dict:
    """Prove generic runtime-surface search, without loading a canonical patch."""

    from fedotllm.agents.evolve.commands.configuration_search import (
        search_configuration_variants,
    )

    tasks = ("rf", "cancer", "kc2")
    operation_hints = {task_id: ("rf",) for task_id in tasks}
    stock = measure_stock(
        tasks,
        checkout=source,
        split="dev",
        seed=42,
        collect_coverage=True,
    )
    baseline_tests = normalize_test_result(measure_fedot_tests(source))
    outcome = search_configuration_variants(
        source,
        workspace,
        run_id=f"configuration-search-{uuid.uuid4().hex[:8]}",
        source_hash=source_fingerprint(source),
        stock=stock,
        operation_hints=operation_hints,
        problem_by_task={task_id: "classification" for task_id in tasks},
        lift_ids=tasks,
        protect_ids=tasks,
        baseline_tests=baseline_tests,
        max_trials=1,
        dev_seed=42,
    )
    if outcome.candidate is None or outcome.experiment is None:
        return {
            "ok": False,
            "component": "configuration-search",
            "stage": "dev_search",
            "decision": asdict(outcome.decision),
            "trials": outcome.trials,
        }
    try:
        final_stock = measure_stock(tasks, checkout=source, split="final", seed=42)
        final_patched = measure_patched(
            tasks,
            checkout=outcome.experiment,
            split="final",
            seed=42,
        )
        final = verdict(
            final_stock,
            final_patched,
            lift_ids=tasks,
            protect_ids=tasks,
        )
        selected = outcome.trials[-1].get("variant") or {}
        checks = {
            "generated_without_canonical_patch": outcome.candidate.candidate_id.startswith(
                "config-"
            ),
            "quick_dev_keep": bool(
                (outcome.trials[-1].get("quick_decision") or {}).get("keep")
            ),
            "full_dev_keep": outcome.decision.keep,
            "behavior_changed": outcome.behavior_probe.get("status") == "changed",
            "tests_passed_comparatively": outcome.tests is not None,
            "final_keep": final.keep,
        }
        return {
            "ok": all(checks.values()),
            "component": "configuration-search",
            "checks": checks,
            "selected_variant": selected,
            "candidate": outcome.candidate.candidate_id,
            "dev": asdict(outcome.decision),
            "final": asdict(final),
            "trials": outcome.trials,
        }
    finally:
        discard_experiment_checkout(
            outcome.experiment,
            workspace=workspace,
            source=source,
        )


def _brief_scores(scores: dict[str, ScoreResult]) -> dict[str, dict]:
    return {
        task_id: {
            "status": result.status,
            "score": result.score,
            "detail": result.detail,
        }
        for task_id, result in scores.items()
    }


def _fixer_feedback(decision, patched, candidate: PatchCandidate) -> str:
    outcome = "regressed" if decision.reason.startswith("regression") else "neutral"
    statuses = ", ".join(
        f"{task_id}:{result.status}" for task_id, result in sorted(patched.items())
    )
    edits = "\n".join(
        f"EDIT {index} {edit.file_path}\nSEARCH:\n{edit.old_code}\nREPLACE:\n{edit.new_code}"
        for index, edit in enumerate(candidate.edits, start=1)
    )
    return (
        f"outcome={outcome}; DEV_delta={decision.target_delta}; "
        f"reason={decision.reason}; task_statuses={statuses}\n\n"
        f"Previous evaluated patch:\n{edits[:6_000]}\n"
        "Use this exact measured result. Do not repeat the patch. If a shared "
        "implementation regressed a sibling operation, scope the correction to "
        "the affected concrete operation while preserving its siblings."
    )


def benchmark_fixer(
    source: Path,
    workspace: Path,
    *,
    verifier_inference,
    fixer_inference,
) -> dict:
    """Oracle-location Fixer benchmark on a confirmed metric-changing workload.

    The oracle supplies only the source location.  Verifier and Fixer receive
    stock runtime evidence, never the canonical patch or benchmark case text.
    Provisional historical files are deliberately excluded: a patch in a file
    that the workload never executes cannot measure metric-guided repair.
    """

    component_workspace = workspace / "confirmed_fixer"
    component_workspace.mkdir(parents=True, exist_ok=True)
    run_id = f"fixer-benchmark-{uuid.uuid4().hex[:8]}"
    lift_ids = ("pca->catboost",)
    protect_ids = ("catboost", "fast_ica->lgbm")
    tasks = (*lift_ids, *protect_ids)
    stock = measure_stock(
        tasks,
        checkout=source,
        split="dev",
        seed=42,
        collect_coverage=True,
    )
    pca_stock = stock.get(lift_ids[0])
    if pca_stock is None or pca_stock.status not in {"ok", "crash"}:
        return {
            "ok": False,
            "component": "fixer",
            "stage": "stock_evaluator",
            "reason": "confirmed workload did not produce a valid stock result",
            "stock": _brief_scores(stock),
        }
    leads = leads_from_scores(
        {lift_ids[0]: pca_stock},
        source,
        operation_hints={lift_ids[0]: ("pca", "catboost")},
    )
    lead = next((item for item in leads if item.file_path == _PCA_FILE), None)
    if lead is None:
        return {
            "ok": False,
            "component": "fixer",
            "stage": "oracle_location",
            "reason": "confirmed workload evidence did not resolve to the oracle file",
            "stock": _brief_scores(stock),
            "localized_files": [item.file_path for item in leads],
        }

    verifier_tree = create_experiment_checkout(
        source,
        workspace,
        run_id=run_id,
        candidate_id="verifier",
    )
    try:
        verification = verify_lead(
            verifier_tree,
            lead,
            inference=verifier_inference,
            workspace=component_workspace,
        )
    finally:
        discard_experiment_checkout(verifier_tree, workspace=workspace, source=source)
    if not verification.proceed:
        return {
            "ok": False,
            "component": "fixer",
            "stage": "verifier",
            "reason": verification.detail or verification.status,
            "verification": asdict(verification),
            "stock": _brief_scores(stock),
        }

    feedback = ""
    attempts: list[dict] = []
    for revision in range(1, 4):
        experiment = create_experiment_checkout(
            source,
            workspace,
            run_id=run_id,
            candidate_id=f"candidate-r{revision}",
        )
        try:
            candidate = fix_lead(
                experiment,
                lead,
                inference=fixer_inference,
                workspace=component_workspace,
                verification=verification_context(verification),
                feedback=feedback,
            )
            if candidate is None:
                return {
                    "ok": False,
                    "component": "fixer",
                    "stage": "fixer",
                    "reason": "fixer returned no applicable patch",
                    "revision": revision,
                    "attempts": attempts,
                    "verification": asdict(verification),
                    "stock": _brief_scores(stock),
                }

            reproduction = replay_reproduction(experiment, verification)
            if (
                verification.status == "verified_bug"
                and reproduction.get("patched") != "resolved"
            ):
                attempts.append(
                    {
                        "revision": revision,
                        "stage": "reproduction",
                        "reason": "verified stock failure remains after the patch",
                        "candidate": asdict(candidate),
                    }
                )
                if revision < 3:
                    feedback = (
                        "outcome=neutral; the independent stock failure still fails after "
                        "the exact previous patch. Correct the verified mechanism.\n"
                        + _fixer_feedback(
                            Decision(False, "verified_bug_not_resolved", None),
                            {},
                            candidate,
                        )
                    )
                    continue
                return {
                    "ok": False,
                    "component": "fixer",
                    "stage": "reproduction",
                    "reason": "verified stock failure remains after the patch",
                    "attempts": attempts,
                    "candidate": asdict(candidate),
                    "verification": asdict(verification),
                    "reproduction": reproduction,
                    "stock": _brief_scores(stock),
                }

            broken = {
                rel: error
                for rel in dict.fromkeys(edit.file_path for edit in candidate.edits)
                if (error := import_error(experiment, rel))
            }
            if broken:
                return {
                    "ok": False,
                    "component": "fixer",
                    "stage": "import_gate",
                    "reason": "patched module is not importable",
                    "revision": revision,
                    "attempts": attempts,
                    "import_errors": broken,
                    "candidate": asdict(candidate),
                    "verification": asdict(verification),
                    "reproduction": reproduction,
                    "stock": _brief_scores(stock),
                }

            patched = measure_patched(
                tasks,
                checkout=experiment,
                split="dev",
                seed=42,
            )
            decision = verdict(
                stock,
                patched,
                lift_ids=lift_ids,
                protect_ids=protect_ids,
            )
            attempt = {
                "revision": revision,
                "stage": "dev_judge",
                "reason": decision.reason,
                "delta": decision.target_delta,
                "regressions": decision.regression_deltas,
                "candidate": asdict(candidate),
                "patched": _brief_scores(patched),
            }
            attempts.append(attempt)
            if not decision.keep and revision < 3:
                feedback = _fixer_feedback(decision, patched, candidate)
                continue

            test_payload = None
            if decision.keep:
                baseline_tests = normalize_test_result(measure_fedot_tests(source))
                patched_tests = normalize_test_result(measure_fedot_tests(experiment))
                blocked = tests_regressed(baseline_tests, patched_tests)
                test_payload = {
                    "baseline_status": baseline_tests.status,
                    "patched_status": patched_tests.status,
                    "new_failed_nodes": sorted(
                        patched_tests.failed_nodes - baseline_tests.failed_nodes
                    ),
                    "passed": blocked is None,
                    "reason": None if blocked is None else blocked.reason,
                }
                if blocked is not None:
                    return {
                        "ok": False,
                        "component": "fixer",
                        "stage": "pytest_gate",
                        "reason": blocked.reason,
                        "confirmed_cases": 1,
                        "oracle_fixer_dev_keeps": 0,
                        "revision": revision,
                        "attempts": attempts,
                        "candidate": asdict(candidate),
                        "verification": asdict(verification),
                        "reproduction": reproduction,
                        "stock": _brief_scores(stock),
                        "patched": _brief_scores(patched),
                        "tests": test_payload,
                    }
            return {
                "ok": bool(decision.keep),
                "component": "fixer",
                "stage": "dev_judge",
                "reason": decision.reason,
                "confirmed_cases": 1,
                "oracle_fixer_dev_keeps": int(bool(decision.keep)),
                "revision": revision,
                "attempts": attempts,
                "candidate": asdict(candidate),
                "verification": asdict(verification),
                "reproduction": reproduction,
                "stock": _brief_scores(stock),
                "patched": _brief_scores(patched),
                "tests": test_payload,
                "dev": {
                    "keep": decision.keep,
                    "reason": decision.reason,
                    "delta": decision.target_delta,
                    "regressions": decision.regression_deltas,
                },
            }
        finally:
            discard_experiment_checkout(experiment, workspace=workspace, source=source)

    raise AssertionError("fixer revision loop terminated without a result")


def run_component(
    component: str,
    source: Path,
    workspace: Path,
    *,
    inference=None,
    scout_inference=None,
    verifier_inference=None,
    fixer_inference=None,
    architecture: str = "staged",
    case_ids: tuple[str, ...] | None = None,
    committee_size: int = 3,
) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    if component == "micro-fast":
        from fedotllm.agents.evolve.benchmark.micro import run_stock_microbenchmark

        payload = run_stock_microbenchmark(source).as_dict()
        (workspace / "micro_fast_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    if component == "micro-agent":
        if inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        from fedotllm.agents.evolve.benchmark.micro_agent import (
            run_micro_agent_benchmark,
        )
        from fedotllm.agents.evolve.model_contract import build_model_contract

        contract = build_model_contract(
            {"scout": inference, "verifier": inference, "fixer": inference}
        )
        (workspace / "model_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not contract["ok"]:
            return {
                "ok": False,
                "component": component,
                "status": "model_contract_invalid",
                "diagnostics": contract["diagnostics"],
            }
        return run_micro_agent_benchmark(
            source,
            workspace,
            inference=inference,
            architecture=architecture,
            case_ids=case_ids,
            committee_size=committee_size,
        )
    if component == "micro-discovery":
        if inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        from fedotllm.agents.evolve.benchmark.micro_discovery import (
            run_micro_discovery_benchmark,
        )
        from fedotllm.agents.evolve.model_contract import build_model_contract

        contract = build_model_contract(
            {"scout": inference, "verifier": inference, "fixer": inference}
        )
        (workspace / "model_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not contract["ok"]:
            return {
                "ok": False,
                "component": component,
                "status": "model_contract_invalid",
                "diagnostics": contract["diagnostics"],
            }
        return run_micro_discovery_benchmark(
            source,
            workspace,
            inference=inference,
            architecture=architecture,
            case_ids=case_ids,
            committee_size=committee_size,
        )
    if component == "hidden-controls":
        if inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        from fedotllm.agents.evolve.benchmark.hidden_controls import (
            run_hidden_control_benchmark,
        )

        return run_hidden_control_benchmark(
            source,
            workspace,
            inference=inference,
            architecture=architecture,
            committee_size=committee_size,
        )
    if component == "hidden-controls-fresh":
        if inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        from fedotllm.agents.evolve.benchmark.hidden_controls import (
            run_fresh_hidden_control_benchmark,
        )

        return run_fresh_hidden_control_benchmark(
            source,
            workspace,
            inference=inference,
            architecture=architecture,
            committee_size=committee_size,
        )
    if component == "hidden-controls-fresh2":
        if inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        from fedotllm.agents.evolve.benchmark.hidden_controls import (
            run_fresh_v2_hidden_control_benchmark,
        )

        return run_fresh_v2_hidden_control_benchmark(
            source,
            workspace,
            inference=inference,
            architecture=architecture,
            committee_size=committee_size,
        )
    if component == "judge":
        payload = benchmark_judge(source, workspace)
        artifact = workspace / "judge_result.json"
        previous = None
        try:
            previous = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        comparable = dict(payload)
        comparable.pop("repeated_run_matches_previous", None)
        old_comparable = dict(previous or {})
        old_comparable.pop("repeated_run_matches_previous", None)
        payload["repeated_run_matches_previous"] = bool(
            previous is not None and old_comparable == comparable
        )
        artifact.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
    if component == "localization":
        return benchmark_localization(source, workspace)
    if component == "configuration":
        return benchmark_configuration(source, workspace)
    if component == "configuration-search":
        return benchmark_configuration_search(source, workspace)
    if component == "affected":
        payload = benchmark_affected(source, workspace)
        artifact = workspace / "affected_result.json"
        artifact.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    if component == "verification":
        payload = benchmark_verification(source, workspace)
        (workspace / "verification_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    if component == "contract-support":
        if verifier_inference is None:
            return {
                "ok": False,
                "component": component,
                "status": "requires_llm",
                "reason": "Pass --allow-llm; paid LLM is never invoked implicitly.",
            }
        return benchmark_contract_support(
            source,
            workspace,
            inference=verifier_inference,
        )
    verifier_client = verifier_inference or scout_inference or inference
    fixer_client = fixer_inference or inference
    if component == "fixer" and fixer_client is not None and verifier_client is not None:
        return benchmark_fixer(
            source,
            workspace,
            verifier_inference=verifier_client,
            fixer_inference=fixer_client,
        )
    if component == "e2e" and inference is not None:
        from dataclasses import asdict

        from fedotllm.agents.evolve.controller.campaign import run_once

        decision = run_once(
            checkout=source,
            inference=inference,
            scout_inference=scout_inference,
            verifier_inference=verifier_inference or scout_inference,
            fixer_inference=fixer_inference,
            workspace=workspace,
        )
        keeps = int(bool(decision.dev_keep))
        return {
            "ok": keeps >= 1,
            "component": "e2e",
            "end_to_end_dev_keeps": keeps,
            "decision": asdict(decision),
        }
    if component in {"fixer", "e2e"}:
        return {
            "ok": False,
            "component": component,
            "status": "requires_llm",
            "reason": "Pass --allow-llm with an API key; paid LLM is never invoked implicitly.",
        }
    raise ValueError(component)
