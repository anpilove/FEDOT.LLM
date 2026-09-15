from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from fedotllm.agents.evolve.execution.checkout import (
    MARKER,
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.evaluation.scorer import (
    _separate_test_split,
    _single_csv_split,
)
from fedotllm.agents.evolve.types import Decision, PatchCandidate, PatchEdit, TaskSpec
from fedotllm.agents.evolve.types import (
    EvolveRunPolicy,
    PatchSite,
    ScoreResult,
    SnippetResult,
    TestResult,
    VerificationResult,
)

FAST_RUN_POLICY = EvolveRunPolicy(
    verify_manifest=False,
    confirm_and_ablate=False,
    confirm_small_signals=False,
    evaluate_final=False,
    fedot_quality_jobs=False,
)
CONFIRM_RUN_POLICY = EvolveRunPolicy(
    verify_manifest=False,
    confirm_and_ablate=True,
    confirm_small_signals=False,
    evaluate_final=False,
    fedot_quality_jobs=False,
)
FINAL_RUN_POLICY = EvolveRunPolicy(
    verify_manifest=False,
    confirm_and_ablate=False,
    confirm_small_signals=False,
    evaluate_final=True,
    fedot_quality_jobs=False,
)


def _source(root: Path) -> Path:
    source = root / "source"
    package = source / "fedot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "a.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (package / "b.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    return source


def _causal_fields(line: int = 1) -> dict[str, object]:
    return {
        "change_line": line,
        "mechanism": "the executed value changes model input",
        "proposed_change": "replace the executed expression with a concrete alternative",
        "expected_metric_effect": "preserve more predictive information",
    }


def _accept_behavior_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.compare_behavior_probe",
        lambda *_a, **_k: {"status": "changed"},
    )


def test_applied_patch_hash_deduplicates_equivalent_json_edits(tmp_path: Path):
    from fedotllm.agents.evolve.storage.hypothesis import normalized_patch_hash

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text('{\n  "rf": {\n    "n_jobs": 1\n  }\n}\n', encoding="utf-8")
    source_hash = source_fingerprint(source)
    candidates = (
        PatchCandidate(
            "narrow",
            edits=[
                PatchEdit(
                    defaults.relative_to(source).as_posix(),
                    '  "rf": {\n    "n_jobs": 1\n  }',
                    '  "logit": {"tol": 1e-4},\n  "rf": {\n    "n_jobs": 1\n  }',
                )
            ],
        ),
        PatchCandidate(
            "wide",
            edits=[
                PatchEdit(
                    defaults.relative_to(source).as_posix(),
                    '{\n  "rf": {\n    "n_jobs": 1\n  }\n}',
                    '{\n  "logit": {"tol": 0.0001},\n  "rf": {\n    "n_jobs": 1\n  }\n}',
                )
            ],
        ),
    )
    hashes = []
    for index, candidate in enumerate(candidates):
        checkout = tmp_path / f"checkout-{index}"
        checkout.mkdir()
        checkout_defaults = checkout / defaults.relative_to(source)
        checkout_defaults.parent.mkdir(parents=True)
        checkout_defaults.write_text(
            defaults.read_text(encoding="utf-8"), encoding="utf-8"
        )
        assert apply_patch(checkout, candidate)
        hashes.append(normalized_patch_hash(candidate, source_hash, checkout=checkout))

    assert hashes[0] == hashes[1]


def test_run_task_selection_is_deduplicated_and_validated():
    from fedotllm.agents.evolve.__main__ import _task_ids

    assert _task_ids("") is None
    assert _task_ids("metocean, temperature,metocean") == (
        "metocean",
        "temperature",
    )
    with pytest.raises(ValueError, match="unknown Evolve task ids: invented"):
        _task_ids("catboost,invented")


def test_correctness_keep_is_success_without_metric_final():
    from fedotllm.agents.evolve.__main__ import _decision_exit_code

    decision = Decision(
        keep=True,
        reason="correctness_keep",
        target_delta=0.0,
        dev_keep=False,
        final_keep=None,
        correctness_keep=True,
        stage="correctness",
    )

    assert _decision_exit_code(decision) == 0


def test_maintenance_keep_is_success_without_claiming_metric_improvement():
    from fedotllm.agents.evolve.__main__ import _decision_exit_code
    from fedotllm.agents.evolve.storage.findings import classify_finding

    decision = Decision(
        keep=True,
        reason="maintenance_keep",
        target_delta=0.0,
        dev_keep=False,
        final_keep=None,
        maintenance_keep=True,
        stage="maintenance",
    )
    finding = {
        "candidate": "candidate",
        "fedot_test_status": "passed",
        "fedot_test_gate_passed": True,
        "keep": False,
        "maintenance_keep": True,
        "reason": "maintenance_keep",
        "target_delta": 0.0,
    }

    assert _decision_exit_code(decision) == 0
    assert classify_finding(finding) == (
        "maintenance_keep",
        "review_maintenance_patch",
    )


def test_confirmed_small_metric_signal_is_preserved_for_review():
    from fedotllm.agents.evolve.__main__ import _decision_exit_code
    from fedotllm.agents.evolve.storage.findings import classify_finding

    decision = Decision(
        keep=True,
        reason="confirmed_small_metric_keep",
        target_delta=0.007,
        dev_keep=False,
        metric_signal_keep=True,
        stage="metric_signal",
    )
    finding = {
        "candidate": "candidate",
        "fedot_test_status": "passed",
        "fedot_test_gate_passed": True,
        "keep": False,
        "metric_signal_keep": True,
        "reason": "confirmed_small_metric_keep",
        "target_delta": 0.007,
    }

    assert _decision_exit_code(decision) == 0
    assert classify_finding(finding) == (
        "confirmed_small_metric_keep",
        "review_metric_patch",
    )


def test_configuration_benchmark_exposes_each_pipeline_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.benchmark import runner
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    defaults = source / runner._RF_DEFAULTS_FILE
    defaults.parent.mkdir(parents=True)
    defaults.write_text(
        "{\n" + runner._RF_DEFAULTS_OLD + '  "other": {}\n}\n',
        encoding="utf-8",
    )
    original_defaults = defaults.read_text(encoding="utf-8")

    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda *args, **kwargs: {
            "status": "changed",
            "stock": {"observation": "1"},
            "patched": {"observation": "2"},
        },
    )
    monkeypatch.setattr(
        runner,
        "_judge_known_case",
        lambda *args, **kwargs: {
            "ok": True,
            "dev": {"keep": True, "delta": 0.02},
            "final": {"keep": True, "delta": 0.015},
        },
    )
    monkeypatch.setattr(
        runner,
        "measure_stock",
        lambda *args, **kwargs: {
            task_id: ScoreResult(
                task_id,
                "ok",
                0.8,
                dataflow=(
                    {
                        "operation": "rf",
                        "implementation": "RandomForestClassifier",
                        "params": {"n_jobs": 1},
                        "supported_parameters": ["min_samples_leaf", "n_jobs"],
                        "estimator_defaults": {"min_samples_leaf": 1, "n_jobs": None},
                    },
                ),
            )
            for task_id in ("rf", "cancer", "kc2")
        },
    )

    result = runner.benchmark_configuration(source, tmp_path / "benchmark")

    assert result["ok"] is True
    assert all(result["checks"].values())
    assert result["lead"]["channel"] == "configuration"
    assert result["lead"]["file_path"] == runner._RF_DEFAULTS_FILE
    assert result["fingerprints"]["stock"] != result["fingerprints"]["patched"]
    assert defaults.read_text(encoding="utf-8") == original_defaults


def test_affected_benchmark_component_persists_reproducible_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.benchmark import runner

    payload = {
        "ok": True,
        "component": "affected",
        "checks": {"unsafe_fix_rejected": True, "lead_reached": True},
    }
    monkeypatch.setattr(runner, "benchmark_affected", lambda *_a, **_k: payload)

    workspace = tmp_path / "benchmark"
    result = runner.run_component("affected", tmp_path / "source", workspace)

    assert result == payload
    assert json.loads((workspace / "affected_result.json").read_text()) == payload


def test_findings_separates_metric_neutral_from_confirmed_fix():
    from fedotllm.agents.evolve.storage.findings import classify_finding

    neutral = {
        "candidate": "candidate",
        "fedot_test_status": "passed",
        "keep": False,
        "reason": "target_delta 0.0000 below per-task threshold",
        "target_delta": 0.0,
    }
    assert classify_finding(neutral) == ("metric_neutral_unverified", "do_not_apply")

    verified = {
        **neutral,
        "reproduction": {
            "stock": "failed_as_predicted",
            "patched": "resolved",
        },
    }
    assert classify_finding(verified) == (
        "correctness_keep",
        "review_correctness_patch",
    )

    known_baseline_failure = {
        **neutral,
        "fedot_test_status": "test_failures",
        "fedot_test_gate_passed": True,
    }
    assert classify_finding(known_baseline_failure) == (
        "metric_neutral_unverified",
        "do_not_apply",
    )

    affected_regression = {
        **verified,
        "reason": "affected_metric_regression",
        "affected_metric": {"status": "regressed"},
    }
    assert classify_finding(affected_regression) == (
        "rejected_affected_metric",
        "do_not_apply",
    )

    affected_keep = {
        **verified,
        "reason": "confirmed_fix_affected_metric_keep",
        "affected_metric": {
            "dev_confirmation": {"confirmed": True},
            "final_confirmation": {"confirmed": True},
        },
    }
    assert classify_finding(affected_keep) == (
        "affected_metric_keep",
        "queue_for_reviewed_branch",
    )


def test_cross_run_memory_never_blacklists_an_entire_source_file():
    from fedotllm.agents.evolve.storage.replay import exclude_whole_file_after_attempt

    assert not exclude_whole_file_after_attempt(
        "fedot/core/repository/data/model_repository.json"
    )
    assert not exclude_whole_file_after_attempt(
        "fedot/core/repository/data/default_operation_params.json"
    )
    assert not exclude_whole_file_after_attempt("fedot/core/operations/model.py")


def test_configuration_patch_gets_controller_owned_behavior_probe():
    from fedotllm.agents.evolve.agents.fixer import configuration_behavior_probe

    lead = PatchSite(
        "configuration",
        "fedot/core/repository/data/model_repository.json",
        417,
        "default parameters for executed operation ridge",
        evidence=("executed operation: ridge",),
    )
    candidate = PatchCandidate(
        "ridge-default",
        edits=[
            PatchEdit(
                "fedot/core/repository/data/default_operation_params.json",
                '"rf": {}',
                '"rf": {}, "ridge": {"alpha": 0.1}',
            )
        ],
    )

    probe = configuration_behavior_probe(lead, candidate)

    assert "DefaultOperationParamsRepository" in probe
    assert 'get_default_params_for_operation("ridge")' in probe
    assert "EVOLVE_OBSERVATION=" in probe


def test_generic_configuration_variants_use_runtime_surface_not_case_catalog(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.commands.configuration_search import (
        candidate_for_variant,
        propose_parameter_variants,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    scores = {
        task_id: ScoreResult(
            task_id,
            "ok",
            0.8,
            dataflow=(
                {
                    "operation": "rf",
                    "implementation": "RandomForestClassifier",
                    "params": {"n_jobs": 1},
                    "supported_parameters": [
                        "class_weight",
                        "max_depth",
                        "min_samples_leaf",
                    ],
                    "estimator_defaults": {
                        "class_weight": None,
                        "max_depth": None,
                        "min_samples_leaf": 1,
                    },
                },
            ),
        )
        for task_id in ("a", "b")
    }
    variants = propose_parameter_variants(
        {"a": ("rf",), "b": ("rf",)},
        scores,
        problem_by_task={"a": "classification", "b": "classification"},
    )

    assert [(row.parameter, row.value) for row in variants] == [
        ("min_samples_leaf", 2),
        ("min_samples_leaf", 4),
    ]
    candidate = candidate_for_variant(source, variants[0])
    assert candidate is not None
    assert apply_patch(source, candidate)
    payload = json.loads(defaults.read_text(encoding="utf-8"))
    assert payload["rf"] == {"min_samples_leaf": 2, "n_jobs": 1}


def test_configuration_surface_falls_back_to_traced_wrapper_params():
    from fedotllm.agents.evolve.commands.configuration_search import (
        propose_parameter_variants,
    )

    scores = {
        "catboost": ScoreResult(
            "catboost",
            "ok",
            0.8,
            dataflow=(
                {
                    "operation": "catboost",
                    "implementation": "FedotCatBoostClassificationImplementation",
                    "params": {"l2_leaf_reg": 0.01, "max_depth": 5},
                    "supported_parameters": [],
                    "estimator_defaults": {},
                },
            ),
        )
    }

    variants = propose_parameter_variants(
        {"catboost": ("catboost",)},
        scores,
        problem_by_task={"catboost": "classification"},
    )

    assert [(row.parameter, row.value) for row in variants] == [
        ("l2_leaf_reg", 0.001),
        ("l2_leaf_reg", 0.1),
    ]


def test_runtime_parameter_surface_reads_nested_fitted_estimator():
    from fedotllm.agents.evolve.evaluation._worker import _parameter_surface

    class Estimator:
        def get_params(self, deep=False):
            assert deep is False
            return {"reg_alpha": 0.0, "reg_lambda": 1.0, "objective": object()}

    class Wrapper:
        def __init__(self):
            self.model = Estimator()

    supported, defaults = _parameter_surface(Wrapper())

    assert supported == ["objective", "reg_alpha", "reg_lambda"]
    assert defaults == {"reg_alpha": 0.0, "reg_lambda": 1.0}


def test_generic_lgbm_surface_reaches_regularization_and_capacity_variants():
    from fedotllm.agents.evolve.commands.configuration_search import (
        propose_parameter_variants,
    )

    flow = (
        {
            "operation": "lgbm",
            "implementation": "FedotLightGBMClassificationImplementation",
            "params": {"max_depth": -1},
            "supported_parameters": [
                "min_child_samples",
                "reg_alpha",
                "reg_lambda",
                "max_depth",
                "learning_rate",
                "num_leaves",
            ],
            "estimator_defaults": {
                "min_child_samples": 20,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "max_depth": -1,
                "learning_rate": 0.1,
                "num_leaves": 31,
            },
        },
    )
    variants = propose_parameter_variants(
        {"lgbm": ("lgbm",)},
        {"lgbm": ScoreResult("lgbm", "ok", 0.8, dataflow=flow)},
        problem_by_task={"lgbm": "classification"},
        max_variants_per_operation=8,
        max_total=8,
    )

    pairs = [(row.parameter, row.value) for row in variants]
    assert pairs[:2] == [("min_child_samples", 10), ("min_child_samples", 40)]
    assert ("reg_lambda", 0.01) in pairs
    assert ("reg_alpha", 0.01) in pairs

    deep_variants = propose_parameter_variants(
        {"lgbm": ("lgbm",)},
        {"lgbm": ScoreResult("lgbm", "ok", 0.8, dataflow=flow)},
        problem_by_task={"lgbm": "classification"},
        max_variants_per_operation=20,
        max_total=20,
    )
    deep_pairs = [(row.parameter, row.value) for row in deep_variants]
    assert ("learning_rate", 0.05) in deep_pairs
    assert ("num_leaves", 15) in deep_pairs


def test_generic_decomposition_surface_proposes_existing_solver_alternatives():
    from fedotllm.agents.evolve.commands.configuration_search import (
        propose_parameter_variants,
    )

    flow = (
        {
            "operation": "fast_ica",
            "implementation": "FastICAImplementation",
            "params": {"whiten": "unit-variance"},
            "supported_parameters": ["whiten", "algorithm", "max_iter", "tol"],
            "estimator_defaults": {
                "whiten": "unit-variance",
                "algorithm": "parallel",
                "max_iter": 200,
                "tol": 0.0001,
            },
        },
    )
    variants = propose_parameter_variants(
        {"ica": ("fast_ica",)},
        {"ica": ScoreResult("ica", "ok", 0.8, dataflow=flow)},
        max_variants_per_operation=4,
        max_total=4,
    )

    assert [(row.parameter, row.value) for row in variants] == [
        ("whiten", "arbitrary-variance"),
        ("algorithm", "deflation"),
        ("max_iter", 400),
        ("tol", 1e-05),
    ]


def test_configuration_bundle_applies_coupled_boosting_defaults_atomically(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.commands.configuration_search import (
        ParameterVariant,
        candidate_for_variant,
        propose_parameter_bundles,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(
        json.dumps({"lgbm": {"learning_rate": 0.1, "num_leaves": 31}}, indent=2),
        encoding="utf-8",
    )
    singles = [
        ParameterVariant("lgbm", "learning_rate", 0.05, 0.1, "slower updates"),
        ParameterVariant("lgbm", "learning_rate", 0.2, 0.1, "faster updates"),
        ParameterVariant("lgbm", "num_leaves", 15, 31, "lower capacity"),
        ParameterVariant("lgbm", "num_leaves", 63, 31, "higher capacity"),
    ]

    bundles = propose_parameter_bundles(singles)
    candidate = candidate_for_variant(source, bundles[0])

    assert candidate is not None
    assert apply_patch(source, candidate)
    assert json.loads(defaults.read_text(encoding="utf-8"))["lgbm"] == {
        "learning_rate": 0.05,
        "num_leaves": 15,
    }


def test_configuration_bundle_is_scheduled_before_unrelated_later_singles():
    from fedotllm.agents.evolve.commands.configuration_search import (
        ParameterBundle,
        ParameterVariant,
        schedule_parameter_variants,
    )

    learning = ParameterVariant("lgbm", "learning_rate", 0.05, 0.1, "slower")
    leaves = ParameterVariant("lgbm", "num_leaves", 15, 31, "smaller")
    unrelated = ParameterVariant("lgbm", "subsample", 0.8, 1.0, "sampling")
    bundle = ParameterBundle("lgbm", (learning, leaves), "coupled capacity")

    scheduled = schedule_parameter_variants(
        [learning, leaves, unrelated],
        [bundle],
    )

    assert scheduled == [learning, leaves, bundle, unrelated]


def test_positive_bundle_near_miss_produces_bounded_numeric_refinements():
    from fedotllm.agents.evolve.commands.configuration_search import (
        ParameterBundle,
        refinements_from_trials,
    )

    rows = [
        {
            "stage": "quick_dev",
            "reason": "target_delta 0.0092 below per-task threshold",
            "variant": {
                "operation": "lgbm",
                "variants": [
                    {
                        "operation": "lgbm",
                        "parameter": "learning_rate",
                        "value": 0.05,
                        "current_value": 0.1,
                        "rationale": "slower updates",
                    },
                    {
                        "operation": "lgbm",
                        "parameter": "num_leaves",
                        "value": 15,
                        "current_value": 31,
                        "rationale": "smaller trees",
                    },
                ],
                "rationale": "coupled update",
            },
        }
    ]

    refined = refinements_from_trials(rows)

    assert len(refined) == 4
    assert all(isinstance(item, ParameterBundle) for item in refined)
    values = [
        tuple((part.parameter, part.value) for part in item.variants)
        for item in refined
    ]
    assert values == [
        (("learning_rate", 0.075), ("num_leaves", 15)),
        (("learning_rate", 0.025), ("num_leaves", 15)),
        (("learning_rate", 0.05), ("num_leaves", 23)),
        (("learning_rate", 0.05), ("num_leaves", 7)),
    ]


def test_configuration_candidate_adds_missing_operation_default(tmp_path: Path):
    from fedotllm.agents.evolve.commands.configuration_search import (
        ParameterVariant,
        candidate_for_variant,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    variant = ParameterVariant(
        "ridge",
        "alpha",
        0.1,
        1.0,
        "nearby regularization scale",
    )

    candidate = candidate_for_variant(source, variant)

    assert candidate is not None
    assert apply_patch(source, candidate)
    payload = json.loads(defaults.read_text(encoding="utf-8"))
    assert payload["ridge"] == {"alpha": 0.1}
    assert payload["rf"] == {"n_jobs": 1}


def test_configuration_search_requires_quick_full_probe_and_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import judge
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.execution.checkout import discard_experiment_checkout
    from fedotllm.agents.evolve.commands.configuration_search import (
        search_configuration_variants,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    rf_flow = (
        {
            "operation": "rf",
            "implementation": "RandomForestClassifier",
            "params": {"n_jobs": 1},
            "supported_parameters": ["min_samples_leaf"],
            "estimator_defaults": {"min_samples_leaf": 1},
        },
    )
    stock = {
        task_id: ScoreResult(task_id, "ok", 0.80, dataflow=rf_flow)
        for task_id in ("rf", "cancer", "kc2")
    }
    stock["catboost"] = ScoreResult("catboost", "ok", 0.80)
    measured: list[tuple[str, tuple[str, ...]]] = []

    def fake_patched(task_ids, **kwargs):
        ids = tuple(task_ids)
        measured.append((kwargs.get("split", "dev"), ids))
        return {
            task_id: ScoreResult(
                task_id,
                "ok",
                0.82 if task_id in {"rf", "cancer", "kc2"} else 0.80,
            )
            for task_id in ids
        }

    monkeypatch.setattr(judge, "measure_patched", fake_patched)
    monkeypatch.setattr(
        judge,
        "measure_stock",
        lambda task_ids, **_kwargs: {
            task_id: ScoreResult(task_id, "ok", 0.80) for task_id in task_ids
        },
    )
    monkeypatch.setattr(
        judge,
        "measure_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda *args, **kwargs: {"status": "changed"},
    )
    workspace = tmp_path / "workspace"
    outcome = search_configuration_variants(
        source,
        workspace,
        run_id="configuration-search",
        source_hash=source_fingerprint(source),
        stock=stock,
        operation_hints={
            "rf": ("rf",),
            "cancer": ("rf",),
            "kc2": ("rf",),
            "catboost": ("catboost",),
        },
        problem_by_task={
            "rf": "classification",
            "cancer": "classification",
            "kc2": "classification",
            "catboost": "classification",
        },
        lift_ids=("rf", "cancer", "kc2", "catboost"),
        protect_ids=("rf", "cancer", "kc2", "catboost"),
        baseline_tests=TestResult("passed", 0),
        max_trials=2,
    )

    try:
        assert outcome.candidate is not None
        assert outcome.experiment is not None
        assert outcome.decision.keep is True
        assert measured[0] == ("dev", ("rf", "cancer", "kc2"))
        assert measured[1] == ("dev", ("rf", "cancer", "kc2", "catboost"))
        assert measured[2] == ("shadow", ("rf", "cancer", "kc2"))
        assert outcome.trials[-1]["stage"] == "dev_keep"
    finally:
        if outcome.experiment is not None:
            discard_experiment_checkout(
                outcome.experiment,
                workspace=workspace,
                source=source,
            )


def test_configuration_search_durable_dedup_does_not_consume_new_trial_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import judge
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.execution.checkout import discard_experiment_checkout
    from fedotllm.agents.evolve.commands.configuration_search import (
        ParameterVariant,
        candidate_for_variant,
        search_configuration_variants,
    )
    from fedotllm.agents.evolve.storage.hypothesis import normalized_patch_hash

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    flow = (
        {
            "operation": "rf",
            "implementation": "RandomForestClassifier",
            "params": {"n_jobs": 1},
            "supported_parameters": ["min_samples_leaf", "min_samples_split"],
            "estimator_defaults": {
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            },
        },
    )
    stock = {"rf": ScoreResult("rf", "ok", 0.80, dataflow=flow)}
    source_hash = source_fingerprint(source)
    skipped: set[str] = set()
    for value in (2, 4):
        candidate = candidate_for_variant(
            source,
            ParameterVariant("rf", "min_samples_leaf", value, 1, "already tried"),
        )
        assert candidate is not None
        skipped.add(normalized_patch_hash(candidate, source_hash))

    monkeypatch.setattr(
        judge,
        "measure_patched",
        lambda task_ids, **kwargs: {
            task_id: ScoreResult(task_id, "ok", 0.82) for task_id in task_ids
        },
    )
    monkeypatch.setattr(
        judge,
        "measure_stock",
        lambda task_ids, **_kwargs: {
            task_id: ScoreResult(task_id, "ok", 0.80) for task_id in task_ids
        },
    )
    monkeypatch.setattr(
        judge,
        "measure_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda *args, **kwargs: {"status": "changed"},
    )
    workspace = tmp_path / "workspace"
    outcome = search_configuration_variants(
        source,
        workspace,
        run_id="configuration-dedup",
        source_hash=source_hash,
        stock=stock,
        operation_hints={"rf": ("rf",)},
        problem_by_task={"rf": "classification"},
        lift_ids=("rf",),
        protect_ids=("rf",),
        baseline_tests=TestResult("passed", 0),
        skip_patch_hashes=skipped,
        max_trials=1,
    )

    try:
        assert outcome.candidate is not None
        assert "min_samples_split" in outcome.candidate.rationale
        assert [row["stage"] for row in outcome.trials[:2]] == ["dedup", "dedup"]
        assert outcome.trials[-1]["stage"] == "dev_keep"
    finally:
        if outcome.experiment is not None:
            discard_experiment_checkout(
                outcome.experiment,
                workspace=workspace,
                source=source,
            )


def test_configuration_search_continues_after_shadow_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import judge
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.execution.checkout import discard_experiment_checkout
    from fedotllm.agents.evolve.commands.configuration_search import (
        search_configuration_variants,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    flow = (
        {
            "operation": "rf",
            "implementation": "RandomForestClassifier",
            "params": {"n_jobs": 1},
            "supported_parameters": ["min_samples_leaf"],
            "estimator_defaults": {"min_samples_leaf": 1},
        },
    )
    stock = {"rf": ScoreResult("rf", "ok", 0.80, dataflow=flow)}

    def fake_stock(task_ids, **_kwargs):
        return {task_id: ScoreResult(task_id, "ok", 0.80) for task_id in task_ids}

    def fake_patched(task_ids, *, checkout, split="dev", **_kwargs):
        params = json.loads(
            (
                checkout / "fedot/core/repository/data/default_operation_params.json"
            ).read_text()
        )["rf"]
        leaf = params.get("min_samples_leaf")
        # Both variants fit the original DEV.  Only the second transfers to the
        # training-only SHADOW sample.
        value = 0.82 if split == "dev" or leaf == 4 else 0.805
        return {task_id: ScoreResult(task_id, "ok", value) for task_id in task_ids}

    monkeypatch.setattr(judge, "measure_stock", fake_stock)
    monkeypatch.setattr(judge, "measure_patched", fake_patched)
    monkeypatch.setattr(
        judge,
        "measure_fedot_tests",
        lambda *_args, **_kwargs: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda *_args, **_kwargs: {"status": "changed"},
    )

    workspace = tmp_path / "workspace"
    outcome = search_configuration_variants(
        source,
        workspace,
        run_id="configuration-shadow",
        source_hash=source_fingerprint(source),
        stock=stock,
        operation_hints={"rf": ("rf",)},
        problem_by_task={"rf": "classification"},
        lift_ids=("rf",),
        protect_ids=("rf",),
        baseline_tests=TestResult("passed", 0),
        max_trials=2,
    )

    try:
        assert outcome.candidate is not None
        assert "min_samples_leaf 1 -> 4" in outcome.candidate.rationale
        assert outcome.trials[0]["stage"] == "shadow_dev"
        assert outcome.trials[0]["shadow_decision"]["keep"] is False
        assert outcome.trials[-1]["stage"] == "dev_keep"
    finally:
        if outcome.experiment is not None:
            discard_experiment_checkout(
                outcome.experiment,
                workspace=workspace,
                source=source,
            )


def test_stock_pytest_cache_is_invalidated_by_test_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import judge

    source = _source(tmp_path)
    test_file = source / "test" / "unit" / "test_runtime.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_runtime():\n    assert True\n", encoding="utf-8")
    monkeypatch.setenv(
        "EVOLVE_AGENT_TEST_BASELINE_CACHE",
        str(tmp_path / "test-cache"),
    )
    calls = {"count": 0}

    def fake_snapshot(_checkout, **kwargs):
        assert kwargs["maxfail"] == 0
        calls["count"] += 1
        return TestResult(
            "test_failures",
            1,
            failed_nodes={"test/unit/test_known.py::test_known"},
            output="known stock failure",
        )

    monkeypatch.setattr(judge, "pytest_snapshot", fake_snapshot)

    first = judge.measure_baseline_fedot_tests(source)
    second = judge.measure_baseline_fedot_tests(source)
    test_file.write_text("def test_runtime():\n    assert 1 == 1\n", encoding="utf-8")
    third = judge.measure_baseline_fedot_tests(source)

    assert first.failed_nodes == {"test/unit/test_known.py::test_known"}
    assert second.failed_nodes == first.failed_nodes
    assert third.failed_nodes == first.failed_nodes
    assert calls["count"] == 2


def test_candidate_pytest_gate_retries_only_a_nonreproducible_test_failure(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.evaluation.judge import confirm_candidate_tests

    baseline = TestResult(
        "test_failures",
        1,
        failed_nodes={"test/unit/test_known.py::test_known"},
    )
    first = TestResult(
        "test_failures",
        1,
        failed_nodes={
            "test/unit/test_known.py::test_known",
            "test/unit/test_flaky.py::test_order_dependent",
        },
    )
    retry = TestResult(
        "test_failures",
        1,
        failed_nodes={"test/unit/test_known.py::test_known"},
    )
    calls = []

    result, blocked, attempts = confirm_candidate_tests(
        baseline,
        tmp_path,
        first=first,
        runner=lambda _checkout: calls.append("retry") or retry,
    )

    assert calls == ["retry"]
    assert result is retry
    assert blocked is None
    assert len(attempts) == 2

    infrastructure = TestResult("collection_error", 2)
    calls.clear()
    result, blocked, attempts = confirm_candidate_tests(
        baseline,
        tmp_path,
        first=infrastructure,
        runner=lambda _checkout: calls.append("must not run") or retry,
    )
    assert calls == []
    assert result is infrastructure
    assert blocked is not None and blocked.infrastructure_error is True
    assert len(attempts) == 1


def test_configuration_search_cannot_keep_quick_lift_with_full_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import judge
    from fedotllm.agents.evolve.commands.configuration_search import (
        search_configuration_variants,
    )

    source = _source(tmp_path)
    defaults = source / "fedot/core/repository/data/default_operation_params.json"
    defaults.parent.mkdir(parents=True)
    defaults.write_text(json.dumps({"rf": {"n_jobs": 1}}, indent=2), encoding="utf-8")
    flow = (
        {
            "operation": "rf",
            "implementation": "RandomForestClassifier",
            "params": {"n_jobs": 1},
            "supported_parameters": ["min_samples_leaf"],
            "estimator_defaults": {"min_samples_leaf": 1},
        },
    )
    stock = {
        "rf": ScoreResult("rf", "ok", 0.80, dataflow=flow),
        "catboost": ScoreResult("catboost", "ok", 0.80),
    }
    calls = 0

    def fake_patched(task_ids, **kwargs):
        nonlocal calls
        calls += 1
        ids = tuple(task_ids)
        return {
            task_id: ScoreResult(
                task_id,
                "ok",
                0.82 if task_id == "rf" else 0.60,
            )
            for task_id in ids
        }

    monkeypatch.setattr(judge, "measure_patched", fake_patched)
    monkeypatch.setattr(
        judge,
        "measure_fedot_tests",
        lambda *args, **kwargs: pytest.fail(
            "tests must not run after full DEV regression"
        ),
    )
    outcome = search_configuration_variants(
        source,
        tmp_path / "workspace",
        run_id="configuration-regression",
        source_hash=source_fingerprint(source),
        stock=stock,
        operation_hints={"rf": ("rf",), "catboost": ("catboost",)},
        problem_by_task={"rf": "classification", "catboost": "classification"},
        lift_ids=("rf", "catboost"),
        protect_ids=("rf", "catboost"),
        baseline_tests=TestResult("passed", 0),
        max_trials=1,
    )

    assert outcome.candidate is None
    assert outcome.decision.keep is False
    assert outcome.decision.reason.startswith("regression catboost")
    assert calls == 2


def test_configuration_search_component_is_offline_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.benchmark import runner

    source = _source(tmp_path)
    monkeypatch.setattr(
        runner,
        "benchmark_configuration_search",
        lambda received_source, received_workspace: {
            "ok": True,
            "component": "configuration-search",
            "source": str(received_source),
            "workspace": str(received_workspace),
        },
    )

    result = runner.run_component(
        "configuration-search",
        source,
        tmp_path / "benchmark",
    )

    assert result["ok"] is True
    assert result["component"] == "configuration-search"


def test_configuration_trial_hashes_are_durable_across_workspaces(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import append_configuration_trials
    from fedotllm.agents.evolve.storage.replay import (
        configuration_trials_from_findings,
        tried_patch_hashes_from_findings,
    )

    dataset = tmp_path / "findings.jsonl"
    append_configuration_trials(
        dataset,
        run_number=1,
        run_id="run-one",
        source_hash="source-a",
        score_protocol_hash="score-a",
        evaluation_protocol_hash="protocol-a",
        workspace=tmp_path / "first-workspace",
        trials=[
            {
                "patch_hash": "hash-rejected",
                "candidate": "candidate-a",
                "variant": {"operation": "ridge", "parameter": "alpha", "value": 0.1},
                "stage": "quick_dev",
                "reason": "target_delta 0.009 below per-task threshold",
                "quick_decision": {
                    "keep": False,
                    "target_delta": 0.009,
                    "reason": "target_delta 0.009 below per-task threshold",
                },
            }
        ],
    )

    assert tried_patch_hashes_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="protocol-a",
    ) == {"hash-rejected"}
    assert not tried_patch_hashes_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="protocol-b",
    )
    assert not tried_patch_hashes_from_findings(dataset, source_hash="source-b")
    history = configuration_trials_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="protocol-a",
    )
    assert len(history) == 1
    assert history[0]["score_protocol_hash"] == "score-a"
    assert history[0]["quick_decision"]["target_delta"] == 0.009
    assert not configuration_trials_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="protocol-b",
    )


def test_llm_finding_dedup_requires_matching_acceptance_protocol(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import append_finding
    from fedotllm.agents.evolve.storage.replay import tried_patch_hashes_from_findings

    dataset = tmp_path / "findings.jsonl"
    append_finding(
        dataset,
        run_number=1,
        run_id="run-one",
        source_commit="commit-a",
        source_hash="source-a",
        workspace=tmp_path,
        row={
            "candidate": "candidate-a",
            "patch_hash": "patch-a",
            "score_protocol_hash": "score-a",
            "evaluation_protocol_hash": "accept-a",
            "keep": False,
            "reason": "target_delta 0.0 below per-task threshold",
            "target_delta": 0.0,
        },
    )

    payload = json.loads(dataset.read_text(encoding="utf-8"))
    assert payload["score_protocol_hash"] == "score-a"
    assert payload["evaluation_protocol_hash"] == "accept-a"
    assert tried_patch_hashes_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="accept-a",
    ) == {"patch-a"}
    assert not tried_patch_hashes_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="accept-b",
    )


def test_shared_defaults_memory_uses_operation_identity_not_selected_line(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.storage.replay import (
        semantic_site_id,
        tried_semantic_sites_from_findings,
    )

    path = "fedot/core/repository/data/default_operation_params.json"
    catalog_site = PatchSite(
        "configuration",
        path,
        26,
        "default parameters for executed operation catboost",
        evidence=("executed operation: catboost",),
    )
    picked_neighbor = PatchSite(
        "llm",
        path,
        30,
        "catboost l2_leaf_reg may overfit",
        evidence=("executed operation: catboost",),
    )
    sibling = PatchSite(
        "configuration",
        path,
        72,
        "default parameters for executed operation lgbm",
        evidence=("executed operation: lgbm",),
    )
    assert semantic_site_id(catalog_site) == semantic_site_id(picked_neighbor)
    assert semantic_site_id(sibling) != semantic_site_id(catalog_site)

    runtime_catalog = PatchSite(
        "execution",
        "fedot/runtime.py",
        159,
        "executed symbol LaggedImplementation._apply_transformation_for_fit",
        evidence=("executed lines in this symbol: 159-160,165,173-174",),
    )
    runtime_pick = PatchSite(
        "llm",
        "fedot/runtime.py",
        173,
        "flatten a time-series slice",
        evidence=(f"catalog semantic site: {semantic_site_id(runtime_catalog)}",),
    )
    assert semantic_site_id(runtime_catalog) == semantic_site_id(runtime_pick)
    legacy_runtime_pick = PatchSite(
        "llm",
        "fedot/runtime.py",
        173,
        "flatten a time-series slice",
        evidence=("executed lines in this symbol: 159-160,165,173-174",),
    )
    assert semantic_site_id(runtime_catalog) == semantic_site_id(legacy_runtime_pick)

    findings = tmp_path / "findings.jsonl"
    findings.write_text(
        json.dumps(
            {
                "record_type": "finding",
                "source_hash": "source-a",
                "lead": {
                    "file_path": path,
                    "line": 30,
                    "why": picked_neighbor.why,
                    "evidence": list(picked_neighbor.evidence),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    excluded = tried_semantic_sites_from_findings(findings, source_hash="source-a")
    assert excluded == {semantic_site_id(catalog_site)}

    source = _source(tmp_path / "repo")
    defaults = source / path
    defaults.parent.mkdir(parents=True, exist_ok=True)
    defaults.write_text("{}\n", encoding="utf-8")
    leads = discover_leads(
        source,
        limit=20,
        trace_leads=[catalog_site, sibling],
        excluded_semantic_sites=excluded,
    )
    identities = {semantic_site_id(lead) for lead in leads}
    assert semantic_site_id(catalog_site) not in identities
    assert semantic_site_id(sibling) in identities


def test_recent_site_cooldown_uses_only_completed_matching_campaigns(tmp_path: Path):
    from fedotllm.agents.evolve.storage.replay import (
        recent_completed_sites_from_findings,
    )

    dataset = tmp_path / "findings.jsonl"

    def start(run_number: int, run_id: str, protocol: str = "accept-a") -> dict:
        return {
            "record_type": "run",
            "event": "run_start",
            "run_number": run_number,
            "run_id": run_id,
            "source_hash": "source-a",
            "campaign_config": {"evaluation_protocol_hash": protocol},
        }

    def finding(run_number: int, run_id: str, file_path: str, line: int) -> dict:
        return {
            "record_type": "finding",
            "event": "finding",
            "run_number": run_number,
            "run_id": run_id,
            "source_hash": "source-a",
            "evaluation_protocol_hash": "accept-a",
            "lead": {"file_path": file_path, "line": line},
        }

    def end(run_number: int, run_id: str) -> dict:
        return {
            "record_type": "run",
            "event": "run_end",
            "run_number": run_number,
            "run_id": run_id,
            "immutable_source": True,
        }

    rows = [
        start(1, "run-one"),
        finding(1, "run-one", "fedot/a.py", 10),
        end(1, "run-one"),
        start(2, "run-two"),
        finding(2, "run-two", "fedot/b.py", 20),
        end(2, "run-two"),
        # Interrupted campaign: its finding must not suppress the next run.
        start(3, "run-interrupted"),
        finding(3, "run-interrupted", "fedot/c.py", 30),
        # A newer completed campaign under another acceptance protocol is not
        # part of this experiment lineage.
        start(4, "run-other-protocol", "accept-b"),
        end(4, "run-other-protocol"),
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert recent_completed_sites_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="accept-a",
        campaigns=1,
    ) == {("fedot/b.py", 20)}
    assert recent_completed_sites_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="accept-a",
        campaigns=2,
    ) == {("fedot/a.py", 10), ("fedot/b.py", 20)}
    assert not recent_completed_sites_from_findings(
        dataset,
        source_hash="source-b",
        evaluation_protocol_hash="accept-a",
    )

    # A completed campaign with no findings consumes the one-campaign window,
    # so older sites become eligible again instead of forming a blacklist.
    rows.extend([start(5, "run-empty"), end(5, "run-empty")])
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert not recent_completed_sites_from_findings(
        dataset,
        source_hash="source-a",
        evaluation_protocol_hash="accept-a",
        campaigns=1,
    )


def test_findings_dataset_numbers_runs_and_imports_once(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import import_workspace, summarize

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    (workspace / "trace.jsonl").write_text(
        json.dumps(
            {"event": "campaign_start", "run_id": "run-a", "source_hash": "hash"}
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "campaign_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "source": str(_source(tmp_path)),
                "source_commit": "commit",
                "source_hash_before": "hash",
                "immutable_source": True,
                "decision": {"keep": False, "reason": "no_patch"},
            }
        ),
        encoding="utf-8",
    )
    (workspace / "journal.jsonl").write_text(
        json.dumps(
            {
                "event": "decision",
                "revision": 1,
                "candidate": "c1",
                "lead": {"file_path": "fedot/a.py", "line": 1},
                "edits": [],
                "fedot_test_status": "passed",
                "keep": False,
                "reason": "target_delta 0.0000 below per-task threshold",
                "target_delta": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "findings.jsonl"
    imported = import_workspace(workspace, dataset)
    assert imported["run_number"] == 1
    assert imported["findings"] == 1
    assert import_workspace(workspace, dataset)["reason"] == "duplicate"
    assert summarize(dataset) == {
        "path": str(dataset.resolve()),
        "runs": 1,
        "findings": 1,
        "rejudges": 0,
        "outcomes": {"metric_neutral_unverified": 1},
        "effective_outcomes": {"metric_neutral_unverified": 1},
    }


def test_findings_summary_uses_rejudge_as_effective_outcome(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import summarize

    dataset = tmp_path / "findings.jsonl"
    rows = [
        {
            "record_type": "finding",
            "candidate_id": "candidate-a",
            "outcome": "confirmed_fix_metric_neutral",
            "run_number": 1,
        },
        {
            "record_type": "rejudge",
            "candidate_id": "candidate-a",
            "outcome": "rejected_affected_metric",
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    summary = summarize(dataset)
    assert summary["outcomes"] == {
        "confirmed_fix_metric_neutral": 1,
        "rejected_affected_metric": 1,
    }
    assert summary["effective_outcomes"] == {"rejected_affected_metric": 1}
    assert summary["rejudges"] == 1


def test_final_outcome_supersedes_dev_keep_in_findings_summary(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import append_final_outcome, summarize

    dataset = tmp_path / "findings.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "record_type": "finding",
                "run_number": 7,
                "run_id": "run-final",
                "candidate_id": "candidate-final",
                "outcome": "quality_keep_dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    append_final_outcome(
        dataset,
        run_number=7,
        run_id="run-final",
        candidate_id="candidate-final",
        workspace=tmp_path,
        decision={
            "keep": False,
            "final_keep": False,
            "stage": "final",
            "reason": "final_confirmation_failed",
            "target_delta": -0.03,
        },
    )
    # The operation is idempotent when campaign finalization is retried.
    append_final_outcome(
        dataset,
        run_number=7,
        run_id="run-final",
        candidate_id="candidate-final",
        workspace=tmp_path,
        decision={"keep": False, "reason": "final_confirmation_failed"},
    )

    summary = summarize(dataset)
    assert summary["rejudges"] == 1
    assert summary["effective_outcomes"] == {"final_drop": 1}


def test_experiments_are_isolated_and_source_is_immutable(tmp_path: Path):
    source = _source(tmp_path)
    workspace = tmp_path / "work"
    before = source_fingerprint(source)
    first = create_experiment_checkout(
        source, workspace, run_id="run", candidate_id="one"
    )
    (first / "fedot" / "a.py").write_text(
        "def value():\n    return 99\n", encoding="utf-8"
    )
    second = create_experiment_checkout(
        source, workspace, run_id="run", candidate_id="two"
    )
    assert "return 1" in (second / "fedot" / "a.py").read_text(encoding="utf-8")
    assert source_fingerprint(source) == before
    discard_experiment_checkout(first, workspace=workspace, source=source)
    discard_experiment_checkout(second, workspace=workspace, source=source)
    assert source_fingerprint(source) == before


def test_source_fingerprint_changes_when_runtime_json_changes(tmp_path: Path):
    source = _source(tmp_path)
    config = source / "fedot" / "core" / "repository" / "data" / "defaults.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"rf": {"n_jobs": 1}}\n', encoding="utf-8")
    before = source_fingerprint(source)

    config.write_text(
        '{"rf": {"n_jobs": 1, "min_samples_leaf": 2}}\n',
        encoding="utf-8",
    )

    assert source_fingerprint(source) != before


def test_discard_requires_matching_owned_marker(tmp_path: Path):
    source = _source(tmp_path)
    workspace = tmp_path / "work"
    tree = create_experiment_checkout(
        source, workspace, run_id="run", candidate_id="one"
    )
    marker = json.loads((tree / MARKER).read_text(encoding="utf-8"))
    marker["candidate_id"] = "somewhere-else"
    (tree / MARKER).write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not match"):
        discard_experiment_checkout(tree, workspace=workspace, source=source)
    assert tree.exists()
    with pytest.raises(PermissionError):
        discard_experiment_checkout(source, workspace=workspace, source=source)


def test_multifile_patch_is_transactional(tmp_path: Path):
    source = _source(tmp_path)
    before_a = (source / "fedot" / "a.py").read_text(encoding="utf-8")
    candidate = PatchCandidate(
        "multi",
        edits=[
            PatchEdit("fedot/a.py", "return 1", "return 3"),
            PatchEdit("fedot/b.py", "missing unique text", "return 4"),
        ],
    )
    assert apply_patch(source, candidate) is False
    assert (source / "fedot" / "a.py").read_text(encoding="utf-8") == before_a


def test_multifile_patch_applies_all_files(tmp_path: Path):
    source = _source(tmp_path)
    candidate = PatchCandidate(
        "multi",
        edits=[
            PatchEdit("fedot/a.py", "return 1", "return 3"),
            PatchEdit("fedot/b.py", "return 2", "return 4"),
        ],
    )
    assert apply_patch(source, candidate)
    assert "return 3" in (source / "fedot" / "a.py").read_text(encoding="utf-8")
    assert "return 4" in (source / "fedot" / "b.py").read_text(encoding="utf-8")


def test_candidate_cannot_make_its_own_test_gate_pass(tmp_path: Path):
    source = _source(tmp_path)
    test_file = source / "test" / "unit" / "test_contract.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_contract():\n    assert False\n", encoding="utf-8")
    candidate = PatchCandidate(
        "reward-hack",
        edits=[
            PatchEdit(
                "test/unit/test_contract.py",
                "assert False",
                "assert True",
            )
        ],
    )

    with pytest.raises(PermissionError, match=r"fedot/\*\*"):
        apply_patch(source, candidate)
    assert "assert False" in test_file.read_text(encoding="utf-8")


def test_proposed_test_edits_are_review_only(tmp_path: Path):
    source = _source(tmp_path)
    test_file = source / "test" / "unit" / "test_contract.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_contract():\n    assert False\n", encoding="utf-8")
    candidate = PatchCandidate(
        "contract-change",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
        proposed_test_edits=[
            PatchEdit("test/unit/test_contract.py", "assert False", "assert True")
        ],
    )

    assert apply_patch(source, candidate) is True
    assert "return 2" in (source / "fedot" / "a.py").read_text(encoding="utf-8")
    assert "assert False" in test_file.read_text(encoding="utf-8")


def test_transactional_patch_supports_fedot_repository_json(tmp_path: Path):
    source = _source(tmp_path)
    config = source / "fedot" / "core" / "repository" / "data" / "defaults.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"rf": {"n_jobs": 1}}\n', encoding="utf-8")
    candidate = PatchCandidate(
        "json-default",
        edits=[
            PatchEdit(
                "fedot/core/repository/data/defaults.json",
                '"n_jobs": 1',
                '"n_jobs": 1, "min_samples_leaf": 2',
            )
        ],
    )

    assert apply_patch(source, candidate) is True
    assert json.loads(config.read_text(encoding="utf-8"))["rf"]["min_samples_leaf"] == 2


def test_invalid_json_rolls_back_coordinated_python_edit(tmp_path: Path):
    source = _source(tmp_path)
    python_file = source / "fedot" / "a.py"
    config = source / "fedot" / "defaults.json"
    config.write_text('{"rf": {"n_jobs": 1}}\n', encoding="utf-8")
    before_python = python_file.read_text(encoding="utf-8")
    before_json = config.read_text(encoding="utf-8")
    diagnostics: list[str] = []
    candidate = PatchCandidate(
        "invalid-json",
        edits=[
            PatchEdit("fedot/a.py", "return 1", "return 2"),
            PatchEdit("fedot/defaults.json", '"n_jobs": 1', '"n_jobs":'),
        ],
    )

    assert apply_patch(source, candidate, diagnostics=diagnostics) is False
    assert "invalid JSON" in diagnostics[0]
    assert python_file.read_text(encoding="utf-8") == before_python
    assert config.read_text(encoding="utf-8") == before_json


def test_default_parameter_leads_prioritize_operation_used_by_more_workloads(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import default_parameter_leads

    source = _source(tmp_path)
    config = (
        source
        / "fedot"
        / "core"
        / "repository"
        / "data"
        / "default_operation_params.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps({"ridge": {"alpha": 1.0}, "rf": {"n_jobs": 1}}, indent=2),
        encoding="utf-8",
    )

    leads = default_parameter_leads(
        source,
        {
            "classification-a": ("rf",),
            "classification-b": ("rf",),
            "regression": ("ridge",),
        },
    )

    assert [lead.why for lead in leads] == [
        "default parameters for executed operation rf",
        "default parameters for executed operation ridge",
    ]
    assert leads[0].file_path.endswith("default_operation_params.json")


def test_default_parameter_leads_include_executed_operation_without_defaults(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import default_parameter_leads

    source = _source(tmp_path)
    repository = source / "fedot" / "core" / "repository" / "data"
    repository.mkdir(parents=True)
    (repository / "default_operation_params.json").write_text(
        json.dumps({"rf": {"n_jobs": 1}}, indent=2),
        encoding="utf-8",
    )
    (repository / "model_repository.json").write_text(
        json.dumps(
            {
                "operations": {
                    "ridge": {
                        "meta": "sklearn_regr",
                        "tags": ["linear", "interpretable"],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scores = {
        "regression": ScoreResult(
            "regression",
            "ok",
            1.0,
            dataflow=(
                {
                    "operation": "ridge",
                    "implementation": "Ridge",
                    "params": {},
                    "supported_parameters": ["alpha", "fit_intercept", "solver"],
                    "estimator_defaults": {
                        "alpha": 1.0,
                        "fit_intercept": True,
                        "solver": "auto",
                    },
                },
            ),
        )
    }
    leads = default_parameter_leads(
        source,
        {"regression": ("ridge",)},
        scores=scores,
    )

    assert len(leads) == 1
    assert leads[0].file_path.endswith("default_operation_params.json")
    assert any(
        item.startswith("operation registry metadata:") for item in leads[0].evidence
    )
    assert "current FEDOT defaults: {}" in leads[0].evidence
    assert any("may be added" in item for item in leads[0].evidence)
    assert any(
        "runtime implementation: Ridge" in item
        and "alpha" in item
        and "estimator defaults" in item
        for item in leads[0].evidence
    )


def test_fixed_pipeline_hunt_uses_registry_as_context_not_patch_target():
    from fedotllm.agents.evolve.discovery.selection import _metric_path

    assert not _metric_path("fedot/core/repository/data/model_repository.json")
    assert not _metric_path("fedot/core/repository/data/data_operation_repository.json")
    assert _metric_path("fedot/core/repository/data/default_operation_params.json")


def test_patch_reports_duplicate_search_before_writing(tmp_path: Path):
    source = _source(tmp_path)
    before = (source / "fedot" / "a.py").read_text(encoding="utf-8")
    diagnostics: list[str] = []
    candidate = PatchCandidate(
        "duplicates",
        edits=[
            PatchEdit("fedot/a.py", "return 1", "return 3"),
            PatchEdit("fedot/a.py", "return 1", "return 4"),
        ],
    )

    assert apply_patch(source, candidate, diagnostics=diagnostics) is False
    assert diagnostics == [
        "edits 1 and 2: duplicate SEARCH for fedot/a.py; remove or combine it"
    ]
    assert (source / "fedot" / "a.py").read_text(encoding="utf-8") == before


def test_fixer_corrects_ambiguous_search_after_apply_feedback(tmp_path: Path):
    from fedotllm.agents.evolve.agents.fixer import fix_lead
    from fedotllm.agents.evolve.agents.propose import PatchHunk, PatchProposal

    source = _source(tmp_path)
    target = source / "fedot" / "a.py"
    target.write_text(
        "def first():\n    return 1\n\ndef second():\n    return 1\n",
        encoding="utf-8",
    )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return PatchProposal(
                    status="patch",
                    file_path="fedot/a.py",
                    edits=[
                        PatchHunk(
                            file_path="fedot/a.py",
                            old_code="    return 1",
                            new_code="    return 3",
                        )
                    ],
                )
            assert "SEARCH occurs 2 times" in prompt
            return PatchProposal(
                status="patch",
                file_path="fedot/a.py",
                edits=[
                    PatchHunk(
                        file_path="fedot/a.py",
                        old_code="def first():\n    return 1",
                        new_code="def first():\n    return 3",
                    )
                ],
            )

    inference = Inference()
    workspace = tmp_path / "work"
    candidate = fix_lead(
        source,
        PatchSite("oracle", "fedot/a.py", 1, "quality-changing behavior"),
        inference=inference,
        workspace=workspace,
        max_edits=1,
    )

    assert candidate is not None
    assert inference.calls == 2
    assert "return 3" in target.read_text(encoding="utf-8")
    failures = list((workspace / "candidates").glob("*/apply_failed.txt"))
    assert len(failures) == 1
    assert "SEARCH occurs 2 times" in failures[0].read_text(encoding="utf-8")


def test_fixer_repairs_invalid_behavior_probe_before_applying_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import fixer
    from fedotllm.agents.evolve.types import SnippetResult

    source = _source(tmp_path)
    proposed_contexts: list[str] = []
    proposed = PatchCandidate(
        "bad-probe",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
        behavior_probe="raise TypeError('wrong API')",
    )

    def propose(**kwargs):
        proposed_contexts.append(kwargs["context"])
        return proposed

    results = iter(
        [
            SnippetResult(
                "runtime_error",
                "bad",
                exit_code=1,
                stderr="TypeError: wrong API",
            ),
        ]
    )
    monkeypatch.setattr(fixer, "propose_patch", propose)
    monkeypatch.setattr(fixer, "run_fedot_snippet", lambda *_a, **_k: next(results))
    repair_calls: list[dict] = []

    def repair(_checkout, **kwargs):
        repair_calls.append(kwargs)
        return "print('EVOLVE_OBSERVATION=1')"

    monkeypatch.setattr(fixer, "repair_behavior_probe", repair)

    workspace = tmp_path / "work"
    candidate = fixer.fix_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=object(),
        workspace=workspace,
        max_edits=1,
        validate_behavior_probe=True,
    )

    assert candidate is not None
    assert candidate.candidate_id == "bad-probe"
    assert candidate.behavior_probe == "print('EVOLVE_OBSERVATION=1')"
    assert "return 2" in (source / "fedot/a.py").read_text(encoding="utf-8")
    assert len(proposed_contexts) == 1
    assert len(repair_calls) == 1
    assert "return 1" in repair_calls[0]["frozen_patch"]
    assert "return 2" in repair_calls[0]["frozen_patch"]
    assert repair_calls[0]["failed_result"].stderr == "TypeError: wrong API"
    assert (workspace / "candidates/bad-probe/probe_preflight_failed.txt").is_file()
    assert (workspace / "candidates/bad-probe/behavior_probe_repaired.py").is_file()


def test_probe_builder_timeout_preserves_candidate_and_surfaces_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import fixer
    from fedotllm.agents.evolve.agents.failures import AgentModelFailure
    from fedotllm.llm import LLMRequestTimeout

    source = _source(tmp_path)
    proposed = PatchCandidate(
        "slow-probe",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
        behavior_probe="raise TypeError('wrong API')",
    )
    monkeypatch.setattr(fixer, "propose_patch", lambda **_kwargs: proposed)
    monkeypatch.setattr(
        fixer,
        "run_fedot_snippet",
        lambda *_a, **_k: SnippetResult("runtime_error", "bad", exit_code=1),
    )
    monkeypatch.setattr(
        fixer,
        "repair_behavior_probe",
        lambda *_a, **_k: (_ for _ in ()).throw(LLMRequestTimeout("slow")),
    )

    workspace = tmp_path / "work"
    with pytest.raises(AgentModelFailure) as caught:
        fixer.fix_lead(
            source,
            PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
            inference=object(),
            workspace=workspace,
            max_edits=1,
            validate_behavior_probe=True,
        )

    assert caught.value.category == "timeout"
    assert (workspace / "candidates/slow-probe/candidate.json").is_file()
    assert (workspace / "candidates/slow-probe/probe_repair_error.txt").is_file()


def test_probe_builder_uses_fresh_role_and_accepts_only_executed_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import probe_builder
    from fedotllm.agents.evolve.agents.probe_builder import (
        ProbeProposal,
        repair_behavior_probe,
    )

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return ProbeProposal(action="symbol", query="value")
            return ProbeProposal(
                action="run",
                run_code="print('EVOLVE_OBSERVATION=working')",
            )

    inference = Inference()
    monkeypatch.setattr(
        probe_builder,
        "symbol_runtime",
        lambda *_a, **_k: "def value(): return 1",
    )
    monkeypatch.setattr(
        probe_builder,
        "run_fedot_snippet",
        lambda *_a, **_k: SnippetResult(
            "ok",
            "probe",
            stdout="EVOLVE_OBSERVATION=working\n",
        ),
    )
    failed = SnippetResult("runtime_error", "bad", stderr="TypeError: wrong FEDOT call")

    repaired = repair_behavior_probe(
        source,
        inference=inference,
        frozen_patch="EDIT 1 fedot/a.py\nSEARCH: return 1\nREPLACE: return 2",
        verification="fit/predict invariant",
        failed_probe="bad()",
        failed_result=failed,
    )

    assert repaired == "print('EVOLVE_OBSERVATION=working')"
    assert len(inference.prompts) == 2
    assert "source patch is frozen" in inference.prompts[0]
    assert "def value(): return 1" in inference.prompts[1]


def test_probe_builder_reserves_final_synthesis_after_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import probe_builder
    from fedotllm.agents.evolve.agents.probe_builder import (
        ProbeProposal,
        repair_behavior_probe,
    )

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            if len(self.prompts) <= 2:
                return ProbeProposal(action="read", file_path="fedot/a.py")
            return ProbeProposal(
                action="run",
                run_code="print('EVOLVE_OBSERVATION=working')",
            )

    inference = Inference()
    monkeypatch.setattr(probe_builder, "open_runtime", lambda *_a, **_k: "source")
    monkeypatch.setattr(
        probe_builder,
        "run_fedot_snippet",
        lambda *_a, **_k: SnippetResult(
            "ok", "probe", stdout="EVOLVE_OBSERVATION=working\n"
        ),
    )

    repaired = repair_behavior_probe(
        source,
        inference=inference,
        frozen_patch="EDIT 1 fedot/a.py",
        verification="shape invariant",
        failed_probe="bad()",
        failed_result=SnippetResult("runtime_error", "bad"),
    )

    assert repaired == "print('EVOLVE_OBSERVATION=working')"
    assert len(inference.prompts) == 3
    assert "Final synthesis after navigation" in inference.prompts[-1]


def test_confirmed_fixer_benchmark_uses_workload_evidence_not_provisional_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.benchmark import runner

    source = _source(tmp_path)
    workspace = tmp_path / "benchmark"
    lead = PatchSite(
        "operation",
        runner._PCA_FILE,
        130,
        "executed workload operation before downstream crash",
        evidence=("runtime data flow proves stale metadata",),
    )
    stock = {
        "pca->catboost": ScoreResult("pca->catboost", "crash", 0.5),
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "fast_ica->lgbm": ScoreResult("fast_ica->lgbm", "ok", 0.80),
    }
    patched = {
        "pca->catboost": ScoreResult("pca->catboost", "ok", 0.85),
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "fast_ica->lgbm": ScoreResult("fast_ica->lgbm", "ok", 0.80),
    }
    calls: list[str] = []

    monkeypatch.setattr(runner, "measure_stock", lambda *args, **kwargs: stock)
    monkeypatch.setattr(runner, "measure_patched", lambda *args, **kwargs: patched)
    monkeypatch.setattr(runner, "leads_from_scores", lambda *args, **kwargs: [lead])
    monkeypatch.setattr(
        runner,
        "verify_lead",
        lambda *args, **kwargs: VerificationResult(
            "quality_hypothesis",
            claim="the executed transformation can preserve coherent metadata",
            current_approach="metadata is copied unchanged",
            proposed_approach="derive metadata from transformed width",
            alternatives_considered=("copy", "derive"),
            generality="dimensionality-reduction operations",
        ),
    )

    def fake_fix(checkout, received_lead, **kwargs):
        calls.append(received_lead.file_path)
        assert "Verifier status: quality_hypothesis" in kwargs["verification"]
        candidate = PatchCandidate(
            "from-workload-evidence",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 3")],
        )
        assert apply_patch(checkout, candidate)
        return candidate

    monkeypatch.setattr(runner, "fix_lead", fake_fix)
    monkeypatch.setattr(
        runner,
        "measure_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )

    result = runner.benchmark_fixer(
        source,
        workspace,
        verifier_inference=object(),
        fixer_inference=object(),
    )

    assert result["ok"] is True
    assert result["confirmed_cases"] == 1
    assert result["oracle_fixer_dev_keeps"] == 1
    assert calls == [runner._PCA_FILE]
    assert "repair" not in result
    assert "holdout" not in result


def test_confirmed_fixer_benchmark_refines_regression_in_clean_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.benchmark import runner

    source = _source(tmp_path)
    stock = {
        "pca->catboost": ScoreResult("pca->catboost", "crash", 0.5),
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "fast_ica->lgbm": ScoreResult("fast_ica->lgbm", "ok", 0.80),
    }
    first = {
        "pca->catboost": ScoreResult("pca->catboost", "ok", 0.85),
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "fast_ica->lgbm": ScoreResult("fast_ica->lgbm", "ok", 0.70),
    }
    second = {
        "pca->catboost": ScoreResult("pca->catboost", "ok", 0.85),
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "fast_ica->lgbm": ScoreResult("fast_ica->lgbm", "ok", 0.80),
    }
    lead = PatchSite("operation", runner._PCA_FILE, 130, "executed workload operation")
    monkeypatch.setattr(runner, "measure_stock", lambda *args, **kwargs: stock)
    patched_runs = iter((first, second))
    monkeypatch.setattr(
        runner,
        "measure_patched",
        lambda *args, **kwargs: next(patched_runs),
    )
    monkeypatch.setattr(runner, "leads_from_scores", lambda *args, **kwargs: [lead])
    monkeypatch.setattr(
        runner,
        "verify_lead",
        lambda *args, **kwargs: VerificationResult(
            "quality_hypothesis",
            current_approach="shared behavior",
            proposed_approach="operation-owned behavior",
            alternatives_considered=("shared", "owned"),
            generality="executed operation",
        ),
    )
    calls: list[str] = []

    def fake_fix(checkout, received_lead, **kwargs):
        # Revision two must start from immutable stock, not revision one's tree.
        assert "return 1" in (checkout / "fedot" / "a.py").read_text(encoding="utf-8")
        calls.append(kwargs["feedback"])
        replacement = "return 3" if len(calls) == 1 else "return 4"
        candidate = PatchCandidate(
            f"revision-{len(calls)}",
            edits=[PatchEdit("fedot/a.py", "return 1", replacement)],
        )
        assert apply_patch(checkout, candidate)
        return candidate

    monkeypatch.setattr(runner, "fix_lead", fake_fix)
    monkeypatch.setattr(
        runner,
        "measure_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )

    result = runner.benchmark_fixer(
        source,
        tmp_path / "benchmark",
        verifier_inference=object(),
        fixer_inference=object(),
    )

    assert result["ok"] is True
    assert result["revision"] == 2
    assert len(result["attempts"]) == 2
    assert calls[0] == ""
    assert "outcome=regressed" in calls[1]
    assert "fast_ica->lgbm" in calls[1]
    assert "return 1" in (source / "fedot" / "a.py").read_text(encoding="utf-8")


def test_patch_rejects_symlink_escape(tmp_path: Path):
    source = _source(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("def value():\n    return 7\n", encoding="utf-8")
    link = source / "fedot" / "escape.py"
    link.symlink_to(outside)
    candidate = PatchCandidate(
        "escape",
        edits=[PatchEdit("fedot/escape.py", "return 7", "return 8")],
    )
    with pytest.raises(PermissionError):
        apply_patch(source, candidate)
    assert "return 7" in outside.read_text(encoding="utf-8")


def test_runner_cleans_api_keys_and_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _source(tmp_path)
    monkeypatch.setenv("EXAMPLE_API_KEY", "must-not-leak")
    trace = tmp_path / "trace.jsonl"
    result = run_fedot_snippet(
        source,
        "import os, fedot\nprint(fedot.VALUE, os.getenv('EXAMPLE_API_KEY'))",
        trace_path=trace,
        timeout_s=10,
    )
    assert result.status == "ok"
    assert "1 None" in result.stdout
    row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert row["event"] == "snippet"
    assert row["code"].startswith("import os")


def test_run_once_fails_before_llm_or_scores_when_manifest_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verify_manifest",
        lambda _source: ["dataset hash mismatch"],
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: pytest.fail("scores must not run after manifest failure"),
    )
    workspace = tmp_path / "manifest-work"

    decision = run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
    )

    assert decision.infrastructure_error is True
    assert decision.stage == "infrastructure"
    assert "dataset hash mismatch" in decision.reason
    assert (workspace / "manifest_error.json").is_file()


def test_fedot_pytest_default_timeout_fits_reference_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.discovery.discover import (
        DEFAULT_PYTEST_TIMEOUT_S,
        pytest_result,
    )

    source = _source(tmp_path)
    (source / "test" / "unit").mkdir(parents=True)
    observed: dict[str, float] = {}

    def fake_run(*_args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            args=kwargs.get("args", []), returncode=0, stdout="", stderr=""
        )

    monkeypatch.delenv("EVOLVE_AGENT_PYTEST_TIMEOUT", raising=False)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.discovery.signals.subprocess.run", fake_run
    )
    result = pytest_result(source)

    assert result.status == "passed"
    assert observed["timeout"] == DEFAULT_PYTEST_TIMEOUT_S == 300.0


def test_cli_loads_dotenv_before_declaring_llm_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.__main__ import _inference

    monkeypatch.delenv("FEDOTLLM_LLM_API_KEY", raising=False)
    calls: list[str] = []

    def fake_dotenv():
        calls.append("dotenv")
        monkeypatch.setenv("FEDOTLLM_LLM_API_KEY", "available-after-dotenv")

    fake_config = SimpleNamespace(llm=object())
    monkeypatch.setattr("dotenv.load_dotenv", fake_dotenv)
    monkeypatch.setattr(
        "fedotllm.configs.loader.load_config",
        lambda **_kwargs: calls.append("config") or fake_config,
    )
    monkeypatch.setattr(
        "fedotllm.llm.AIInference",
        lambda config: calls.append("inference") or ("inference", config),
    )

    result = _inference("fedotllm:openrouter")

    assert result == ("inference", fake_config.llm)
    assert calls[0] == "dotenv"
    assert calls.index("dotenv") < calls.index("config") < calls.index("inference")


def test_dev_final_indices_are_disjoint_and_repeatable():
    x = np.arange(200).reshape(100, 2)
    y = np.array([0, 1] * 50)
    dev = _single_csv_split(x, y, seed=42, problem="classification", split_name="dev")
    final = _single_csv_split(
        x, y, seed=42, problem="classification", split_name="final"
    )
    shadow = _single_csv_split(
        x, y, seed=42, problem="classification", split_name="shadow"
    )
    assert len(dev[0]) == len(final[0]) == 60
    assert len(dev[2]) == len(final[2]) == 20
    assert set(map(tuple, dev[2])).isdisjoint(set(map(tuple, final[2])))
    assert set(map(tuple, shadow[2])).isdisjoint(set(map(tuple, dev[2])))
    assert set(map(tuple, shadow[2])).isdisjoint(set(map(tuple, final[2])))
    assert set(map(tuple, shadow[0])).isdisjoint(set(map(tuple, shadow[2])))
    dev_again = _single_csv_split(
        x, y, seed=42, problem="classification", split_name="dev"
    )
    assert np.array_equal(dev[2], dev_again[2])

    explicit_dev, _ = _separate_test_split(
        x, y, seed=42, problem="classification", split_name="dev"
    )
    explicit_final, _ = _separate_test_split(
        x, y, seed=42, problem="classification", split_name="final"
    )
    assert set(map(tuple, explicit_dev)).isdisjoint(set(map(tuple, explicit_final)))


def test_confirmation_model_seed_does_not_change_frozen_partition(
    monkeypatch: pytest.MonkeyPatch,
):
    """Model seeds 42/43/44 must never reshuffle DEV and FINAL rows."""

    import fedot.core.pipelines.pipeline_builder as pipeline_builder
    import fedotllm.agents.evolve.evaluation.scorer as scorer

    split_calls: list[tuple[str, dict]] = []
    train = SimpleNamespace(target=np.array([0, 1]))
    test = SimpleNamespace(target=np.array([0, 1]))

    def fake_split(_spec, split_name="dev", **kwargs):
        split_calls.append((split_name, kwargs))
        return train, test, 2

    class FakePipeline:
        def fit(self, _train):
            return self

        def predict(self, _test, output_mode="default"):
            assert output_mode == "probs"
            return SimpleNamespace(predict=np.array([0.1, 0.9]))

    class FakeBuilder:
        def add_node(self, _node):
            return self

        def build(self):
            return FakePipeline()

    monkeypatch.setattr(
        scorer,
        "load_task",
        lambda _task_id: TaskSpec(task_id="fixed", kind="seq", nodes=("model",)),
    )
    monkeypatch.setattr(scorer, "_make_split", fake_split)
    monkeypatch.setattr(pipeline_builder, "PipelineBuilder", FakeBuilder)

    result_43 = scorer.score_task("fixed", seed=43, split_name="dev")
    result_44 = scorer.score_task("fixed", seed=44, split_name="final")

    # score_task deliberately does not pass the model seed into _make_split;
    # its default FROZEN_SPLIT_SEED=42 owns the immutable partition.
    assert split_calls == [("dev", {}), ("final", {})]
    assert result_43["model_seed"] == 43
    assert result_44["model_seed"] == 44
    assert result_43["split_seed"] == result_44["split_seed"] == 42


def test_dev_confirmation_requires_independent_shadow_keep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import _confirm_dev

    def scores(split: str, *, patched: bool) -> dict[str, ScoreResult]:
        value = 0.70
        if patched:
            value = 0.72 if split == "dev" else 0.705
        return {"task": ScoreResult("task", "ok", value)}

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda _ids, *, split="dev", **_kwargs: scores(split, patched=False),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda _ids, *, split="dev", **_kwargs: scores(split, patched=True),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verdict",
        lambda stock, patched, **_kwargs: Decision(
            keep=(patched["task"].score - stock["task"].score) >= 0.01,
            reason=(
                "keep"
                if (patched["task"].score - stock["task"].score) >= 0.01
                else "target_delta below threshold"
            ),
            target_delta=patched["task"].score - stock["task"].score,
        ),
    )

    confirmed, detail = _confirm_dev(
        tmp_path / "source",
        tmp_path / "experiment",
        ("task",),
        ("task",),
        ("task",),
    )

    assert detail["seed_confirmed"] is True
    assert detail["improved_seeds"] == 3
    assert detail["shadow"]["evaluated"] is True
    assert detail["shadow"]["keep"] is False
    assert confirmed is False


def test_shadow_confirmation_failure_drives_same_hypothesis_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidates = [
        PatchCandidate(
            "near-miss-a",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
            behavior_probe="print('EVOLVE_OBSERVATION=value')",
        ),
        PatchCandidate(
            "near-miss-b",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 3")],
            behavior_probe="print('EVOLVE_OBSERVATION=value')",
        ),
    ]
    feedback_seen: list[str] = []

    def fixer(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    stock = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    improved = {"catboost": ScoreResult("catboost", "ok", 0.82)}
    confirmation = {
        "confirmed": False,
        "improved_seeds": 3,
        "regressed_task_seed_pairs": 0,
        "infrastructure_failures": 0,
        "shadow": {
            "evaluated": True,
            "keep": False,
            "reason": "regression rf delta -0.0101",
            "target_delta": 0.015,
        },
    }

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost",), ("catboost",)),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fixer)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.compare_behavior_probe",
        lambda *_a, **_k: {
            "status": "changed",
            "stock": {"observation": "1"},
            "patched": {"observation": "2"},
        },
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: improved,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._confirm_dev",
        lambda *_a, **_k: (False, dict(confirmation)),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )

    workspace = tmp_path / "work-shadow-revision"
    decision = run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=2,
        policy=CONFIRM_RUN_POLICY,
    )

    assert len(feedback_seen) == 2
    assert "outcome=dev_confirmation_failed" in feedback_seen[1]
    assert "regression rf delta -0.0101" in feedback_seen[1]
    assert decision.reason == "dev_confirmation_failed"
    assert sorted(path.suffix for path in (workspace / "promising").iterdir()) == [
        ".json",
        ".json",
        ".patch",
        ".patch",
    ]


def test_quick_quality_screen_checks_only_affected_workloads_and_cannot_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import _quick_quality_screen

    calls: list[tuple[str, tuple[str, ...]]] = []

    def stock(task_ids, *, split, **_kwargs):
        calls.append((f"stock-{split}", tuple(task_ids)))
        return {task_id: ScoreResult(task_id, "ok", 0.80) for task_id in task_ids}

    def patched(task_ids, *, split, **_kwargs):
        calls.append((f"patched-{split}", tuple(task_ids)))
        score = 0.80 if split == "dev" else 0.82
        return {task_id: ScoreResult(task_id, "ok", score) for task_id in task_ids}

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock", stock
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched", patched
    )
    stock_dev = {
        "catboost": ScoreResult("catboost", "ok", 0.80),
        "rf": ScoreResult("rf", "ok", 0.75),
    }

    passed, detail = _quick_quality_screen(
        tmp_path / "source",
        tmp_path / "experiment",
        ("catboost",),
        ("catboost", "rf"),
        ("catboost", "rf"),
        stock_dev,
    )

    assert passed is True
    assert detail["reason"] == "technically_valid"
    assert detail["queue_priority"] == "normal"
    assert calls == [("patched-dev", ("catboost",))]
    assert "rf" not in json.dumps(detail)


def test_quick_quality_screen_does_not_drop_numeric_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import _quick_quality_screen

    def scores(task_ids, *, patched: bool):
        value = 0.70 if patched else 0.80
        return {task_id: ScoreResult(task_id, "ok", value) for task_id in task_ids}

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda task_ids, **_kwargs: scores(task_ids, patched=False),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda task_ids, **_kwargs: scores(task_ids, patched=True),
    )
    stock_dev = {"catboost": ScoreResult("catboost", "ok", 0.80)}

    passed, detail = _quick_quality_screen(
        tmp_path / "source",
        tmp_path / "experiment",
        ("catboost",),
        ("catboost",),
        ("catboost",),
        stock_dev,
    )

    assert passed is True
    assert detail["reason"] == "technically_valid"
    assert detail["shadow"] == {"evaluated": False}


def test_quick_quality_screen_drops_only_deterministic_patch_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import _quick_quality_screen

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda task_ids, **_kwargs: {
            task_id: ScoreResult(task_id, "crash", 0.0) for task_id in task_ids
        },
    )
    stock_dev = {"catboost": ScoreResult("catboost", "ok", 0.80)}

    passed, detail = _quick_quality_screen(
        tmp_path / "source",
        tmp_path / "experiment",
        ("catboost",),
        ("catboost",),
        ("catboost",),
        stock_dev,
    )

    assert passed is False
    assert "crashed after patch" in detail["reason"]


def test_affected_metric_moved_ignores_unrelated_and_bit_identical_scores():
    from fedotllm.agents.evolve.controller.confirmation import affected_metric_moved

    stock = {
        "lgbm": ScoreResult("lgbm", "ok", 0.86),
        "rf": ScoreResult("rf", "ok", 0.80),
    }
    identical = {
        "lgbm": ScoreResult("lgbm", "ok", 0.86),
        "rf": ScoreResult("rf", "ok", 0.80),
    }
    moved = {
        "lgbm": ScoreResult("lgbm", "ok", 0.8600000001),
        "rf": ScoreResult("rf", "ok", 0.80),
    }

    assert affected_metric_moved(stock, identical, ("lgbm",)) is False
    assert affected_metric_moved(stock, moved, ("lgbm",)) is True
    assert affected_metric_moved(stock, moved, ("rf",)) is False


def test_metric_signal_confirmation_uses_evidence_gate_but_keeps_protect_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import _confirm_dev

    calls: list[bool] = []

    def scores(split: str, *, patched: bool) -> dict[str, ScoreResult]:
        ridge = 50.0
        logit = 0.80
        if patched:
            ridge -= 0.2 if split == "dev" else 0.4
            logit -= 0.003 if split == "dev" else 0.02
        return {
            "ridge": ScoreResult("ridge", "ok", ridge),
            "logit": ScoreResult("logit", "ok", logit),
        }

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda _ids, *, split="dev", **_kwargs: scores(split, patched=False),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda _ids, *, split="dev", **_kwargs: scores(split, patched=True),
    )

    def fake_verdict(stock, patched, *, evidence_only=False, **_kwargs):
        calls.append(evidence_only)
        ridge_delta = stock["ridge"].score - patched["ridge"].score
        logit_delta = patched["logit"].score - stock["logit"].score
        regressed = logit_delta < -0.01
        return Decision(
            keep=evidence_only and ridge_delta > 0 and not regressed,
            reason=(
                f"regression logit delta {logit_delta:.4f}"
                if regressed
                else "metric_signal"
            ),
            target_delta=ridge_delta,
            regression_deltas={"logit": logit_delta},
        )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verdict", fake_verdict
    )

    confirmed, detail = _confirm_dev(
        tmp_path / "source",
        tmp_path / "experiment",
        ("ridge", "logit"),
        ("ridge",),
        ("ridge", "logit"),
        evidence_only=True,
    )

    assert calls == [True, True, True, True]
    assert detail["mode"] == "metric_signal"
    assert detail["seed_confirmed"] is True
    assert detail["shadow"]["reason"].startswith("regression logit")
    assert confirmed is False


def test_ts_evaluator_rejects_training_on_holdout_targets():
    from fedotllm.agents.evolve.evaluation.scorer import _assert_frozen_ts_boundary

    series = np.arange(10, dtype=float)
    train = SimpleNamespace(target=series[:8], idx=np.arange(8))
    test = SimpleNamespace(target=series[8:], idx=np.arange(8, 10))
    _assert_frozen_ts_boundary(series, 2, train, test)

    leaked = SimpleNamespace(target=series, idx=np.arange(10))
    with pytest.raises(RuntimeError, match="training rows must not include"):
        _assert_frozen_ts_boundary(series, 2, leaked, test)


def test_feedback_loop_refines_neutral_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    _accept_behavior_probe(monkeypatch)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidates = [
        PatchCandidate("c1", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]),
        PatchCandidate("c2", edits=[PatchEdit("fedot/a.py", "return 1", "return 3")]),
    ]
    feedback_seen: list[str] = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost",), ("catboost",)),
    )

    def fake_fix(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    stock = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    patched = [
        {"catboost": ScoreResult("catboost", "ok", 0.80)},
        {"catboost": ScoreResult("catboost", "ok", 0.82)},
    ]
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fake_fix)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: patched.pop(0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._render_candidate_patch",
        lambda *_a, **_k: "diff",
    )

    decision = run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=3,
        policy=FAST_RUN_POLICY,
    )
    assert decision.keep
    assert len(feedback_seen) == 2
    assert feedback_seen[0] == ""
    assert "outcome=neutral" in feedback_seen[1]
    assert "Previous evaluated patch:" in feedback_seen[1]
    assert "return 2" in feedback_seen[1]
    assert "materially different causal mechanism" in feedback_seen[1]
    rows = [
        json.loads(line)
        for line in (tmp_path / "work" / "hypotheses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    hypotheses = [row for row in rows if row["event"] == "hypothesis"]
    assert len(hypotheses) == 2
    assert hypotheses[1]["parent_id"] == hypotheses[0]["id"]
    summary = json.loads(
        (tmp_path / "work" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert summary["immutable_source"] is True
    assert not list((tmp_path / "work" / "experiments").glob("*/*"))


def test_revision_feedback_keeps_prior_rejected_mechanism_visible():
    from fedotllm.agents.evolve.controller.campaign import _revision_feedback_context

    pytest_feedback = (
        "outcome=test_failures; failed_nodes=test_params_filter_with_non_default\n"
        "failing_test_contract_source:\n"
        "assert len(list(updated_params.keys())) == 1\n"
        "Previous evaluated patch:\n"
        "self.params.update(weights='distance')"
    )
    probe_feedback = (
        "outcome=behavior_probe_no_change; stock and patched both printed uniform"
    )

    result = _revision_feedback_context([pytest_feedback, probe_feedback])

    assert "Persistent branch constraints" in result
    assert "test_params_filter_with_non_default" in result
    assert "self.params.update(weights='distance')" in result
    assert "behavior_probe_no_change" in result
    assert "probe-only failure permits the same source edits" in result
    assert len(result) <= 8_000

    long_result = _revision_feedback_context(
        [pytest_feedback * 500, probe_feedback * 500],
        max_chars=8_000,
    )
    assert long_result.startswith("Persistent branch constraints:")
    assert len(long_result) == 8_000


def test_test_result_diagnostics_preserves_infrastructure_evidence():
    from fedotllm.agents.evolve.controller.campaign import _test_result_diagnostics

    result = _test_result_diagnostics(
        TestResult(
            "execution_error",
            -9,
            output="worker output\nprocess killed",
            duration_s=183.5,
            cmd="python -m pytest test/unit",
        )
    )

    assert '"status": "execution_error"' in result
    assert '"exit_code": -9' in result
    assert "python -m pytest test/unit" in result
    assert "process killed" in result


def test_focused_keep_must_pass_global_protect_before_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    _accept_behavior_probe(monkeypatch)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidates = [
        PatchCandidate(
            "shared", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]
        ),
        PatchCandidate(
            "scoped", edits=[PatchEdit("fedot/a.py", "return 1", "return 3")]
        ),
    ]
    feedback_seen: list[str] = []
    patched_results = [
        {"catboost": ScoreResult("catboost", "ok", 0.82)},
        {"lgbm": ScoreResult("lgbm", "ok", 0.70)},
        {"catboost": ScoreResult("catboost", "ok", 0.82)},
        {"lgbm": ScoreResult("lgbm", "ok", 0.80)},
    ]

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost",), ("catboost", "lgbm")),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )

    def fake_fix(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    def fake_stock(ids, **_kwargs):
        return {task_id: ScoreResult(task_id, "ok", 0.80) for task_id in ids}

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fake_fix)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock", fake_stock
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: patched_results.pop(0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._render_candidate_patch",
        lambda *_a, **_k: "diff",
    )

    workspace = tmp_path / "global-protect"
    decision = run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        # The first local KEEP fails at the nominal limit. One dedicated safety
        # revision must still be available to narrow the patch scope.
        max_revisions=1,
        policy=FAST_RUN_POLICY,
    )

    assert decision.keep is True
    assert len(feedback_seen) == 2
    assert "outcome=regressed" in feedback_seen[1]
    assert "Mandatory global protect suite rejected" in feedback_seen[1]
    rows = [
        json.loads(line)
        for line in (workspace / "journal.jsonl").read_text().splitlines()
    ]
    safety = [row for row in rows if row.get("event") == "global_dev_protect"]
    assert safety[0]["decision"]["reason"].startswith("regression global protect")
    assert safety[-1]["decision"]["keep"] is True


def test_last_revision_pytest_feedback_gets_one_contract_repair_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    _accept_behavior_probe(monkeypatch)
    contract_test = source / "test/unit/test_contract.py"
    contract_test.parent.mkdir(parents=True)
    contract_test.write_text(
        "def test_width():\n"
        "    observed = 75\n"
        "    expected = 85\n"
        "    assert (\n"
        "        observed == expected\n"
        "    )\n",
        encoding="utf-8",
    )
    lead = PatchSite("execution", "fedot/a.py", 1, "executed value")
    candidates = [
        PatchCandidate(
            "break-contract",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
        ),
        PatchCandidate(
            "repair-contract",
            edits=[PatchEdit("fedot/a.py", "return 1", "return 3")],
        ),
    ]
    feedback_seen: list[str] = []
    tests = [
        TestResult(
            "test_failures",
            1,
            failed_nodes={"test/unit/test_contract.py::test_width"},
            output="E assert 75 == 85",
        ),
        TestResult(
            "test_failures",
            1,
            failed_nodes={"test/unit/test_contract.py::test_width"},
            output="E assert 75 == 85",
        ),
        TestResult("passed", 0),
    ]
    stock = {"catboost": ScoreResult("catboost", "ok", 0.8)}

    def fake_fix(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(loop, "fix_lead", fake_fix)
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "_hydrate_configuration_surfaces", lambda *a, **k: 0)
    monkeypatch.setattr(
        loop,
        "measure_baseline_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )
    monkeypatch.setattr(loop, "measure_fedot_tests", lambda *_a, **_k: tests.pop(0))
    monkeypatch.setattr(loop, "measure_patched", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "diff")

    workspace = tmp_path / "contract-extension"
    decision = loop.run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=1,
        policy=FAST_RUN_POLICY,
    )

    assert decision.keep is False
    assert len(feedback_seen) == 2
    assert "test_contract.py::test_width" in feedback_seen[1]
    assert "pytest_failure_details" in feedback_seen[1]
    assert "failing_test_contract_source" in feedback_seen[1]
    assert "observed == expected" in feedback_seen[1]
    rows = [
        json.loads(line)
        for line in (workspace / "journal.jsonl").read_text().splitlines()
    ]
    extension = [
        row for row in rows if row.get("event") == "targeted_revision_extension"
    ]
    assert len(extension) == 1
    assert extension[0]["gate"] == "pytest"


def test_pytest_contract_source_is_bounded_and_rejects_path_escape(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import pytest_contract_source

    source = _source(tmp_path)
    test_file = source / "test/unit/test_contract.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_multiline_contract(value):\n"
        "    observed = value + 1\n"
        "    assert (\n"
        "        observed > value\n"
        "    )\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.py"
    outside.write_text("def test_secret():\n    assert False\n", encoding="utf-8")

    result = pytest_contract_source(
        source,
        {
            "test/unit/test_contract.py::test_multiline_contract[1]",
            "test/../../outside.py::test_secret",
        },
        max_chars=1_000,
    )

    assert "test_multiline_contract" in result
    assert "observed > value" in result
    assert "test_secret" not in result
    assert len(result) <= 1_000


def test_verified_bug_fix_is_correctness_keep_without_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.types import VerificationResult

    source = _source(tmp_path)
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "executed symbol value",
        evidence=("executed lines in this symbol: 1-2",),
        hypothesis_kind="correctness",
    )
    candidate = PatchCandidate(
        "correctness-fix",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
    )
    fixer_calls: list[str] = []

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verify_lead",
        lambda *_a, **_k: VerificationResult(
            "verified_bug",
            claim="valid input violates the return contract",
            reproduction_code="from fedot.a import value\nassert value() == 2\n",
        ),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.replay_reproduction",
        lambda *_a, **_k: {
            "status": "verified_bug",
            "stock": "failed_as_predicted",
            "patched": "resolved",
        },
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.fix_lead",
        lambda *_a, **_k: fixer_calls.append("called") or candidate,
    )
    stock = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    neutral = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: neutral,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )

    final_calls: list[str] = []
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._record_final",
        lambda *_a, **_k: final_calls.append("called"),
    )

    workspace = tmp_path / "work"
    decision = run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        verifier_inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=3,
        policy=EvolveRunPolicy(
            verify_manifest=False,
            confirm_and_ablate=False,
            confirm_small_signals=False,
            evaluate_final=True,
            fedot_quality_jobs=False,
        ),
    )

    assert fixer_calls == ["called"]
    assert decision.keep is True
    assert decision.correctness_keep is True
    assert decision.dev_keep is False
    assert decision.final_keep is None
    assert decision.stage == "correctness"
    assert decision.reason == "correctness_keep"
    assert final_calls == []
    saved = list((workspace / "correctness_fixes").glob("*.patch"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == "diff"
    summary = json.loads((workspace / "campaign_summary.json").read_text())
    assert summary["decision"]["correctness_keep"] is True
    assert summary["artifacts"]["correctness_patches"] == [
        f"correctness_fixes/{saved[0].name}"
    ]


def test_affected_metric_regression_drives_a_clean_policy_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "executed value policy",
        hypothesis_kind="correctness",
    )
    candidates = [
        PatchCandidate(
            "policy-a", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]
        ),
        PatchCandidate(
            "policy-b", edits=[PatchEdit("fedot/a.py", "return 1", "return 3")]
        ),
    ]
    feedback_seen: list[str] = []

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verify_lead",
        lambda *_a, **_k: VerificationResult(
            "verified_bug",
            reproduction_code=(
                "from fedot.a import value\n"
                "PipelineBuilder().add_node('lagged', params={'window_size': 100})\n"
                "assert value() > 1\n"
            ),
        ),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.replay_reproduction",
        lambda *_a, **_k: {
            "status": "verified_bug",
            "stock": "failed_as_predicted",
            "patched": "resolved",
        },
    )

    def fixer(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fixer)
    affected_calls = {"count": 0}

    def affected(*_args, **_kwargs):
        affected_calls["count"] += 1
        if affected_calls["count"] == 1:
            return {
                "status": "regressed",
                "operation": "lagged",
                "params": {"window_size": 100},
                "rows": [
                    {
                        "problem": "ts",
                        "metric": "holdout_rmse",
                        "lead_reached": True,
                        "classification": "regressed",
                        "normalized_delta": -0.2,
                    }
                ],
            }
        return {
            "status": "neutral",
            "operation": "lagged",
            "params": {"window_size": 100},
            "rows": [],
        }

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.evaluate_affected_metric", affected
    )
    stock = {"catboost": ScoreResult("catboost", "ok", 0.8)}
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )

    decision = run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        verifier_inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=2,
        policy=FAST_RUN_POLICY,
    )

    assert affected_calls["count"] == 2
    assert len(feedback_seen) == 2
    assert "affected_metric_status=regressed" in feedback_seen[1]
    assert "normalized_delta" in feedback_seen[1]
    assert decision.keep is True
    assert decision.correctness_keep is True
    assert decision.reason == "correctness_keep"


def test_confirmed_affected_metric_runs_final_once_and_stays_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "executed value policy",
        hypothesis_kind="correctness",
    )
    candidate = PatchCandidate(
        "policy", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verify_lead",
        lambda *_a, **_k: VerificationResult(
            "verified_bug",
            reproduction_code=(
                "from fedot.a import value\n"
                "PipelineBuilder().add_node('lagged', params={'window_size': 100})\n"
                "assert value() == 2\n"
            ),
        ),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.replay_reproduction",
        lambda *_a, **_k: {
            "status": "verified_bug",
            "stock": "failed_as_predicted",
            "patched": "resolved",
        },
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.fix_lead",
        lambda *_a, **_k: candidate,
    )
    first = {
        "status": "improved",
        "operation": "lagged",
        "params": {"window_size": 100},
        "seed": 42,
        "split": "dev",
        "rows": [],
    }
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.evaluate_affected_metric",
        lambda *_a, **_k: dict(first),
    )
    confirmation_splits: list[str] = []

    def confirm(*_args, split="dev", **_kwargs):
        confirmation_splits.append(split)
        return {
            "confirmed": True,
            "split": split,
            "improved_seeds": 3,
            "regressed_task_seed_pairs": 0,
            "infrastructure_failures": 0,
            "runs": [],
        }

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.confirm_affected_metric", confirm
    )
    scores = {
        "catboost": ScoreResult("catboost", "ok", 0.8),
        "rf": ScoreResult("rf", "ok", 0.8),
    }
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: scores,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: scores,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )

    decision = run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        verifier_inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost", "rf"),
        max_leads=1,
        max_revisions=1,
        policy=FAST_RUN_POLICY,
    )

    assert confirmation_splits == ["dev"]
    assert decision.keep is True
    assert decision.correctness_keep is True
    assert decision.stage == "correctness"
    assert decision.reason == "correctness_keep_with_metric_signal"


def test_test_failure_feedback_gets_a_clean_fixer_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    source = _source(tmp_path)
    _accept_behavior_probe(monkeypatch)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidates = [
        PatchCandidate(
            "broken", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]
        ),
        PatchCandidate(
            "repaired", edits=[PatchEdit("fedot/a.py", "return 1", "return 3")]
        ),
    ]
    feedback_seen: list[str] = []

    def fake_fix(*_args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return candidates.pop(0)

    test_results = [
        TestResult(
            "test_failures",
            1,
            failed_nodes={"test/unit/test_known.py::test_known_stock_failure"},
        ),
        TestResult(
            "test_failures",
            1,
            failed_nodes={
                "test/unit/test_known.py::test_known_stock_failure",
                "test/unit/test_contract.py::test_existing_contract",
            },
            output=(
                "FAILED test/unit/test_known.py::test_known_stock_failure\n"
                "FAILED test/unit/test_contract.py::test_existing_contract"
            ),
        ),
        TestResult(
            "test_failures",
            1,
            failed_nodes={
                "test/unit/test_known.py::test_known_stock_failure",
                "test/unit/test_contract.py::test_existing_contract",
            },
            output=(
                "FAILED test/unit/test_known.py::test_known_stock_failure\n"
                "FAILED test/unit/test_contract.py::test_existing_contract"
            ),
        ),
        TestResult(
            "test_failures",
            1,
            failed_nodes={"test/unit/test_known.py::test_known_stock_failure"},
        ),
    ]
    stock = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    improved = {"catboost": ScoreResult("catboost", "ok", 0.82)}
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost",), ("catboost",)),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fake_fix)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: improved,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: test_results.pop(0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._render_candidate_patch",
        lambda *_a, **_k: "diff",
    )

    decision = run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=2,
        policy=FAST_RUN_POLICY,
    )

    assert decision.keep
    assert len(feedback_seen) == 2
    assert "outcome=test_failures" in feedback_seen[1]
    assert "test_existing_contract" in feedback_seen[1]
    assert "test_known_stock_failure" not in feedback_seen[1]
    assert "return 2" in feedback_seen[1]


def test_failed_final_becomes_overall_drop_but_preserves_dev_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.transfer.evaluate_transfer",
        lambda *_a, **_k: (True, {"reason": "transfer_confirmed"}),
    )

    source = _source(tmp_path)
    _accept_behavior_probe(monkeypatch)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidate = PatchCandidate(
        "candidate", edits=[PatchEdit("fedot/a.py", "return 1", "return 3")]
    )
    stock = {"catboost": ScoreResult("catboost", "ok", 0.80)}
    patched = {"catboost": ScoreResult("catboost", "ok", 0.82)}
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost",), ("catboost",)),
    )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: [lead]
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.fix_lead",
        lambda *_a, **_k: candidate,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: stock,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: patched,
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.snapshot_diff",
        lambda *_a, **_k: "diff",
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._render_candidate_patch",
        lambda *_a, **_k: "diff",
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign._record_final",
        lambda *_a, **_k: Decision(
            keep=False,
            reason="final_confirmation_failed",
            target_delta=-0.03,
            regression_deltas={"catboost": -0.03},
            dev_keep=None,
            final_keep=False,
            stage="final",
        ),
    )

    decision = run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=1,
        policy=FINAL_RUN_POLICY,
    )

    assert decision.keep is False
    assert decision.dev_keep is True
    assert decision.final_keep is False
    assert decision.stage == "final"
    assert decision.reason == "final_confirmation_failed"
    assert decision.target_delta == -0.03


def test_execution_ranking_keeps_one_best_symbol_per_file(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    source = _source(tmp_path)
    rows = [
        {
            "file_path": "fedot/a.py",
            "symbol": "Container",
            "kind": "class",
            "line": 1,
            "count": 12,
            "line_ranges": [[1, 12]],
        },
        {
            "file_path": "fedot/a.py",
            "symbol": "Container.value",
            "kind": "method",
            "line": 1,
            "count": 2,
            "line_ranges": [[1, 2], [7, 7]],
        },
        {"file_path": "fedot/b.py", "symbol": "value", "line": 1, "count": 4},
    ]
    leads = discover_leads(source, limit=10, execution=rows)
    executed = [lead for lead in leads if lead.channel == "execution"]
    assert {lead.file_path for lead in executed[:2]} == {"fedot/a.py", "fedot/b.py"}
    selected_a = next(lead for lead in executed if lead.file_path == "fedot/a.py")
    assert selected_a.why.endswith("Container.value")
    assert "executed lines in this symbol: 1-2,7" in selected_a.evidence
    assert "executed metric-bearing lines in file: 1-2,7" in selected_a.evidence
    assert "method" in selected_a.signals


def test_worker_coverage_keeps_exact_ranges_and_qualified_method():
    import ast

    from fedotllm.agents.evolve.evaluation._worker import (
        _line_ranges,
        _qualified_symbol,
    )

    tree = ast.parse("class A:\n    def transform(self, data):\n        return data\n")
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

    assert _line_ranges({2, 3, 7, 9, 10}) == [[2, 3], [7, 7], [9, 10]]
    assert _qualified_symbol(method, parents) == ("A.transform", "method")


def test_execution_ranking_prefers_concrete_runtime_implementation_over_registry(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import (
        _execution_causal_priority,
        discover_leads,
    )

    source = _source(tmp_path)
    registry = (
        source / "fedot" / "core" / "repository" / "operation_types_repository.py"
    )
    implementation = (
        source
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "ts_transformations.py"
    )
    dispatcher = source / "fedot" / "core" / "operations" / "operation.py"
    registry.parent.mkdir(parents=True, exist_ok=True)
    implementation.parent.mkdir(parents=True, exist_ok=True)
    dispatcher.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("def load_repository():\n    return {}\n", encoding="utf-8")
    dispatcher.write_text("def predict():\n    return None\n", encoding="utf-8")
    implementation.write_text(
        "class LaggedImplementation:\n"
        "    def transform(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    runtime = (
        "runtime operation instances: "
        "lagged/LaggedTransformationImplementation fit params={} width=1->97"
    )
    rows = [
        {
            "file_path": dispatcher.relative_to(source).as_posix(),
            "symbol": "Operation._predict",
            "kind": "method",
            "line": 1,
            "count": 200,
            "runtime": runtime,
        },
        {
            "file_path": registry.relative_to(source).as_posix(),
            "symbol": "OperationTypesRepository.load",
            "kind": "method",
            "line": 1,
            "count": 100,
            "runtime": runtime,
        },
        {
            "file_path": implementation.relative_to(source).as_posix(),
            "symbol": "LaggedImplementation.transform",
            "kind": "method",
            "line": 2,
            "count": 5,
            "runtime": runtime,
        },
    ]

    leads = discover_leads(source, limit=10, execution=rows)
    executed = [lead for lead in leads if lead.channel == "execution"]

    assert executed[0].file_path.endswith("ts_transformations.py")
    dispatcher_lead = next(
        lead for lead in executed if lead.file_path.endswith("operations/operation.py")
    )
    assert _execution_causal_priority(executed[0]) == 0
    assert _execution_causal_priority(dispatcher_lead) == 4
    import_only = PatchSite(
        "execution",
        "fedot/core/operations/evaluation/operation_implementations/models/arima.py",
        1,
        "executed symbol ARIMAImplementation",
        evidence=("runtime line hits: 20", runtime),
        signals=("executed", "class"),
    )
    assert _execution_causal_priority(import_only) == 5

    abstract_interface = PatchSite(
        "execution",
        "fedot/core/operations/evaluation/operation_implementations/implementation_interfaces.py",
        20,
        "executed symbol _convert_to_output_function",
        evidence=("runtime line hits: 500", runtime),
        signals=("executed", "function", "registry"),
    )
    assert _execution_causal_priority(abstract_interface) == 4


def test_runtime_context_keeps_full_params_only_for_related_implementation():
    from fedotllm.agents.evolve.controller.campaign import _runtime_operation_evidence

    rows = (
        {
            "operation": "catboost",
            "implementation": "FedotCatBoostClassificationImplementation",
            "stage": "fit",
            "params": {"depth": 5, "learning_rate": 0.03},
            "supported_parameters": ["depth", "learning_rate", "l2_leaf_reg"],
            "input": {"active_width": 30},
            "output": {"active_width": 1},
        },
        {
            "operation": "lgbm",
            "implementation": "FedotLightGBMClassificationImplementation",
            "stage": "fit",
            "params": {"num_leaves": 31},
            "supported_parameters": ["num_leaves", "learning_rate"],
            "input": {"active_width": 30},
            "output": {"active_width": 1},
        },
    )

    concrete = _runtime_operation_evidence(
        rows,
        symbol="FedotCatBoostClassificationImplementation.fit",
    )
    assert "params=" in concrete and "supports=" in concrete
    assert "catboost/FedotCatBoostClassificationImplementation" in concrete
    assert "lgbm/FedotLightGBMClassificationImplementation" not in concrete

    shared = _runtime_operation_evidence(
        rows, symbol="divide_data_categorical_numerical"
    )
    assert "catboost/FedotCatBoostClassificationImplementation" in shared
    assert "lgbm/FedotLightGBMClassificationImplementation" in shared
    assert "params=" not in shared and "supports=" not in shared


def test_execution_ranking_demotes_imported_function_definition(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import (
        _execution_causal_priority,
        discover_leads,
    )

    source = _source(tmp_path)
    execution = [
        {
            "file_path": "fedot/a.py",
            "symbol": "unrelated_helper",
            "kind": "function",
            "line": 1,
            "count": 100,
            "line_ranges": [[1, 1]],
            "body_executed": False,
        },
        {
            "file_path": "fedot/b.py",
            "symbol": "LaggedImplementation.transform",
            "kind": "method",
            "line": 2,
            "count": 2,
            "line_ranges": [[3, 4]],
            "body_executed": True,
            "runtime": (
                "runtime operation instances: lagged/LaggedImplementation "
                "stage=predict params={} width=1->10"
            ),
        },
    ]

    leads = discover_leads(source, execution=execution, limit=10)
    executed = [lead for lead in leads if lead.channel == "execution"]

    assert executed[0].file_path == "fedot/b.py"
    imported = next(lead for lead in executed if lead.file_path == "fedot/a.py")
    assert "definition" in imported.signals
    assert _execution_causal_priority(imported) == 5


def test_coverage_body_execution_distinguishes_import_from_call():
    import ast

    from fedotllm.agents.evolve.evaluation._worker import _body_was_executed

    tree = ast.parse("def helper():\n    value = 1\n    return value\n")
    function = tree.body[0]

    assert not _body_was_executed(function, {1})
    assert _body_was_executed(function, {1, 2, 3})


def test_long_file_context_includes_inherited_fit_transform(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.context import context_from_lead

    source = _source(tmp_path)
    target = source / "fedot" / "long_operation.py"
    padding = "\n".join(f"PADDING_{i} = {i}" for i in range(410))
    target.write_text(
        "class RuntimeBase:\n"
        "    def fit(self, data):\n"
        "        return self.preprocess_input(data)\n\n"
        "    def transform(self, data):\n"
        "        return self.preprocess_input(data)\n\n"
        "    def preprocess_input(self, data):\n"
        "        return data.features\n\n"
        + padding
        + "\n\nclass ThinOperation(RuntimeBase):\n"
        "    def __init__(self):\n"
        "        self.model = object()\n",
        encoding="utf-8",
    )
    line = len(target.read_text(encoding="utf-8").splitlines()) - 1
    context = context_from_lead(
        PatchSite("oracle", "fedot/long_operation.py", line, "ThinOperation"),
        source,
    )

    assert "Inherited runtime behavior:" in context
    assert "Base class RuntimeBase:" in context
    assert "def fit(self, data):" in context
    assert "def transform(self, data):" in context


def test_long_file_context_retrieves_related_implementations(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.context import context_from_lead

    source = _source(tmp_path)
    target = source / "fedot" / "long_operation.py"
    padding = "\n".join(f"PADDING_{i} = {i}" for i in range(410))
    target.write_text(
        "class SharedRuntime:\n"
        "    def transform(self, data):\n"
        "        return data\n\n"
        + padding
        + "\n\nclass TargetOperation(SharedRuntime):\n"
        "    def __init__(self):\n"
        "        self.model = object()\n\n"
        "class HealthySibling(SharedRuntime):\n"
        "    def transform(self, data):\n"
        "        return data.features\n",
        encoding="utf-8",
    )
    lines = target.read_text(encoding="utf-8").splitlines()
    line = lines.index("class TargetOperation(SharedRuntime):") + 1

    context = context_from_lead(
        PatchSite("operation", "fedot/long_operation.py", line, "TargetOperation"),
        source,
    )

    assert "Related implementations sharing a base class" in context
    assert "Shares SharedRuntime" in context
    assert "class HealthySibling(SharedRuntime):" in context
    assert "protect their behavior" in context


def test_named_dependency_context_cannot_starve_delegated_helper(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.context import _named_runtime_sources

    source = _source(tmp_path)
    contract = source / "fedot" / "implementation_interfaces.py"
    contract.write_text(
        "class WrapperA:\n"
        "    def _convert_to_output(self, data, value):\n"
        "        return _convert_to_output_function(data, value)\n\n"
        "class WrapperB:\n"
        "    def _convert_to_output(self, data, value):\n"
        "        return _convert_to_output_function(data, value)\n\n"
        "class WrapperC:\n"
        "    def _convert_to_output(self, data, value):\n"
        "        return _convert_to_output_function(data, value)\n\n"
        "class WrapperD:\n"
        "    def _convert_to_output(self, data, value):\n"
        "        return _convert_to_output_function(data, value)\n\n"
        "def _convert_to_output_function(input_data, value):\n"
        "    return {'numerical_idx': input_data.numerical_idx}\n",
        encoding="utf-8",
    )

    context = _named_runtime_sources(
        source,
        ("_convert_to_output", "_convert_to_output_function"),
        limit=4,
    )

    assert "def _convert_to_output(" in context
    assert "def _convert_to_output_function(" in context
    assert "input_data.numerical_idx" in context


def test_context_traces_contract_fields_named_only_in_runtime_evidence(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.context import context_from_lead

    source = _source(tmp_path)
    consumer = source / "fedot" / "consumer.py"
    consumer.write_text(
        "def consume(data):\n    return data.categorical_idx, data.encoded_idx\n",
        encoding="utf-8",
    )
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "width-changing transform",
        evidence=(
            'output={"categorical_idx": {"within_width": false}, '
            '"encoded_idx": {"within_width": false}}',
        ),
    )

    context = context_from_lead(lead, source)

    assert "Data-contract fields named by runtime evidence" in context
    assert (
        "do not assume a direct Data field lives inside supplementary_data" in context
    )
    assert "Field categorical_idx:" in context
    assert "fedot/consumer.py" in context


def test_reproduction_feedback_prioritizes_patched_probe_failure():
    from fedotllm.agents.evolve.controller.campaign import (
        _compact_reproduction_feedback,
    )

    feedback = _compact_reproduction_feedback(
        {
            "status": "verified_bug",
            "stock": "failed_as_predicted",
            "patched": "still_failing",
            "claim": "stale numerical_idx",
            "observed": "stock trace " * 5_000,
            "patched_probe": {
                "status": "runtime_error",
                "exit_code": 1,
                "stderr": "prefix " * 1_000 + "IndexError: stale numerical_idx=26",
            },
        }
    )

    assert "IndexError: stale numerical_idx=26" in feedback
    assert "stock trace" not in feedback
    assert len(feedback) <= 4_000


def test_final_uses_run_once_workloads_not_global_exam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import _record_final
    monkeypatch.setattr("fedotllm.agents.evolve.controller.transfer.validate_transfer", lambda _: None)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.transfer.evaluate_transfer",
        lambda *_a, **_k: (True, {"reason": "transfer_confirmed"}),
    )

    source = _source(tmp_path)
    workspace = tmp_path / "work"
    calls: list[tuple[str, tuple[str, ...], str, int]] = []

    def fake_stock(ids, **kwargs):
        calls.append(("stock", ids, kwargs["split"], kwargs["seed"]))
        return {"catboost": ScoreResult("catboost", "ok", 0.80)}

    def fake_patched(ids, **kwargs):
        calls.append(("patched", ids, kwargs["split"], kwargs["seed"]))
        return {"catboost": ScoreResult("catboost", "ok", 0.82)}

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock", fake_stock
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched", fake_patched
    )
    candidate = PatchCandidate(
        "final-contract",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 3")],
    )
    decision = _record_final(
        source,
        workspace,
        workspace / "journal.jsonl",
        candidate,
        Decision(True, "keep", 0.02),
        run_id="run",
        seeds=(42, 43),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
    )
    assert decision is not None and decision.keep
    assert calls == [
        ("stock", ("catboost",), "final", 42),
        ("patched", ("catboost",), "final", 42),
    ]


def test_scout_can_search_symbols_and_pick_returned_runtime_path(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                assert "status=search" in prompt
                return SiteProposal(
                    status="search", query="def value", file_path="fedot/a.py"
                )
            if len(self.prompts) == 2:
                assert "fedot/b.py:1" in prompt
                assert "Catalog site: fedot/a.py:1" in prompt
                return SiteProposal(
                    status="symbol", query="value", file_path="fedot/a.py"
                )
            assert "Previous tool results:" in prompt
            assert "Catalog site: fedot/a.py:1" in prompt
            return SiteProposal(
                status="pick",
                file_path="fedot/b.py",
                line=1,
                why="consumer contract needs independent verification",
                **_causal_fields(1),
            )

    inference = Inference()
    selected = _llm_pick(
        inference,
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=4,
    )

    assert len(inference.prompts) == 3
    assert [(item.file_path, item.line) for item in selected] == [("fedot/b.py", 1)]


def test_scout_complete_pick_may_omit_current_catalog_path(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def create(self, prompt, _schema):
            assert "Catalog site: fedot/a.py:1" in prompt
            return SiteProposal(status="pick", **_causal_fields(1))

    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite(
                "execution",
                "fedot/a.py",
                1,
                "executed symbol value",
                evidence=("executed lines in this symbol: 1-2",),
                signals=("executed", "function"),
            ),
            # A later static row for the same file must not erase richer
            # runtime evidence during the LLM-pick handoff.
            PatchSite(
                "registry",
                "fedot/a.py",
                1,
                "registered operation",
                signals=("registry",),
            ),
        ],
        max_picks=1,
        max_actions=1,
    )

    assert len(selected) == 1
    assert selected[0].file_path == "fedot/a.py"
    assert selected[0].line == 1
    assert selected[0].evidence == (
        "catalog semantic site: fedot/a.py#executed-range:1",
        "executed lines in this symbol: 1-2",
    )
    assert selected[0].signals == ("executed", "function")


def test_scout_preserves_stock_crash_when_same_file_has_larger_generic_evidence(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def create(self, _prompt, _schema):
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                **_causal_fields(1),
            )

    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite(
                "operation",
                "fedot/a.py",
                1,
                "operation before crash",
                evidence=(
                    "stock runtime crash: IndexError: stale metadata",
                    "FEDOT frame chain:\nfedot/a.py:1 in value",
                    "workload operation value: fedot/a.py:1 (value)",
                ),
                signals=("executed", "upstream_of_crash"),
            ),
            PatchSite(
                "execution",
                "fedot/a.py",
                1,
                "executed symbol value",
                evidence=(
                    "runtime line hits: 100",
                    "executed metric-bearing lines in file: 1-2",
                    "generic evidence one",
                    "generic evidence two",
                ),
                signals=("executed", "function"),
            ),
        ],
        max_picks=1,
        max_actions=1,
    )

    assert len(selected) == 1
    assert any(item.startswith("stock runtime crash:") for item in selected[0].evidence)
    assert "executed metric-bearing lines in file: 1-2" in selected[0].evidence
    assert "upstream_of_crash" in selected[0].signals


def test_scout_rejects_unexecuted_line_inside_an_executed_file(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def create(self, _prompt, _schema):
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                change_line=10,
                mechanism="an untouched sibling branch might change predictions",
                proposed_change="replace that sibling branch",
                expected_metric_effect="improve the measured metric",
            )

    trace = {}
    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite(
                "execution",
                "fedot/a.py",
                1,
                "executed symbol value",
                evidence=(
                    "executed lines in this symbol: 1-2",
                    "executed metric-bearing lines in file: 1-2",
                ),
                signals=("executed", "function"),
            )
        ],
        max_picks=1,
        max_actions=1,
        trace=trace,
    )

    assert selected == []
    assert any(row["status"] == "unexecuted_pick" for row in trace["llm_pick_rounds"])


def test_shared_defaults_pick_must_name_the_executed_operation(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.selection import (
        SiteProposal,
        _default_pick_matches_executed_operation,
    )

    rel = "fedot/core/repository/data/default_operation_params.json"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        '{\n  "lagged": {\n    "window_size": 0\n  },\n'
        '  "sparse_lagged": {\n    "use_svd": false\n  }\n}\n'
    )
    evidence = (
        "executed operation: lagged",
        "executed operation: ridge",
    )
    sibling = PatchSite("llm", rel, 6, "change sparse lagged", evidence=evidence)
    assert not _default_pick_matches_executed_operation(
        SiteProposal(
            operation_id="sparse_lagged",
            proposed_change='change "use_svd" to true',
        ),
        sibling,
        tmp_path,
    )
    new_default = PatchSite("llm", rel, 8, "add ridge", evidence=evidence)
    assert _default_pick_matches_executed_operation(
        SiteProposal(
            operation_id="ridge",
            proposed_change='add "ridge": {"alpha": 10.0}',
        ),
        new_default,
        tmp_path,
    )


def test_shared_defaults_rejects_same_concrete_proposal_at_adjacent_line():
    from fedotllm.agents.evolve.discovery.selection import (
        SiteProposal,
        _matches_recent_concrete_proposal,
    )

    path = "fedot/core/repository/data/default_operation_params.json"
    proposal = SiteProposal(
        proposed_change=(
            'replace lagged with {"window_size": 0, "sparse_transform": true, '
            '"use_svd": true, "n_components": 0.5}'
        )
    )
    prior = [
        {
            "file_path": path,
            "line": 95,
            "proposed_change": (
                'Change entry to {"n_components": 0.5, "use_svd": true, '
                '"window_size": 0, "sparse_transform": true}'
            ),
        }
    ]
    assert _matches_recent_concrete_proposal(proposal, path, prior)
    proposal.proposed_change = '{"window_size": 0, "use_svd": false}'
    assert not _matches_recent_concrete_proposal(proposal, path, prior)


def test_shared_defaults_rejects_values_already_effective_at_runtime():
    from fedotllm.agents.evolve.discovery.selection import (
        SiteProposal,
        _default_pick_changes_effective_value,
    )

    lead = PatchSite(
        "llm",
        "fedot/core/repository/data/default_operation_params.json",
        72,
        "lgbm defaults",
        evidence=(
            'current FEDOT defaults: {"max_depth": -1}',
            'runtime implementation: LGBM; effective params: {"max_depth": -1}; '
            'estimator defaults: {"learning_rate": 0.1, "n_estimators": 100, '
            '"num_leaves": 31}',
        ),
    )
    inert = SiteProposal(
        proposed_change=(
            '{"max_depth": -1, "learning_rate": 0.1, '
            '"n_estimators": 100, "num_leaves": 31}'
        )
    )
    assert not _default_pick_changes_effective_value(inert, lead)
    inert.proposed_change = '{"learning_rate": 0.05, "n_estimators": 100}'
    assert _default_pick_changes_effective_value(inert, lead)


def test_scout_uses_declared_executed_change_line_as_patch_site(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def create(self, _prompt, _schema):
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=2,
                change_line=1,
                mechanism="the executed return changes model input",
                proposed_change="replace the executed return expression",
                expected_metric_effect="preserve more predictive information",
            )

    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite(
                "execution",
                "fedot/a.py",
                1,
                "executed symbol value",
                evidence=("executed metric-bearing lines in file: 1",),
                signals=("executed", "function"),
            )
        ],
        max_picks=1,
        max_actions=1,
    )

    assert [(item.file_path, item.line) for item in selected] == [("fedot/a.py", 1)]
    assert selected[0].mechanism == "the executed return changes model input"
    assert selected[0].proposed_change == "replace the executed return expression"
    assert selected[0].expected_metric_effect == "preserve more predictive information"


def test_scout_checkpoints_each_pick_before_catalog_walk_finishes(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    snapshots = []

    class Inference:
        def create(self, _prompt, _schema):
            return SiteProposal(status="pick", **_causal_fields(1))

    selected = _llm_pick(
        Inference(),
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=1,
        on_pick=lambda rows: snapshots.append(list(rows)),
    )

    assert snapshots == [selected]


def test_scout_rejects_cooled_site_after_llm_selects_change_line(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.a_calls = 0

        def create(self, prompt, _schema):
            assert "fedot/a.py:1" in prompt
            if "Catalog site: fedot/a.py:1" in prompt:
                self.a_calls += 1
                if self.a_calls == 1:
                    return SiteProposal(status="pick", **_causal_fields(1))
                assert "pick rejected: that exact change_line is cooling down" in prompt
                return SiteProposal(status="skip")
            assert "Catalog site: fedot/b.py:1" in prompt
            return SiteProposal(status="pick", **_causal_fields(1))

    trace: dict = {}
    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite("execution", "fedot/a.py", 1, "executed value"),
            PatchSite("execution", "fedot/b.py", 1, "executed other value"),
        ],
        max_picks=1,
        max_actions=3,
        excluded_sites={("fedot/a.py", 1)},
        trace=trace,
    )

    assert [(item.file_path, item.line) for item in selected] == [("fedot/b.py", 1)]
    assert any(
        row["status"] == "recent_site_cooldown"
        and row["file_path"] == "fedot/a.py"
        and row["line"] == 1
        for row in trace["llm_pick_rounds"]
    )


def test_scout_rejects_cooled_default_operation_after_neighbor_navigation(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    defaults_rel = "fedot/core/repository/data/default_operation_params.json"
    defaults = source / defaults_rel
    defaults.parent.mkdir(parents=True)
    defaults.write_text("{}\n", encoding="utf-8")

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return SiteProposal(
                    status="pick",
                    file_path=defaults_rel,
                    line=1,
                    change_line=1,
                    operation_id="logit",
                    mechanism="lower regularization changes fitted coefficients",
                    proposed_change='{"logit": {"C": 0.5}}',
                    expected_metric_effect="reduce variance on small classifications",
                )
            assert "logical source site is cooling down" in prompt
            return SiteProposal(status="skip")

    lead = PatchSite(
        "configuration",
        defaults_rel,
        1,
        "default parameters for executed operation logit",
        evidence=(
            "executed operation: logit",
            "current FEDOT defaults: {}",
            'runtime implementation: LogisticRegression; effective params: {"C": 1.0}; '
            'estimator defaults: {"C": 1.0}',
        ),
    )
    trace: dict = {}
    selected = _llm_pick(
        Inference(),
        source,
        [lead],
        max_picks=1,
        max_actions=2,
        excluded_semantic_sites={f"{defaults_rel}#operation:logit"},
        trace=trace,
    )

    assert selected == []
    assert trace["llm_pick_rounds"][0]["status"] == "recent_semantic_site_cooldown"


def test_discovery_prioritizes_measured_execution_over_generic_defaults(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    source = _source(tmp_path)
    defaults_rel = "fedot/core/repository/data/default_operation_params.json"
    defaults = source / defaults_rel
    defaults.parent.mkdir(parents=True)
    defaults.write_text("{}\n", encoding="utf-8")
    trace: dict = {}
    discover_leads(
        source,
        limit=10,
        trace=trace,
        execution=[
            {
                "file_path": "fedot/a.py",
                "symbol": "value",
                "line": 1,
                "count": 5,
                "kind": "function",
                "body_executed": True,
                "line_ranges": [[1, 2]],
            }
        ],
        trace_leads=[
            PatchSite(
                "configuration",
                defaults_rel,
                1,
                "default parameters for executed operation logit",
                evidence=("executed operation: logit",),
            )
        ],
    )

    assert trace["pool_rows_static"][0]["file_path"] == "fedot/a.py"


def test_discovery_does_not_mix_workloads_from_unrelated_symbols_in_one_file(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    source = _source(tmp_path)
    leads = discover_leads(
        source,
        limit=20,
        execution=[
            {
                "file_path": "fedot/a.py",
                "symbol": "value",
                "line": 1,
                "count": 5,
                "kind": "function",
                "body_executed": True,
                "line_ranges": [[1, 2]],
                "workload": "target workload",
                "runtime": "target runtime",
            },
            {
                "file_path": "fedot/a.py",
                "symbol": "ImportedDefinition",
                "line": 1,
                "count": 100,
                "kind": "function",
                "body_executed": False,
                "line_ranges": [[1, 1]],
                "workload": "unrelated workload",
                "runtime": "unrelated runtime",
            },
            {
                "file_path": "fedot/a.py",
                "symbol": "value",
                "line": 1,
                "count": 2,
                "kind": "function",
                "body_executed": False,
                "line_ranges": [[1, 1]],
                "workload": "same-symbol import-only workload",
                "runtime": "same-symbol import-only runtime",
            },
            {
                "file_path": "fedot/a.py",
                "symbol": "value",
                "line": 1,
                "count": 3,
                "kind": "function",
                "body_executed": True,
                "line_ranges": [[1, 2]],
                "workload": "second target workload",
                "runtime": "second target runtime",
            },
        ],
    )

    lead = next(item for item in leads if item.file_path == "fedot/a.py")
    evidence = "\n".join(lead.evidence)
    assert "target workload" in evidence
    assert "second target workload" in evidence
    assert "unrelated workload" not in evidence
    assert "unrelated runtime" not in evidence
    assert "same-symbol import-only workload" not in evidence
    assert "same-symbol import-only runtime" not in evidence


def test_scout_rejects_structured_pick_without_change_line(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def create(self, _prompt, _schema):
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                mechanism="an executed return changes model input",
                proposed_change="replace an unspecified line",
                expected_metric_effect="preserve more predictive information",
            )

    trace = {}
    selected = _llm_pick(
        Inference(),
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=1,
        trace=trace,
    )

    assert selected == []
    assert trace["llm_pick_rounds"][0]["status"] == "missing_change_line"


def test_scout_retries_incomplete_pick_with_causal_feedback(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return SiteProposal(status="pick")
            assert "pick rejected: fill mechanism" in prompt
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                **_causal_fields(1),
            )

    inference = Inference()
    trace = {}
    selected = _llm_pick(
        inference,
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=2,
        trace=trace,
    )

    assert inference.calls == 2
    assert [(item.file_path, item.line) for item in selected] == [("fedot/a.py", 1)]
    assert trace["llm_pick_rounds"][0]["status"] == "incomplete_pick_claim"


def test_frozen_workload_crash_is_authoritative_without_synthetic_reproduction(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import (
        is_controller_observed_crash,
        replay_reproduction,
        verification_from_observed_crash,
    )

    lead = PatchSite(
        "operation",
        "fedot/core/operations/model.py",
        42,
        "executed model before crash",
        evidence=(
            "stock runtime crash: IndexError: stale metadata",
            "FEDOT frame chain:\nfedot/core/operations/model.py:42 in fit",
            "workload operation model: fedot/core/operations/model.py:42 (Model.fit)",
        ),
        signals=("executed", "upstream_of_crash"),
        mechanism="metadata width reaches model fit",
        proposed_change="validate metadata against active width",
        expected_metric_effect="recover a currently crashing workload",
    )

    result = verification_from_observed_crash(lead)

    assert result is not None and result.proceed
    assert is_controller_observed_crash(result)
    replay = replay_reproduction(tmp_path, result)
    assert replay["stock"] == "failed_as_predicted"
    assert replay["patched"] is None
    assert replay["source"] == "controller_observed_frozen_workload_crash"


def test_regression_scope_rejects_cross_file_shared_base_edit(tmp_path: Path):
    from fedotllm.agents.evolve.agents.fixer import _regression_scope_diagnostics

    source = _source(tmp_path)
    transformations = source / "fedot/transforms.py"
    transformations.write_text(
        "class ComponentAnalysisImplementation:\n"
        "    def transform(self, data):\n"
        "        return data\n\n"
        "class PCAImplementation(ComponentAnalysisImplementation):\n"
        "    pass\n\n"
        "class FastICAImplementation(ComponentAnalysisImplementation):\n"
        "    pass\n",
        encoding="utf-8",
    )
    candidate = PatchCandidate(
        "candidate",
        edits=[
            PatchEdit(
                "fedot/transforms.py",
                "    def transform(self, data):\n        return data",
                "    def transform(self, data):\n        data.metadata = None\n        return data",
            )
        ],
    )
    lead = PatchSite(
        "llm",
        "fedot/a.py",
        1,
        mechanism="PCAImplementation emits stale metadata",
        proposed_change="clear PCA output metadata",
    )

    diagnostics = _regression_scope_diagnostics(
        source,
        lead,
        candidate,
        "outcome=regressed; reason=regression fast_ica->lgbm delta -0.04",
    )

    assert len(diagnostics) == 1
    assert "shared base ComponentAnalysisImplementation" in diagnostics[0]
    assert "FastICAImplementation" in diagnostics[0]
    assert "PCAImplementation" in diagnostics[0]


def test_dev_feedback_keeps_only_failed_or_regressed_runtime_dataflow():
    from fedotllm.agents.evolve.controller.campaign import _dev_feedback

    patched = {
        "improved": ScoreResult(
            "improved",
            "ok",
            0.9,
            dataflow=({"operation": "pca", "stage": "fit"},),
        ),
        "regressed": ScoreResult(
            "regressed",
            "ok",
            0.7,
            dataflow=({"operation": "fast_ica", "stage": "fit"},),
        ),
    }
    decision = Decision(
        False,
        "regression regressed delta -0.04",
        0.09,
        regression_deltas={"improved": 0.09, "regressed": -0.04},
    )

    stock = {
        "improved": ScoreResult("improved", "ok", 0.81),
        "regressed": ScoreResult("regressed", "ok", 0.74),
    }
    feedback = _dev_feedback(decision, patched, stock=stock)

    assert "regressed fast_ica/fit" in feedback
    assert "improved pca/fit" not in feedback


def test_neutral_dev_feedback_hides_unchanged_protect_crash():
    from fedotllm.agents.evolve.controller.campaign import _dev_feedback

    stock = {
        "healthy": ScoreResult("healthy", "ok", 0.8),
        "known-protect-crash": ScoreResult(
            "known-protect-crash", "crash", 0.5, detail="IndexError: known"
        ),
    }
    patched = {
        "healthy": ScoreResult("healthy", "ok", 0.8),
        "known-protect-crash": ScoreResult(
            "known-protect-crash", "crash", 0.5, detail="IndexError: known"
        ),
    }
    decision = Decision(
        False,
        "target_delta 0.0000 below per-task threshold",
        0.0,
        regression_deltas={"healthy": 0.0, "known-protect-crash": 0.0},
    )
    candidate = PatchCandidate(
        "neutral",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
    )

    feedback = _dev_feedback(decision, patched, candidate, stock=stock)

    assert "affected_task_statuses=none" in feedback
    assert "known-protect-crash" not in feedback
    assert "IndexError" not in feedback
    assert "Stay within this lead and its verified causal data flow" in feedback
    assert "do not target unrelated baseline failures" in feedback


def test_dev_feedback_exposes_only_status_changes_and_nonzero_deltas():
    from fedotllm.agents.evolve.controller.campaign import _dev_feedback

    stock = {
        "recovered": ScoreResult("recovered", "crash", 0.5),
        "unchanged-crash": ScoreResult("unchanged-crash", "crash", 0.5),
        "regressed": ScoreResult("regressed", "ok", 0.8),
    }
    patched = {
        "recovered": ScoreResult("recovered", "ok", 0.9),
        "unchanged-crash": ScoreResult("unchanged-crash", "crash", 0.5),
        "regressed": ScoreResult("regressed", "ok", 0.7),
    }
    decision = Decision(
        False,
        "regression regressed delta -0.1",
        0.4,
        regression_deltas={
            "recovered": 0.4,
            "unchanged-crash": 0.0,
            "regressed": -0.1,
        },
    )

    feedback = _dev_feedback(decision, patched, stock=stock)

    assert "recovered:ok" in feedback
    assert "regressed:ok" in feedback
    assert "unchanged-crash" not in feedback


def test_historical_duplicate_patch_returns_measured_feedback(tmp_path: Path):
    from fedotllm.agents.evolve.storage.findings import append_dev_rejudge
    from fedotllm.agents.evolve.storage.replay import patch_feedback_from_findings

    findings = tmp_path / "findings.jsonl"
    findings.write_text(
        json.dumps(
            {
                "record_type": "finding",
                "source_hash": "source-a",
                "score_protocol_hash": "score-a",
                "evaluation_protocol_hash": "protocol-a",
                "patch_hash": "patch-a",
                "outcome": "rejected_dev_regression",
                "edits": [
                    {
                        "file_path": "fedot/a.py",
                        "old_code": "return old",
                        "new_code": "return new",
                    }
                ],
                "dev": {
                    "reason": "regression sibling delta -0.04",
                    "target_delta": 0.09,
                    "regression_deltas": {"target": 0.09, "sibling": -0.04},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    feedback = patch_feedback_from_findings(
        findings,
        source_hash="source-a",
        patch_hash="patch-a",
        evaluation_protocol_hash="protocol-a",
    )

    assert "already evaluated" in feedback
    assert "regression sibling delta -0.04" in feedback
    assert '"sibling": -0.04' in feedback
    assert "return old" in feedback and "return new" in feedback
    assert not patch_feedback_from_findings(
        findings,
        source_hash="other-source",
        patch_hash="patch-a",
        evaluation_protocol_hash="protocol-a",
    )
    assert not patch_feedback_from_findings(
        findings,
        source_hash="source-a",
        patch_hash="patch-a",
        evaluation_protocol_hash="protocol-b",
    )

    append_dev_rejudge(
        findings,
        run_number=1,
        run_id="run-a",
        candidate_id="candidate-a",
        patch_hash="patch-a",
        source_hash="source-a",
        workspace=tmp_path,
        decision={
            "keep": False,
            "reason": "target_delta 0.22 below per-task threshold",
            "target_delta": 0.22,
            "regression_deltas": {"target": 0.22, "sibling": -0.003},
            "infrastructure_error": False,
        },
        reason="protect-only crash shortcut used the wrong workload",
        score_protocol_hash="score-a",
        evaluation_protocol_hash="protocol-a",
    )
    corrected = patch_feedback_from_findings(
        findings,
        source_hash="source-a",
        patch_hash="patch-a",
        evaluation_protocol_hash="protocol-a",
    )

    assert "target_delta 0.22 below per-task threshold" in corrected
    assert '"target": 0.22' in corrected
    assert "return old" in corrected and "return new" in corrected
    assert "regression sibling delta -0.04" not in corrected


def test_protocol_identity_ignores_docs_and_unselected_controller_fields(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.protocol import semantic_python_fingerprint

    module = tmp_path / "protocol_input.py"
    module.write_text(
        '"""first module docs"""\n'
        "def score(value):\n"
        '    """first function docs"""\n'
        "    return value + 1\n\n"
        "class ControllerConfig:\n"
        "    max_actions = 10\n",
        encoding="utf-8",
    )
    before = semantic_python_fingerprint(module, names=("score",))

    module.write_text(
        '"""changed module docs"""\n'
        "# formatting and comments are not score semantics\n"
        "def score(value):\n"
        '    """changed function docs"""\n'
        "    return value + 1\n\n"
        "class ControllerConfig:\n"
        "    max_actions = 999\n"
        "    new_field = True\n",
        encoding="utf-8",
    )
    metadata_only = semantic_python_fingerprint(module, names=("score",))

    module.write_text(
        "def score(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    changed_score = semantic_python_fingerprint(module, names=("score",))

    assert metadata_only == before
    assert changed_score != before


def test_confirmation_scope_uses_leaf_operation_and_rejects_shared_base(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.controller.campaign import _candidate_confirmation_scope

    source = _source(tmp_path)
    module = source / "fedot" / "transforms.py"
    module.write_text(
        "class ComponentAnalysisImplementation:\n"
        "    def transform(self, data):\n"
        "        return data\n\n"
        "class PCAImplementation(ComponentAnalysisImplementation):\n"
        "    def __init__(self):\n"
        "        self.n_components = 0.5\n\n"
        "class FastICAImplementation(ComponentAnalysisImplementation):\n"
        "    pass\n",
        encoding="utf-8",
    )
    stock = {
        "pca-task": ScoreResult(
            "pca-task",
            "crash",
            0.5,
            dataflow=({"operation": "pca", "implementation": "PCAImplementation"},),
        ),
        "ica-task": ScoreResult(
            "ica-task",
            "ok",
            0.8,
            dataflow=(
                {
                    "operation": "fast_ica",
                    "implementation": "FastICAImplementation",
                },
            ),
        ),
    }
    hints = {
        "pca-task": ("pca", "catboost"),
        "ica-task": ("fast_ica", "lgbm"),
    }
    exam_ids = ("pca-task", "ica-task")
    leaf = PatchCandidate(
        "leaf",
        edits=[
            PatchEdit(
                "fedot/transforms.py",
                "    def __init__(self):\n        self.n_components = 0.5",
                "    def __init__(self):\n        self.n_components = 0.9",
            )
        ],
    )
    shared = PatchCandidate(
        "shared",
        edits=[
            PatchEdit(
                "fedot/transforms.py",
                "    def transform(self, data):\n        return data",
                "    def transform(self, data):\n        data.metadata = None\n        return data",
            )
        ],
    )

    assert _candidate_confirmation_scope(source, leaf, stock, hints, exam_ids) == (
        "pca-task",
    )
    assert _candidate_confirmation_scope(source, shared, stock, hints, exam_ids) is None


def test_protect_only_workload_does_not_steer_scout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    observed: dict[str, object] = {}

    def fake_stock(task_ids, **_kwargs):
        rows = {
            "catboost": ScoreResult(
                "catboost",
                "ok",
                0.8,
                coverage=(
                    {
                        "file_path": "fedot/a.py",
                        "symbol": "value",
                        "line": 1,
                        "count": 1,
                    },
                ),
                dataflow=({"operation": "catboost", "implementation": "CatBoost"},),
            ),
            "pca->catboost": ScoreResult(
                "pca->catboost",
                "crash",
                0.5,
                traceback=(
                    f'  File "{source / "fedot" / "b.py"}", line 1, in value\n'
                    "IndexError: stale PCA metadata"
                ),
                coverage=(
                    {
                        "file_path": "fedot/b.py",
                        "symbol": "value",
                        "line": 1,
                        "count": 1,
                    },
                ),
                dataflow=({"operation": "pca", "implementation": "PCAImplementation"},),
            ),
        }
        return {task_id: rows[task_id] for task_id in task_ids}

    def fake_scout(_checkout, **kwargs):
        observed["execution"] = kwargs["execution"]
        observed["trace_leads"] = kwargs["trace_leads"]
        return []

    monkeypatch.setattr(loop, "measure_stock", fake_stock)
    monkeypatch.setattr(loop, "scout", fake_scout)
    monkeypatch.setattr(loop, "_hydrate_configuration_surfaces", lambda *a, **k: 0)
    inference = SimpleNamespace(usage={})

    decision = loop.run_once(
        checkout=source,
        scout_inference=inference,
        verifier_inference=inference,
        fixer_inference=inference,
        workspace=tmp_path / "workspace",
        findings_path=tmp_path / "findings.jsonl",
        lift_ids=("catboost",),
        protect_ids=("pca->catboost",),
        max_leads=1,
        policy=FAST_RUN_POLICY,
    )

    assert decision.reason == "no_lead"
    assert {row["file_path"] for row in observed["execution"]} == {"fedot/a.py"}
    assert all(lead.file_path != "fedot/b.py" for lead in observed["trace_leads"])


@pytest.mark.parametrize("read_failed", [False, True])
@pytest.mark.parametrize("max_actions,expected_scout_limit", [(20, 10), (80, 40)])
def test_run_once_passes_recent_completed_site_cooldown_to_scout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_failed: bool,
    max_actions: int,
    expected_scout_limit: int,
):
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.execution.checkout import source_fingerprint
    from fedotllm.agents.evolve.protocol import score_protocol_fingerprint

    source = _source(tmp_path)
    findings = tmp_path / "findings.jsonl"
    source_hash = source_fingerprint(source)
    protocol_hash = "prior-test-protocol"
    score_hash = score_protocol_fingerprint()
    cooled_site = ("fedot/a.py", 1)
    prior_rows = [
        {
            "record_type": "finding",
            "event": "finding",
            "run_number": 0,
            "run_id": "older-setting-trial",
            "source_hash": source_hash,
            "lead": {
                "file_path": "fedot/core/repository/data/default_operation_params.json",
                "line": 7,
                "evidence": ["executed operation: lgbm"],
            },
        },
        {
            "record_type": "run",
            "event": "run_start",
            "run_number": 1,
            "run_id": "prior-run",
            "source_hash": source_hash,
            "campaign_config": {
                "evaluation_protocol_hash": protocol_hash,
                "score_protocol_hash": score_hash,
            },
        },
        {
            "record_type": "finding",
            "event": "finding",
            "run_number": 1,
            "run_id": "prior-run",
            "source_hash": source_hash,
            "evaluation_protocol_hash": protocol_hash,
            "score_protocol_hash": score_hash,
            "lead": {"file_path": cooled_site[0], "line": cooled_site[1]},
        },
        {
            "record_type": "run",
            "event": "run_end",
            "run_number": 1,
            "run_id": "prior-run",
            "immutable_source": True,
        },
    ]
    findings.write_text(
        "\n".join(json.dumps(row) for row in prior_rows) + "\n",
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        loop,
        "measure_stock",
        lambda task_ids, **kwargs: {
            task_id: ScoreResult(task_id, "ok", 0.8) for task_id in task_ids
        },
    )
    monkeypatch.setattr(loop, "_hydrate_configuration_surfaces", lambda *a, **k: 0)

    def fake_scout(_checkout, **kwargs):
        observed["excluded_sites"] = kwargs["excluded_sites"]
        observed["excluded_semantic_sites"] = kwargs.get("excluded_semantic_sites")
        observed["trace"] = kwargs["trace"]
        observed["scout_limit"] = kwargs["max_actions"]
        if read_failed:
            kwargs["trace"]["llm_pick_rounds"] = [
                {"file_path": "fedot/b.py", "status": "error", "error_type": "APIError"}
            ]
        return []

    monkeypatch.setattr(loop, "scout", fake_scout)
    inference = SimpleNamespace(usage={})

    decision = loop.run_once(
        checkout=source,
        scout_inference=inference,
        verifier_inference=inference,
        fixer_inference=inference,
        workspace=tmp_path / "workspace",
        findings_path=findings,
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_actions=max_actions,
        policy=FAST_RUN_POLICY,
    )

    assert decision.reason == (
        "scout_incomplete: 1 failed reads" if read_failed else "no_lead"
    )
    assert decision.infrastructure_error == read_failed
    assert observed["scout_limit"] == expected_scout_limit
    assert cooled_site in observed["excluded_sites"]
    assert observed["excluded_semantic_sites"] == {"fedot/a.py:1"}
    assert observed["trace"]["recent_site_cooldown"] == [
        {"file_path": cooled_site[0], "line": cooled_site[1]}
    ]
    assert observed["trace"]["recent_semantic_site_cooldown"] == ["fedot/a.py:1"]


def test_scout_stops_catalog_walk_immediately_when_external_budget_is_exhausted(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.selection import _llm_pick

    source = _source(tmp_path)

    class ExhaustedInference:
        def __init__(self):
            self.calls = 0

        def create(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("experiment_budget_exhausted: no API request sent")

    inference = ExhaustedInference()
    trace: dict = {}
    result = _llm_pick(
        inference,
        source,
        [
            PatchSite("execution", "fedot/a.py", 1, "first executed site"),
            PatchSite("execution", "fedot/b.py", 1, "second executed site"),
        ],
        max_picks=1,
        max_actions=10,
        trace=trace,
    )

    assert result == []
    assert inference.calls == 1
    assert trace["scout_budget_exhausted"] is True
    assert trace["scout_external_stop"] == "experiment_budget_exhausted"
    assert trace["llm_pick_rounds"] == [
        {
            "file_path": "fedot/a.py",
            "status": "budget_exhausted",
            "error_type": "RuntimeError",
        }
    ]


def test_scout_returns_existing_picks_when_llm_stage_reserve_is_reached(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.selection import SiteProposal, _llm_pick
    from fedotllm.agents.evolve.storage.run_budget import EvolveBudgetExhausted

    source = _source(tmp_path)

    class ExhaustedAfterPickInference:
        def __init__(self):
            self.calls = 0

        def create(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SiteProposal(
                    status="pick",
                    file_path="fedot/a.py",
                    line=1,
                    why="concrete first site",
                    **_causal_fields(1),
                )
            raise EvolveBudgetExhausted(
                "evolve_run_stage_reserve_exhausted: no provider request sent"
            )

    inference = ExhaustedAfterPickInference()
    trace: dict = {}
    result = _llm_pick(
        inference,
        source,
        [
            PatchSite("execution", "fedot/a.py", 1, "first executed site"),
            PatchSite("execution", "fedot/b.py", 1, "second executed site"),
        ],
        max_picks=2,
        max_actions=10,
        trace=trace,
    )

    assert [item.file_path for item in result] == ["fedot/a.py"]
    assert inference.calls == 2
    assert trace["scout_budget_exhausted"] is True
    assert trace["scout_external_stop"] == "llm_budget_exhausted"
    assert trace["llm_pick_rounds"][-1] == {
        "file_path": "fedot/b.py",
        "status": "budget_exhausted",
        "error_type": "EvolveBudgetExhausted",
    }


def test_protect_only_crash_does_not_suppress_healthy_lift_evaluation():
    from fedotllm.agents.evolve.controller.campaign import _blocking_lift_crash_ids

    stock = {
        "healthy": ScoreResult("healthy", "ok", 0.8),
        "known-crash": ScoreResult("known-crash", "crash", 0.5),
    }

    assert _blocking_lift_crash_ids(stock, ("healthy",)) == ()
    assert _blocking_lift_crash_ids(stock, ("healthy", "known-crash")) == ()
    assert _blocking_lift_crash_ids(stock, ("known-crash",)) == ("known-crash",)


def test_resume_branch_restores_agent_lead_and_verification(tmp_path: Path):
    from fedotllm.agents.evolve.storage.replay import load_resume_branch

    workspace = tmp_path / "prior"
    workspace.mkdir()
    rows = [
        {
            "event": "verification",
            "hypothesis_id": "h-1",
            "status": "quality_hypothesis",
            "lead": {
                "channel": "llm",
                "file_path": "fedot/a.py",
                "line": 2,
            },
            "claim": "random selection can discard signal",
            "expected": "deterministic selection improves quality",
            "observed": "selection differs across fits",
            "current_approach": "random.sample",
            "proposed_approach": "train-only ranking",
            "alternatives_considered": ["seeded random"],
            "generality": "wide tables",
            "risks": ["classification tradeoff"],
        },
        {
            "event": "decision",
            "hypothesis_id": "h-1-r2",
            "candidate": "candidate-a",
            "patch_hash": "patch-a",
            "lead": {
                "channel": "llm",
                "file_path": "fedot/a.py",
                "line": 2,
                "why": "executed quality path",
                "mechanism": "random selection",
                "proposed_change": "rank columns",
                "expected_metric_effect": "retain signal",
            },
        },
    ]
    (workspace / "journal.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    branch = load_resume_branch(workspace, candidate_id="candidate-a")

    assert branch is not None
    assert branch["lead"].file_path == "fedot/a.py"
    assert branch["lead"].mechanism == "random selection"
    assert branch["verification"].status == "quality_hypothesis"
    assert branch["verification"].current_approach == "random.sample"
    assert branch["patch_hash"] == "patch-a"


def test_run_once_resume_skips_scout_and_reuses_measured_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    captured: dict[str, str] = {}
    lead = PatchSite("llm", "fedot/a.py", 1, mechanism="measured branch")
    verification = VerificationResult(
        "quality_hypothesis",
        claim="measured branch",
        expected="improve quality",
    )
    monkeypatch.setattr(
        loop,
        "measure_stock",
        lambda task_ids, **kwargs: {
            task_id: ScoreResult(task_id, "ok", 0.8) for task_id in task_ids
        },
    )
    monkeypatch.setattr(loop, "_hydrate_configuration_surfaces", lambda *a, **k: 0)
    monkeypatch.setattr(
        loop,
        "measure_baseline_fedot_tests",
        lambda *args, **kwargs: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        loop,
        "scout",
        lambda *args, **kwargs: pytest.fail("resume must not call Scout"),
    )
    monkeypatch.setattr(
        loop,
        "verify_lead",
        lambda *args, **kwargs: pytest.fail("resume must reuse Verifier result"),
    )

    def fake_fix(_checkout, _lead, **kwargs):
        captured["feedback"] = kwargs["feedback"]
        return None

    monkeypatch.setattr(loop, "fix_lead", fake_fix)
    inference = SimpleNamespace(usage={})

    decision = loop.run_once(
        checkout=source,
        scout_inference=inference,
        verifier_inference=inference,
        fixer_inference=inference,
        workspace=tmp_path / "continued",
        findings_path=tmp_path / "findings.jsonl",
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=1,
        resume_lead=lead,
        resume_verification=verification,
        resume_feedback="measured delta=0.22",
        policy=FAST_RUN_POLICY,
    )

    assert decision.reason == "no_patch"
    assert captured["feedback"] == "measured delta=0.22"


def test_scout_repairs_one_unambiguous_hallucinated_runtime_path(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    implementation = (
        source
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "ts_transformations.py"
    )
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        "class LaggedTransformationImplementation:\n"
        "    def transform(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    trace: dict = {}

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return SiteProposal(
                    status="read",
                    file_path=(
                        "fedot/core/operations/evaluation/operation_realisations/"
                        "ts_transformations.py"
                    ),
                )
            assert "class LaggedTransformationImplementation" in prompt
            assert "resolved from requested" in prompt
            return SiteProposal(
                status="pick",
                file_path=(
                    "fedot/core/operations/evaluation/operation_implementations/"
                    "data_operations/ts_transformations.py"
                ),
                line=1,
                why="lagged transform has a measurable causal mechanism",
                **_causal_fields(1),
            )

    selected = _llm_pick(
        Inference(),
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=2,
        trace=trace,
    )

    assert [item.file_path for item in selected] == [
        "fedot/core/operations/evaluation/operation_implementations/"
        "data_operations/ts_transformations.py"
    ]
    assert trace["llm_pick_rounds"][0]["status"] == "read_resolved"
    assert trace["llm_pick_rounds"][0]["requested_file_path"].endswith(
        "operation_realisations/ts_transformations.py"
    )


def test_scout_does_not_count_a_selected_file_twice(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    trace: dict = {}

    class Inference:
        def create(self, prompt, _schema):
            if "Catalog site: fedot/a.py:1" in prompt:
                return SiteProposal(status="pick", **_causal_fields(1))
            assert "Catalog site: fedot/b.py:1" in prompt
            assert (
                "Already selected files (do not pick them again):\n- fedot/a.py"
                in prompt
            )
            if "Previous tool results:\n(none)" in prompt:
                return SiteProposal(status="read", file_path="fedot/a.py", line=1)
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                **_causal_fields(1),
            )

    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
            PatchSite("execution", "fedot/b.py", 1, "executed symbol other"),
        ],
        max_picks=2,
        max_actions=3,
        trace=trace,
    )

    assert [(item.file_path, item.line) for item in selected] == [("fedot/a.py", 1)]
    assert any(row["status"] == "duplicate_pick" for row in trace["llm_pick_rounds"])


def test_scout_reserves_final_decision_after_four_navigation_turns(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls <= 4:
                assert "Final decision turn" not in prompt
                return SiteProposal(status="read", file_path="fedot/a.py", line=1)
            assert "Final decision turn" in prompt
            return SiteProposal(status="pick", **_causal_fields(1))

    inference = Inference()
    selected = _llm_pick(
        inference,
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=5,
    )

    assert inference.calls == 5
    assert [(item.file_path, item.line) for item in selected] == [("fedot/a.py", 1)]


def test_scout_does_not_repick_a_file_declined_in_the_same_catalog_walk(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    trace: dict = {}

    class Inference:
        def create(self, prompt, _schema):
            if "Catalog site: fedot/a.py:" in prompt:
                return SiteProposal(status="skip", why="no metric mechanism here")
            assert "Files explicitly declined earlier" in prompt
            if "Step 1/" in prompt:
                return SiteProposal(
                    status="read",
                    file_path="fedot/a.py",
                    line=1,
                    why="use the old file only as dependency context",
                )
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                why="attempt to revisit the declined file",
                **_causal_fields(1),
            )

    selected = _llm_pick(
        Inference(),
        source,
        [
            PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
            PatchSite("execution", "fedot/b.py", 1, "executed symbol other"),
        ],
        max_picks=1,
        max_actions=5,
        trace=trace,
    )

    assert selected == []
    assert trace["scout_declined_files"] == ["fedot/a.py"]
    assert any(row["status"] == "declined_repick" for row in trace["llm_pick_rounds"])


def test_scout_rejects_a_pick_that_semantically_says_to_skip(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, _llm_pick

    source = _source(tmp_path)
    selected = _llm_pick(
        type(
            "Inference",
            (),
            {
                "create": lambda self, prompt, schema: SiteProposal(
                    status="pick",
                    file_path="fedot/a.py",
                    line=1,
                    why=(
                        "This passthrough has no meaningful patch site and is "
                        "low-value; better to skip."
                    ),
                )
            },
        )(),
        source,
        [PatchSite("execution", "fedot/a.py", 1, "executed symbol value")],
        max_picks=1,
        max_actions=1,
    )

    assert selected == []


def test_verifier_preserves_lead_while_gathering_context_and_replays_probe(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import (
        VerificationProposal,
        replay_reproduction,
        verify_lead,
    )

    source = _source(tmp_path)
    reproduction = (
        "from fedot.a import value\n"
        "observed = value()\n"
        "assert observed == 2, f'expected 2, got {observed}'\n"
    )

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            assert "Lead context (always preserved):" in prompt
            assert "Site: fedot/a.py:1" in prompt
            if len(self.prompts) == 1:
                return VerificationProposal(action="search", query="def value")
            if len(self.prompts) == 2:
                assert "fedot/a.py:1" in prompt
                return VerificationProposal(action="symbol", query="value")
            if len(self.prompts) == 3:
                assert "def value" in prompt
                return VerificationProposal(
                    action="run",
                    run_code="from fedot.a import value\nprint(value())",
                )
            assert "action=run" in prompt
            return VerificationProposal(
                action="verify_bug",
                claim="value violates the supported return contract",
                expected="value() returns 2",
                observed="value() returns 1",
                reproduction_code=reproduction,
            )

    result = verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=Inference(),
        workspace=tmp_path / "work",
    )

    assert result.status == "verified_bug"
    assert (
        result.stock_probe is not None and result.stock_probe.status == "runtime_error"
    )
    assert (tmp_path / "work" / "verifications" / "a-1" / "reproduction.py").is_file()

    patched = tmp_path / "patched"
    import shutil

    shutil.copytree(source, patched)
    (patched / "fedot" / "a.py").write_text(
        "def value():\n    return 2\n", encoding="utf-8"
    )
    replay = replay_reproduction(patched, result)
    assert replay["stock"] == "failed_as_predicted"
    assert replay["patched"] == "resolved"


def test_verifier_rejects_bug_story_when_stock_satisfies_healthy_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 2)
    reproduction = (
        "from fedot.a import value\n"
        "observed = value()\n"
        "assert observed == 1, f'expected 1, got {observed}'\n"
    )

    class Inference:
        def create(self, _prompt, _schema):
            return verifier.VerificationProposal(
                action="verify_bug",
                claim="value allegedly violates its contract",
                reproduction_code=reproduction,
            )

    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=Inference(),
    )

    assert result.status == "rejected"
    assert "stock satisfied" in result.detail


def test_verifier_recovers_api_after_bug_probe_crashes_before_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    (source / "fedot/a.py").write_text(
        "class Operation:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 3)
    reproduction = (
        "from fedot.a import Operation\n"
        "value = Operation().fit(features=1)\n"
        "assert value == 2\n"
    )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return verifier.VerificationProposal(
                    action="verify_bug",
                    claim="fit allegedly violates a contract",
                    reproduction_code=reproduction,
                )
            assert "Automatic FEDOT API recovery" in prompt
            assert "exact method Operation.fit" in prompt
            assert "def fit(self, data)" in prompt
            return verifier.VerificationProposal(
                action="reject",
                why="the reproduction used an unsupported keyword",
            )

    inference = Inference()
    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 2, "executed Operation.fit"),
        inference=inference,
    )

    assert inference.calls == 2
    assert result.status == "rejected"


def test_verifier_reserves_probe_correction_for_valid_public_call_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    (source / "fedot/a.py").write_text(
        "def value():\n    raise ValueError('unexpected implementation crash')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 1)
    direct_crash = (
        "from fedot.a import value\nobserved = value()\nassert observed == 1\n"
    )
    asserted_contract = (
        "from fedot.a import value\n"
        "caught = None\n"
        "try:\n"
        "    value()\n"
        "except Exception as exc:\n"
        "    caught = exc\n"
        "assert caught is None, f'valid public call crashed: {caught}'\n"
    )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return verifier.VerificationProposal(
                    action="verify_bug",
                    claim="the supported public call crashes",
                    reproduction_code=direct_crash,
                )
            assert "extra correction turn has been reserved" in prompt.lower()
            assert "catch only the tested public call" in prompt.lower()
            return verifier.VerificationProposal(
                action="verify_bug",
                claim="the supported public call crashes",
                expected="value completes normally",
                observed="value raises ValueError",
                reproduction_code=asserted_contract,
            )

    inference = Inference()
    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed public value"),
        inference=inference,
    )

    assert inference.calls == 2
    assert result.status == "verified_bug"
    assert result.stock_probe is not None
    assert "AssertionError" in result.stock_probe.stderr


def test_verifier_accepts_grounded_exception_from_public_fedot_entry(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    operation = source / "fedot/core/operations/operation.py"
    operation.parent.mkdir(parents=True)
    (operation.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    (operation.parent / "__init__.py").write_text("", encoding="utf-8")
    operation.write_text(
        "class Operation:\n"
        "    def fit(self, data):\n"
        "        raise ValueError('implementation rejected valid data')\n",
        encoding="utf-8",
    )
    reproduction = (
        "from fedot.core.operations.operation import Operation\n"
        "result = Operation().fit(data=1)\n"
        "assert result == 1, 'valid fit must complete'\n"
    )

    class Inference:
        def create(self, _prompt, _schema):
            return verifier.VerificationProposal(
                action="verify_bug",
                claim="public Operation.fit crashes on valid data",
                expected="fit completes",
                observed="fit raises ValueError",
                file_path="fedot/core/operations/operation.py",
                reproduction_code=reproduction,
                why="the traceback reaches the claimed implementation",
            )

    result = verifier.verify_lead(
        source,
        PatchSite(
            "execution",
            "fedot/core/operations/operation.py",
            2,
            "executed Operation.fit",
        ),
        inference=Inference(),
    )

    assert result.status == "verified_bug"
    assert result.stock_probe is not None
    assert "ValueError" in result.stock_probe.stderr
    assert "public FEDOT call reached" in result.evidence[-1]


def test_verifier_does_not_accept_public_api_argument_mistake_as_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    operation = source / "fedot/core/operations/operation.py"
    operation.parent.mkdir(parents=True)
    (operation.parent.parent / "__init__.py").write_text("", encoding="utf-8")
    (operation.parent / "__init__.py").write_text("", encoding="utf-8")
    operation.write_text(
        "class Operation:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 1)
    monkeypatch.setattr(verifier, "MAX_VERIFY_PROBE_CORRECTIONS", 0)
    reproduction = (
        "from fedot.core.operations.operation import Operation\n"
        "result = Operation().fit(features=1)\n"
        "assert result == 1\n"
    )

    class Inference:
        def create(self, _prompt, _schema):
            return verifier.VerificationProposal(
                action="verify_bug",
                claim="fit allegedly crashes",
                file_path="fedot/core/operations/operation.py",
                reproduction_code=reproduction,
            )

    result = verifier.verify_lead(
        source,
        PatchSite(
            "execution",
            "fedot/core/operations/operation.py",
            2,
            "executed Operation.fit",
        ),
        inference=Inference(),
    )

    assert result.status != "verified_bug"


def test_verifier_requires_repository_evidence_before_quality_hypothesis(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import VerificationProposal, verify_lead

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return VerificationProposal(
                    action="quality_hypothesis",
                    claim="an unsupported guess",
                )
            if self.calls == 2:
                assert "requires at least one search" in prompt
                return VerificationProposal(action="callers", symbol="value")
            if self.calls == 3:
                return VerificationProposal(
                    action="quality_hypothesis",
                    claim="value is consumed by a runtime caller",
                    current_approach="a fixed value is returned",
                    proposed_approach="derive the value from fitted runtime state",
                    alternatives_considered=["fixed value", "fitted-state value"],
                    expected="changing its output can affect that operation",
                    generality="workloads that execute this operation",
                    risks=["the fitted estimate may be noisy"],
                )
            if self.calls == 4:
                assert "mandatory post-hypothesis challenge turn" in prompt.lower()
                assert _schema.__name__ == "QualityChallengeProposal"
                return VerificationProposal(action="symbol", query="value")
            return VerificationProposal(
                action="quality_hypothesis",
                claim="value is consumed by a runtime caller",
                current_approach="a fixed value is returned",
                proposed_approach="derive the value from fitted runtime state",
                alternatives_considered=["fixed value", "fitted-state value"],
                expected="changing its output can affect that operation",
                generality="workloads that execute this operation",
                risks=["the fitted estimate may be noisy"],
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )
    assert inference.calls == 5, result
    assert result.status == "quality_hypothesis"


def test_verifier_does_not_count_broken_probe_as_quality_challenge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier
    from fedotllm.agents.evolve.types import SnippetResult

    source = _source(tmp_path)
    monkeypatch.setattr(
        verifier,
        "run_fedot_snippet",
        lambda *_a, **_k: SnippetResult(
            "runtime_error",
            "from fedot import missing",
            stderr="AttributeError: guessed API does not exist",
        ),
    )

    def quality() -> verifier.VerificationProposal:
        return verifier.VerificationProposal(
            action="quality_hypothesis",
            claim="derive the output from fitted state",
            current_approach="return a fixed value",
            proposed_approach="use the already fitted state",
            alternatives_considered=["fixed value", "fitted-state value"],
            expected="the informative fitted state may improve the metric",
            generality="workloads executing this operation",
            risks=["small samples may be noisy"],
        )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return verifier.VerificationProposal(action="callers", symbol="value")
            if self.calls == 2:
                return quality()
            if self.calls == 3:
                return verifier.VerificationProposal(
                    action="run", run_code="from fedot import missing"
                )
            if self.calls == 4:
                return quality()
            if self.calls == 5:
                assert "requires a NEW adversarial" in prompt
                return verifier.VerificationProposal(action="symbol", query="value")
            return quality()

    inference = Inference()
    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )

    assert inference.calls == 6
    assert result.status == "quality_hypothesis"


def test_verifier_blocks_duplicate_navigation_and_reserves_final_verdict(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import VerificationProposal, verify_lead

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 3:
                assert "Duplicate tool action" in prompt
            if self.calls < 10:
                assert "mandatory final verdict turn" not in prompt.lower()
                return VerificationProposal(
                    action="read", file_path="fedot/a.py", line=1
                )
            assert "mandatory final verdict turn" in prompt.lower()
            return VerificationProposal(
                action="reject",
                claim="repeated reading found no metric mechanism",
                why="repository evidence does not justify a source change",
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )

    assert inference.calls == 10
    assert result.status == "rejected"


def test_verifier_reserves_challenge_when_hypothesis_arrives_on_final_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier

    source = _source(tmp_path)
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 3)

    def quality() -> verifier.VerificationProposal:
        return verifier.VerificationProposal(
            action="quality_hypothesis",
            claim="the executed fixed rule discards useful fitted information",
            current_approach="return the same fixed value for every fitted sample",
            proposed_approach="derive the value from already fitted runtime state",
            alternatives_considered=["fixed rule", "fitted-state adaptive rule"],
            expected="the retained fitted signal may improve the measured metric",
            generality="workloads that execute this operation",
            risks=["the fitted estimate may be noisy on small samples"],
        )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls < 3:
                return verifier.VerificationProposal(
                    action="read", file_path="fedot/a.py", line=self.calls
                )
            if self.calls == 3:
                assert "mandatory final verdict turn" in prompt.lower()
                assert _schema is verifier.FinalVerificationProposal
                return quality()
            if self.calls == 4:
                assert "Step 4/5" in prompt
                assert "mandatory post-hypothesis challenge turn" in prompt.lower()
                assert _schema is verifier.QualityChallengeProposal
                return verifier.VerificationProposal(action="symbol", query="value")
            assert "Step 5/5" in prompt
            assert "mandatory final verdict turn" in prompt.lower()
            assert _schema is verifier.FinalVerificationProposal
            return quality()

    inference = Inference()
    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )

    assert inference.calls == 5
    assert result.status == "quality_hypothesis"


def test_verifier_final_schema_excludes_navigation_actions():
    from pydantic import ValidationError

    from fedotllm.agents.evolve.agents.verifier import FinalVerificationProposal
    from fedotllm.agents.evolve.types import VerificationAction

    parsed = FinalVerificationProposal.model_validate({"action": "quality_hypothesis"})
    assert parsed.action is VerificationAction.QUALITY_HYPOTHESIS
    with pytest.raises(ValidationError):
        FinalVerificationProposal.model_validate(
            {"action": "search", "query": "another file"}
        )


def test_verifier_forces_source_challenge_after_broken_post_hypothesis_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.agents import verifier
    from fedotllm.agents.evolve.types import SnippetResult

    source = _source(tmp_path)
    monkeypatch.setattr(verifier, "MAX_VERIFY_STEPS", 4)
    monkeypatch.setattr(
        verifier,
        "run_fedot_snippet",
        lambda *_a, **_k: SnippetResult(
            "runtime_error",
            "from fedot import value",
            stderr="TypeError: invalid probe setup",
        ),
    )

    def quality() -> verifier.VerificationProposal:
        return verifier.VerificationProposal(
            action="quality_hypothesis",
            claim="the fixed rule discards useful fitted information",
            current_approach="return a fixed value",
            proposed_approach="derive the value from fitted runtime state",
            alternatives_considered=["fixed value", "fitted-state value"],
            expected="the retained fitted signal may improve the metric",
            generality="workloads executing this operation",
            risks=["small samples may be noisy"],
        )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, schema):
            self.calls += 1
            if self.calls == 1:
                return verifier.VerificationProposal(
                    action="read", file_path="fedot/a.py", line=1
                )
            if self.calls == 2:
                return quality()
            if self.calls == 3:
                assert schema is verifier.QualityChallengeProposal
                return verifier.VerificationProposal(
                    action="run", run_code="from fedot import value"
                )
            if self.calls == 4:
                assert schema is verifier.SourceQualityChallengeProposal
                assert "runtime `run` is now excluded" in prompt.lower()
                return verifier.VerificationProposal(
                    action="read", file_path="fedot/b.py", line=1
                )
            assert schema is verifier.FinalVerificationProposal
            assert "mandatory final verdict turn" in prompt.lower()
            return quality()

    inference = Inference()
    result = verifier.verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )

    assert inference.calls == 5, result
    assert result.status == "quality_hypothesis"


def test_verifier_blocks_near_duplicate_reads_and_requests_symbol(tmp_path: Path):
    from fedotllm.agents.evolve.agents.verifier import VerificationProposal, verify_lead

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                return VerificationProposal(
                    action="read", file_path="fedot/a.py", line=1
                )
            if self.calls == 2:
                return VerificationProposal(
                    action="read", file_path="fedot/a.py", line=2
                )
            assert "Near-duplicate read blocked" in prompt
            assert "action=symbol" in prompt
            return VerificationProposal(
                action="reject",
                why="the already shown region contains no metric mechanism",
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite("execution", "fedot/a.py", 1, "executed symbol value"),
        inference=inference,
    )

    assert inference.calls == 3
    assert result.status == "rejected"


def test_fixer_keeps_base_and_verifier_context_after_navigation(tmp_path: Path):
    from fedotllm.agents.evolve.agents.propose import PatchProposal, propose_patch

    source = _source(tmp_path)
    context = "LEAD_SENTINEL\n" + ("middle\n" * 8_000) + "VERIFIER_SENTINEL"

    class Inference:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            assert "LEAD_SENTINEL" in prompt
            assert "VERIFIER_SENTINEL" in prompt
            if len(self.prompts) == 1:
                return PatchProposal(
                    status="search",
                    file_path="fedot/a.py",
                    query="def value",
                )
            assert "action=search" in prompt
            assert len(prompt) < 40_000
            return PatchProposal(
                status="patch",
                file_path="fedot/a.py",
                old_code="return 1",
                new_code="return 2",
                rationale="test the return-value mechanism",
                behavior_probe="print('EVOLVE_OBSERVATION=2')",
            )

    inference = Inference()
    candidate = propose_patch(
        inference=inference,
        context=context,
        checkout=source,
    )
    assert candidate is not None
    assert candidate.behavior_probe == "print('EVOLVE_OBSERVATION=2')"
    assert inference.prompts and len(inference.prompts) == 2


def test_behavior_probe_rejects_semantic_noop_and_accepts_runtime_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.types import SnippetResult

    source = tmp_path / "source"
    experiment = tmp_path / "experiment"
    source.mkdir()
    experiment.mkdir()

    results = iter(
        [
            SnippetResult("ok", "probe", stdout="EVOLVE_OBSERVATION=old\n"),
            SnippetResult("ok", "probe", stdout="EVOLVE_OBSERVATION=old\n"),
            SnippetResult("ok", "probe", stdout="EVOLVE_OBSERVATION=old\n"),
            SnippetResult("ok", "probe", stdout="EVOLVE_OBSERVATION=new\n"),
        ]
    )
    monkeypatch.setattr(loop, "run_fedot_snippet", lambda *_a, **_k: next(results))

    no_change = loop.compare_behavior_probe(source, experiment, "print('probe')")
    changed = loop.compare_behavior_probe(source, experiment, "print('probe')")

    assert no_change["status"] == "no_change"
    assert no_change["probe_valid"] is True
    assert no_change["hypothesis_result"] == "toy_probe_did_not_see_effect"
    assert changed["status"] == "changed"
    assert changed["hypothesis_result"] == "mechanism_supported_pending_independent_gates"


def test_behavior_probe_requires_declared_observation_and_real_stock_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.types import SnippetResult

    results = iter(
        [
            SnippetResult("runtime_error", "probe", stderr="NameError: bad probe"),
            SnippetResult("ok", "probe", stdout="EVOLVE_OBSERVATION=new\n"),
        ]
    )
    monkeypatch.setattr(loop, "run_fedot_snippet", lambda *_a, **_k: next(results))

    result = loop.compare_behavior_probe(
        tmp_path / "source", tmp_path / "experiment", "print('probe')"
    )

    assert result["status"] == "invalid"
    assert result["probe_valid"] is False
    assert result["hypothesis_result"] == "inconclusive_invalid_probe"


def test_verifier_rejects_model_quality_threshold_as_bug_proof():
    from fedotllm.agents.evolve.agents.verifier import _probe_is_meaningful

    invalid = """from fedot.core.pipelines.pipeline_builder import PipelineBuilder
accuracy = 0.5
assert accuracy == 1.0, 'tiny model must be perfect'
"""
    valid = """import numpy as np
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
prediction = np.array([0.2, 0.8])
assert np.isfinite(prediction).all(), 'prediction must be finite'
"""

    accepted, detail = _probe_is_meaningful(invalid)
    assert accepted is False
    assert "quality threshold" in detail
    assert _probe_is_meaningful(valid) == (True, "")


@pytest.mark.parametrize(
    "first_status,second_probe,expected_probes,expected_metrics",
    [
        ("no_change", "print(2)", 2, 1),
        ("invalid", "print(2)", 2, 1),
        ("missing", "print(2)", 2, 1),
        ("patched_error", "print(2)", 2, 1),
        ("no_change", "print(1) # formatting only", 1, 0),
        ("changed", "print(2)", 1, 1),
    ],
)
def test_probe_repair_does_not_consume_an_unmeasured_patch(
    tmp_path,
    monkeypatch,
    first_status,
    second_probe,
    expected_probes,
    expected_metrics,
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed symbol value")
    candidates = iter(
        [
            PatchCandidate(
                str(i),
                edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
                behavior_probe=code,
            )
            for i, code in enumerate(("print(1)", second_probe))
        ]
    )
    probes, metrics = [], []
    stock = {"catboost": ScoreResult("catboost", "ok", 0.8)}

    def probe(_source, _experiment, code):
        probes.append(code)
        return {"status": first_status if len(probes) == 1 else "changed", "code": code}

    def measure(*_args, **_kwargs):
        metrics.append(True)
        return stock

    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(loop, "fix_lead", lambda *_a, **_k: next(candidates))
    monkeypatch.setattr(loop, "compare_behavior_probe", probe)
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "measure_patched", measure)
    monkeypatch.setattr(
        loop, "measure_fedot_tests", lambda *_a, **_k: TestResult("passed", 0)
    )
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "diff")
    decision = loop.run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=2,
        policy=FAST_RUN_POLICY,
    )

    assert len(probes) == expected_probes
    assert len(metrics) == expected_metrics
    assert not decision.keep  # A better probe must never grant metric acceptance.
    if expected_probes == 1:
        assert decision.reason.startswith("duplicate_patch")
    else:
        assert not decision.reason.startswith("duplicate_patch")


def test_duplicate_reason_refers_to_the_same_patch_not_intervening_attempt(
    tmp_path, monkeypatch
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    lead = PatchSite("execution", "fedot/a.py", 1, "executed value")
    candidates = iter(
        [
            PatchCandidate(
                str(i),
                edits=[PatchEdit("fedot/a.py", "return 1", f"return {value}")],
                behavior_probe=probe,
            )
            for i, value, probe in [(1, 2, "print(1)"), (2, 3, ""), (3, 2, "print(2)")]
        ]
    )
    stock = {"catboost": ScoreResult("catboost", "ok", 0.8)}
    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(loop, "fix_lead", lambda *_a, **_k: next(candidates))
    monkeypatch.setattr(
        loop,
        "compare_behavior_probe",
        lambda _s, _e, code: {
            "status": "missing"
            if not code
            else "no_change"
            if "-r2-" in _e.name
            else "changed",
            "code": code,
        },
    )
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "measure_patched", lambda *_a, **_k: stock)
    monkeypatch.setattr(
        loop, "measure_fedot_tests", lambda *_a, **_k: TestResult("passed", 0)
    )
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "diff")
    work = tmp_path / "work"
    decision = loop.run_once(
        checkout=source,
        workspace=work,
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost",),
        max_leads=1,
        max_revisions=3,
        policy=FAST_RUN_POLICY,
    )
    rows = [
        json.loads(line) for line in (work / "journal.jsonl").read_text().splitlines()
    ]
    attempts = [row for row in rows if row.get("event") == "decision"]
    assert attempts[1]["reason"] == "behavior_probe_missing"
    assert decision.reason == "duplicate_patch_after:" + attempts[0]["reason"]


def test_small_signal_with_unknown_scope_confirms_all_workloads(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    lead = PatchSite("execution", "fedot/a.py", 1, "shared implementation")
    candidate = PatchCandidate(
        "shared", edits=[PatchEdit("fedot/a.py", "return 1", "return 2")]
    )
    stock = {key: ScoreResult(key, "ok", 0.8) for key in ("catboost", "rf")}
    patched = {**stock, "catboost": ScoreResult("catboost", "ok", 0.805)}
    scopes = []

    def confirm(_s, _e, scope, lift, protect, **kwargs):
        scopes.append((scope, lift, protect, kwargs["evidence_only"]))
        return True, {"confirmed": True, "improved_seeds": 3, "shadow": {"keep": True}}

    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(loop, "fix_lead", lambda *_a, **_k: candidate)
    monkeypatch.setattr(
        loop, "compare_behavior_probe", lambda *_a, **_k: {"status": "changed"}
    )
    monkeypatch.setattr(loop, "_candidate_confirmation_scope", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: stock)
    monkeypatch.setattr(loop, "measure_patched", lambda *_a, **_k: patched)
    monkeypatch.setattr(
        loop, "measure_fedot_tests", lambda *_a, **_k: TestResult("passed", 0)
    )
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "diff")
    monkeypatch.setattr(loop, "_confirm_dev", confirm)
    result = loop.run_once(
        checkout=source,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost", "rf"),
        max_leads=1,
        max_revisions=1,
        policy=EvolveRunPolicy(
            verify_manifest=False,
            confirm_and_ablate=False,
            confirm_small_signals=True,
            evaluate_final=False,
            fedot_quality_jobs=False,
        ),
    )
    assert scopes == [(("catboost", "rf"), ("catboost",), ("catboost", "rf"), True)]
    assert result.keep
    assert result.metric_signal_keep
    assert result.reason == "confirmed_small_metric_keep"
    assert (tmp_path / "work" / "metric_signal_fixes" / "shared.patch").is_file()


def test_neutral_verified_quality_change_is_preserved_as_maintenance_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller import campaign as loop

    source = _source(tmp_path)
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "executed runtime invariant",
        mechanism="fit-derived state must be stable during predict",
        proposed_change="persist the state at fit time",
        expected_metric_effect="remove train/predict drift",
    )
    candidate = PatchCandidate(
        "maintenance",
        edits=[PatchEdit("fedot/a.py", "return 1", "return 2")],
        behavior_probe="print('EVOLVE_OBSERVATION=changed')",
    )
    scores = {
        "catboost": ScoreResult("catboost", "ok", 0.8),
        "rf": ScoreResult("rf", "ok", 0.8),
    }
    _accept_behavior_probe(monkeypatch)
    monkeypatch.setattr(loop, "scout", lambda *_a, **_k: [lead])
    monkeypatch.setattr(loop, "fix_lead", lambda *_a, **_k: candidate)
    monkeypatch.setattr(loop, "measure_stock", lambda *_a, **_k: scores)
    monkeypatch.setattr(loop, "measure_patched", lambda *_a, **_k: scores)
    monkeypatch.setattr(loop, "_hydrate_configuration_surfaces", lambda *a, **k: 0)
    monkeypatch.setattr(
        loop,
        "measure_baseline_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        loop, "measure_fedot_tests", lambda *_a, **_k: TestResult("passed", 0)
    )
    monkeypatch.setattr(
        loop, "_candidate_confirmation_scope", lambda *_a, **_k: ("catboost",)
    )
    monkeypatch.setattr(loop, "snapshot_diff", lambda *_a, **_k: "maintenance diff")

    workspace = tmp_path / "maintenance-run"
    decision = loop.run_once(
        checkout=source,
        workspace=workspace,
        inference=object(),
        lift_ids=("catboost",),
        protect_ids=("catboost", "rf"),
        max_leads=1,
        max_revisions=1,
        policy=FAST_RUN_POLICY,
    )

    assert decision.keep is True
    assert decision.maintenance_keep is True
    assert decision.correctness_keep is False
    assert decision.reason == "maintenance_keep"
    assert decision.stage == "maintenance"
    assert decision.final_keep is None
    assert (workspace / "maintenance_fixes/maintenance.patch").read_text() == (
        "maintenance diff"
    )
    summary = json.loads((workspace / "campaign_summary.json").read_text())
    assert summary["artifacts"]["maintenance_patches"] == [
        "maintenance_fixes/maintenance.patch"
    ]
    from fedotllm.agents.evolve.storage.scoreboard import summarize

    scoreboard = summarize(workspace)
    assert scoreboard["keeps"] == 1
    assert scoreboard["maintenance_keeps"] == 1
    assert scoreboard["getting_better"] is False


def test_historical_probe_failure_only_deduplicates_the_patch_probe_pair(tmp_path):
    from fedotllm.agents.evolve.storage.hypothesis import behavior_probe_fingerprint
    from fedotllm.agents.evolve.storage.replay import (
        rejected_probe_hashes_from_findings,
        tried_patch_hashes_from_findings,
        patch_feedback_from_findings,
    )

    path = tmp_path / "findings.jsonl"
    row = {
        "record_type": "finding",
        "source_hash": "source",
        "evaluation_protocol_hash": "protocol",
        "patch_hash": "patch",
        "behavior_probe": {"status": "no_change", "code": "print(1)"},
        "dev": {
            "reason": "behavior_probe_no_change",
            "patched": None,
            "target_delta": None,
        },
    }
    path.write_text(json.dumps(row) + "\n[]\ninvalid\n")
    scope = {"source_hash": "source", "evaluation_protocol_hash": "protocol"}
    assert tried_patch_hashes_from_findings(path, **scope) == set()
    assert rejected_probe_hashes_from_findings(path, **scope) == {
        "patch": {behavior_probe_fingerprint("print( 1 ) # same probe")},
    }
    assert not rejected_probe_hashes_from_findings(path, source_hash="other")
    assert not rejected_probe_hashes_from_findings(
        path, evaluation_protocol_hash="other"
    )
    assert behavior_probe_fingerprint("print(1)") != behavior_probe_fingerprint(
        "print(2)"
    )
    # The retrieval advice must match the controller's allowed next action.
    path.write_text(json.dumps(row) + "\n")
    feedback = patch_feedback_from_findings(path, patch_hash="patch", **scope)
    assert "before metric evaluation" in feedback
    assert "keep these source edits" in feedback
    measured = {
        **row,
        "dev": {"reason": "regression", "patched": {"task": {}}, "target_delta": -0.1},
    }
    # A measured rejection remains durable even if another probe-only failure follows it.
    path.write_text(json.dumps(measured) + "\n" + json.dumps(row) + "\n")
    assert tried_patch_hashes_from_findings(path, **scope) == {"patch"}
    feedback = patch_feedback_from_findings(path, patch_hash="patch", **scope)
    assert "reason=regression" in feedback
    assert "keep these source edits" not in feedback


def test_broken_behavior_probe_blocks_quality_candidate_before_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller import campaign as loop
    from fedotllm.agents.evolve.types import SnippetResult

    results = iter(
        [
            SnippetResult("runtime_error", "probe", stderr="TypeError: bad stock API"),
            SnippetResult(
                "runtime_error", "probe", stderr="TypeError: bad patched API"
            ),
        ]
    )
    monkeypatch.setattr(loop, "run_fedot_snippet", lambda *_a, **_k: next(results))

    result = loop.compare_behavior_probe(
        tmp_path / "source", tmp_path / "experiment", "print('probe')"
    )

    assert result["status"] == "invalid"
    assert loop._behavior_probe_blocks_candidate(result) is True
    assert loop._behavior_probe_blocks_candidate({"status": "changed"}) is False
    assert loop._behavior_probe_blocks_candidate({"status": "no_change"}) is False
    assert loop._behavior_probe_blocks_candidate({"status": "patched_error"}) is True


def test_lead_context_retrieves_frozen_fedot_docs_docstrings_and_metadata(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.context import context_from_lead
    from fedotllm.agents.evolve.discovery.research_tools import docs_runtime

    source = _source(tmp_path)
    (source / "fedot" / "a.py").write_text(
        "class WidgetOperation:\n"
        '    """Transforms a widget feature table for downstream nodes."""\n'
        "    def transform(self, data):\n"
        '        """Preserve row indices while changing widget features."""\n'
        "        return data\n",
        encoding="utf-8",
    )
    docs = source / "docs" / "source" / "advanced"
    docs.mkdir(parents=True)
    (docs / "architecture.rst").write_text(
        "Widget pipeline architecture\n"
        "============================\n\n"
        "WidgetOperation transforms feature tables while preserving row indices "
        "for downstream pipeline nodes.\n",
        encoding="utf-8",
    )
    repository = source / "fedot" / "core" / "repository" / "data"
    repository.mkdir(parents=True)
    (repository / "data_operation_repository.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "widget_transform": {
                        "input_type": "[DataTypesEnum.table]",
                        "output_type": "[DataTypesEnum.table]",
                        "description": "Widget feature transformation",
                    }
                },
                "operations": {
                    "widget": {
                        "meta": "widget_transform",
                        "tags": ["feature_engineering"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    lead = PatchSite(
        "execution",
        "fedot/a.py",
        1,
        "WidgetOperation.transform is executed in the pipeline",
    )
    context = context_from_lead(lead, source, max_chars=8_000)
    docs_result = docs_runtime(source, "WidgetOperation pipeline row indices")

    assert "Retrieved FEDOT architecture/docstring context" in context
    assert "preserving row indices" in context
    assert "operation_metadata" in context
    assert "architecture.rst" in docs_result
    assert "evaluator" not in docs_result.lower()


def test_snippet_failure_recovers_wrong_module_and_actual_api(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.research_tools import format_snippet_feedback
    from fedotllm.agents.evolve.types import SnippetResult

    source = _source(tmp_path)
    module = source / "fedot/core/operations/operation_parameters.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "class OperationParameters:\n"
        "    @staticmethod\n"
        "    def from_operation_type(operation_type, **parameters):\n"
        "        return OperationParameters()\n",
        encoding="utf-8",
    )
    result = SnippetResult(
        "runtime_error",
        ("from fedot.core.repository.operation_parameters import OperationParameters"),
        stdout="irrelevant training output\n" * 1_000,
        stderr=(
            "ModuleNotFoundError: No module named "
            "'fedot.core.repository.operation_parameters'"
        ),
    )

    feedback = format_snippet_feedback(source, result, max_chars=3_000)

    assert "Automatic FEDOT API recovery" in feedback
    assert "fedot/core/operations/operation_parameters.py" in feedback
    assert "def from_operation_type" in feedback
    assert "ModuleNotFoundError" in feedback
    assert len(feedback) <= 3_002


def test_snippet_failure_recovers_inherited_constructor_fields(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.research_tools import snippet_failure_context
    from fedotllm.agents.evolve.types import SnippetResult

    source = _source(tmp_path)
    module = source / "fedot/core/data/data.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "class Data:\n"
        "    idx: object\n"
        "    features: object\n"
        "    target: object = None\n\n"
        "class InputData(Data):\n"
        "    pass\n",
        encoding="utf-8",
    )
    result = SnippetResult(
        "runtime_error",
        "InputData(features_types={})",
        stderr=(
            "TypeError: InputData.__init__() got an unexpected keyword argument "
            "'features_types'"
        ),
    )

    recovery = snippet_failure_context(source, result)

    assert "constructor fields for InputData" in recovery
    assert "constructor fields for Data" in recovery
    assert "features: object" in recovery
    assert "features_types" not in recovery


def test_symbol_tool_normalizes_declaration_and_returns_full_ast_body(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.research_tools import symbol_runtime

    source = _source(tmp_path)
    methods = "\n".join(
        f"    def method_{index}(self):\n        return {index}" for index in range(30)
    )
    (source / "fedot" / "long_runtime.py").write_text(
        f"class LongRuntime:\n{methods}\n",
        encoding="utf-8",
    )

    result = symbol_runtime(source, "class LongRuntime", limit=1)

    assert "exact class LongRuntime" in result
    assert "def method_29" in result


def test_lead_rag_uses_semantic_evidence_not_unrelated_source_imports(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.knowledge import knowledge_for_lead

    source = _source(tmp_path)
    (source / "fedot" / "a.py").write_text(
        "from fedot.unrelated_gaussian_filter import Noise\n"
        "class LaggedRuntime:\n"
        "    pass\n",
        encoding="utf-8",
    )
    docs = source / "docs" / "source"
    docs.mkdir(parents=True)
    (docs / "forecasting.rst").write_text(
        "Lagged forecasting\n==================\n\n"
        "Lagged windows and sparse lagged alternatives serve forecasting workloads.\n",
        encoding="utf-8",
    )
    lead = PatchSite(
        "execution",
        "fedot/a.py",
        2,
        "lagged window mechanism for forecasting",
        evidence=("metric=holdout_rmse; operations=lagged,ridge",),
    )

    result = knowledge_for_lead(source, lead, max_chars=4_000)

    assert "Lagged forecasting" in result
    assert "unrelated_gaussian_filter" not in result


def test_scout_searches_one_metric_linked_change_without_preclassified_mode(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.prompt = ""

        def create(self, prompt, _schema):
            self.prompt = prompt
            return SiteProposal(
                status="pick",
                file_path="fedot/a.py",
                line=1,
                why=(
                    "the fixed current rule discards training information; compare "
                    "a data-adaptive alternative"
                ),
                **_causal_fields(1),
            )

    inference = Inference()
    leads = discover_leads(
        source,
        inference=inference,
        limit=3,
        max_picks=1,
        execution=[
            {
                "file_path": "fedot/a.py",
                "symbol": "value",
                "line": 1,
                "count": 9,
                "workload": (
                    "workload family=classification; pipeline operations=rf; "
                    "evaluation objective withheld during discovery"
                ),
            }
        ],
    )
    picked = next(lead for lead in leads if lead.channel == "llm")
    assert "classify the lead in advance" in inference.prompt
    assert "primary goal is one source change" in inference.prompt
    assert "workload family=classification" in inference.prompt
    assert "evaluation objective withheld during discovery" in inference.prompt
    assert "metric=holdout_roc_auc" not in inference.prompt
    assert "data-adaptive alternative" in picked.why


def test_scout_source_keeps_complete_medium_sized_method(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.context import (
        scout_source_context,
        show_source,
    )

    source = _source(tmp_path)
    body = [
        "class Operation:",
        "    def fit(self, data):",
        "        marker_start = data",
    ]
    body.extend(f"        value_{index} = {index}" for index in range(120))
    body.extend(
        [
            "        marker_end = value_119",
            "        return marker_end",
            "    def transform(self, data):",
            "        sibling_transform_contract = data",
            "        return sibling_transform_contract",
        ]
    )
    (source / "fedot/a.py").write_text("\n".join(body) + "\n", encoding="utf-8")

    rendered = show_source("fedot/a.py", checkout=source, around=65, radius=20)

    assert "marker_start = data" in rendered
    assert "marker_end = value_119" in rendered

    semantic = scout_source_context(
        PatchSite("execution", "fedot/a.py", 65, "executed method Operation.fit"),
        checkout=source,
    )
    assert "marker_start = data" in semantic
    assert "marker_end = value_119" in semantic
    assert "sibling_transform_contract = data" in semantic


def test_researcher_requires_grounded_comparative_quality_hypothesis(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import (
        VerificationProposal,
        verification_context,
        verify_lead,
    )

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, _schema):
            self.calls += 1
            if self.calls == 1:
                assert "quality_hypothesis" in prompt
                return VerificationProposal(
                    action="docs", query="value operation contract"
                )
            if self.calls == 2:
                return VerificationProposal(
                    action="quality_hypothesis",
                    proposed_approach="use an adaptive return rule",
                )
            if self.calls == 3:
                assert "must name the current mechanism" in prompt
                return VerificationProposal(
                    action="quality_hypothesis",
                    claim="an adaptive rule may use available training information",
                    current_approach="a fixed constant is returned for every supported input",
                    proposed_approach="derive the value from already fitted runtime state",
                    alternatives_considered=[
                        "fixed default",
                        "fitted-state adaptive rule",
                    ],
                    expected="improve the classification metric when the fitted state is informative",
                    generality="all classification workloads using this operation",
                    risks=["small samples may make the fitted estimate noisy"],
                )
            if self.calls == 4:
                assert "mandatory post-hypothesis challenge turn" in prompt.lower()
                assert _schema.__name__ == "QualityChallengeProposal"
                return VerificationProposal(action="symbol", query="value")
            assert _schema.__name__ == "FinalVerificationProposal"
            return VerificationProposal(
                action="quality_hypothesis",
                claim="an adaptive rule may use available training information",
                current_approach="a fixed constant is returned for every supported input",
                proposed_approach="derive the value from already fitted runtime state",
                alternatives_considered=["fixed default", "fitted-state adaptive rule"],
                expected="improve the classification metric when the fitted state is informative",
                generality="all classification workloads using this operation",
                risks=["small samples may make the fitted estimate noisy"],
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite(
            "execution",
            "fedot/a.py",
            1,
            "executed operation",
        ),
        inference=inference,
    )
    context = verification_context(result)

    assert inference.calls == 5
    assert result.status == "quality_hypothesis"
    assert result.proceed
    assert "fitted-state adaptive rule" in context
    assert "small samples" in context


def test_affected_metric_extracts_only_literal_public_operation_params():
    from fedotllm.agents.evolve.evaluation.affected_eval import (
        extract_operation_overlays,
    )

    code = """
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
window = 77
literal = PipelineBuilder().add_node(
    'lagged', params={'window_size': 100, 'use_cache': False}
)
dynamic = PipelineBuilder().add_node('rf', params={'n_jobs': window})
"""

    assert extract_operation_overlays(code) == {
        "lagged": {"window_size": 100, "use_cache": False}
    }
    assert (
        extract_operation_overlays("PipelineBuilder().add_node(op, params=params)")
        == {}
    )
    assert extract_operation_overlays("not valid python (") == {}


def test_bounded_correctness_verifier_uses_at_most_three_model_calls(tmp_path: Path):
    from fedotllm.agents.evolve.agents.verifier import (
        FinalVerificationProposal,
        VerificationProposal,
        verify_lead,
    )

    source = _source(tmp_path)

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, _prompt, schema):
            self.calls += 1
            if self.calls < 3:
                return VerificationProposal(action="search", query="def value")
            assert schema is FinalVerificationProposal
            return FinalVerificationProposal(
                action="reject", why="no public contract failure was established"
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite(
            "execution",
            "fedot/a.py",
            1,
            "possible return contract mismatch",
            hypothesis_kind="correctness",
        ),
        inference=inference,
        max_model_calls=3,
        correctness_only=True,
    )

    assert inference.calls == 3
    assert result.status == "rejected"


def test_correctness_verifier_rejects_reproduced_unsupported_precondition(
    tmp_path: Path,
):
    from fedotllm.agents.evolve.agents.verifier import (
        ContractSupportAudit,
        VerificationProposal,
        verify_lead,
    )

    source = _source(tmp_path)
    (source / "fedot/a.py").write_text(
        "def value(source_name='default'):\n"
        "    \"\"\"source_name is the label of an existing source.\"\"\"\n"
        "    return source_name == 'default'\n",
        encoding="utf-8",
    )
    reproduction = (
        "from fedot.a import value\n"
        "observed = value(source_name='missing-source')\n"
        "assert observed is True, 'an unknown source must be treated as present'\n"
    )

    class Inference:
        def __init__(self):
            self.calls = 0

        def create(self, prompt, schema):
            self.calls += 1
            if self.calls == 1:
                return VerificationProposal(
                    action="verify_bug",
                    claim="unknown source allegedly requires positive behavior",
                    expected="unknown source is accepted",
                    reproduction_code=reproduction,
                )
            assert schema is ContractSupportAudit
            assert "label of an existing source" in prompt
            assert "missing-source" in prompt
            return ContractSupportAudit(
                verdict="unsupported",
                reason="the reproduction violates the documented source-name precondition",
                evidence=["target docstring requires an existing source label"],
            )

    inference = Inference()
    result = verify_lead(
        source,
        PatchSite(
            "execution",
            "fedot/a.py",
            1,
            "possible source-name contract mismatch",
            hypothesis_kind="correctness",
        ),
        inference=inference,
        max_model_calls=3,
        correctness_only=True,
    )

    assert inference.calls == 2
    assert result.status == "rejected"
    assert "unsupported public-contract precondition" in result.detail


def test_affected_metric_requires_coverage_and_classifies_real_score_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import affected_eval

    spec = TaskSpec(
        task_id="short-ts",
        kind="seq",
        nodes=("lagged", "ridge"),
        problem="ts",
        metric="holdout_rmse",
        higher_is_better=False,
        min_delta=0.01,
        min_delta_mode="relative",
        forecast_horizon=24,
    )
    monkeypatch.setattr(affected_eval, "all_tasks", lambda: [spec])
    monkeypatch.setattr(affected_eval, "load_task", lambda _task_id: spec)
    calls = []

    def stock(*_args, **kwargs):
        calls.append(("stock", kwargs["task_override"]))
        return ScoreResult(
            "short-ts",
            "ok",
            10.0,
            coverage=(
                {
                    "file_path": "fedot/ts.py",
                    "line_ranges": [[10, 16]],
                },
            ),
        )

    patched_score = {"value": 8.0}

    def patched(*_args, **kwargs):
        calls.append(("patched", kwargs["task_override"]))
        return ScoreResult("short-ts", "ok", patched_score["value"])

    monkeypatch.setattr(affected_eval, "run_stock", stock)
    monkeypatch.setattr(affected_eval, "run_patched", patched)
    verification = VerificationResult(
        "verified_bug",
        reproduction_code=(
            "from fedot.core.pipelines.pipeline_builder import PipelineBuilder\n"
            "from fedot.core.repository.tasks import TsForecastingParams\n"
            "PipelineBuilder().add_node('lagged', params={'window_size': 100})\n"
            "assert True\n"
        ),
    )
    lead = PatchSite("llm", "fedot/ts.py", 12)

    improved = affected_eval.evaluate_affected_metric(
        tmp_path / "stock", tmp_path / "patched", verification, lead
    )
    assert improved["status"] == "improved"
    assert improved["rows"][0]["lead_reached"] is True
    assert improved["rows"][0]["normalized_delta"] == pytest.approx(0.2)
    assert calls[0][1] == {
        "operation_params": {"lagged": {"window_size": 100}},
        "history_size": 120,
    }

    patched_score["value"] = 12.0
    regressed = affected_eval.evaluate_affected_metric(
        tmp_path / "stock", tmp_path / "patched", verification, lead
    )
    assert regressed["status"] == "regressed"


def test_affected_metric_does_not_judge_an_unexecuted_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import affected_eval

    spec = TaskSpec(task_id="ts", kind="seq", nodes=("lagged",), problem="ts")
    monkeypatch.setattr(affected_eval, "all_tasks", lambda: [spec])
    monkeypatch.setattr(affected_eval, "load_task", lambda _task_id: spec)
    monkeypatch.setattr(
        affected_eval,
        "run_stock",
        lambda *_a, **_k: ScoreResult(
            "ts",
            "ok",
            1.0,
            coverage=({"file_path": "fedot/ts.py", "line_ranges": [[1, 5]]},),
        ),
    )
    monkeypatch.setattr(
        affected_eval,
        "run_patched",
        lambda *_a, **_k: ScoreResult("ts", "ok", 2.0),
    )
    verification = VerificationResult(
        "verified_bug",
        reproduction_code="PipelineBuilder().add_node('lagged', params={'window_size': 100})",
    )
    result = affected_eval.evaluate_affected_metric(
        tmp_path / "stock",
        tmp_path / "patched",
        verification,
        PatchSite("llm", "fedot/ts.py", 12),
    )

    assert result["status"] == "not_reached"


def test_affected_metric_tries_next_reproduction_operation_until_lead_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import affected_eval

    specs = [
        TaskSpec(task_id="table", kind="seq", nodes=("rf",)),
        TaskSpec(
            task_id="ts",
            kind="seq",
            nodes=("lagged",),
            problem="ts",
            metric="holdout_rmse",
            higher_is_better=False,
            min_delta_mode="relative",
            forecast_horizon=12,
        ),
    ]
    monkeypatch.setattr(affected_eval, "all_tasks", lambda: specs)
    monkeypatch.setattr(
        affected_eval,
        "load_task",
        lambda task_id: next(spec for spec in specs if spec.task_id == task_id),
    )

    def stock(task_id, **_kwargs):
        reached = task_id == "ts"
        return ScoreResult(
            task_id,
            "ok",
            10.0,
            coverage=(
                {
                    "file_path": "fedot/ts.py",
                    "line_ranges": [[10, 14] if reached else [1, 2]],
                },
            ),
        )

    patched_calls = []
    monkeypatch.setattr(affected_eval, "run_stock", stock)
    monkeypatch.setattr(
        affected_eval,
        "run_patched",
        lambda task_id, **_kwargs: patched_calls.append(task_id)
        or ScoreResult(task_id, "ok", 8.0),
    )
    verification = VerificationResult(
        "verified_bug",
        reproduction_code=(
            "PipelineBuilder().add_node('rf', params={'n_jobs': 1})\n"
            "PipelineBuilder().add_node('lagged', params={'window_size': 100})\n"
            "TsForecastingParams(forecast_length=5)\n"
        ),
    )
    result = affected_eval.evaluate_affected_metric(
        tmp_path / "stock",
        tmp_path / "patched",
        verification,
        PatchSite("llm", "fedot/ts.py", 12),
        max_tasks=1,
    )

    assert result["operation"] == "lagged"
    assert result["operations_considered"] == ["rf", "lagged"]
    assert result["status"] == "improved"
    assert patched_calls == ["ts"]


def test_affected_metric_confirmation_requires_two_seeds_and_zero_regressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import json

    from fedotllm.agents.evolve.evaluation import affected_eval

    statuses = {42: "improved", 43: "improved", 44: "neutral"}

    def one_run(*_args, seed=42, split="dev", **_kwargs):
        status = statuses[seed]
        return {
            "status": status,
            "seed": seed,
            "split": split,
            "rows": [
                {
                    "lead_reached": True,
                    "classification": status,
                }
            ],
        }

    monkeypatch.setattr(affected_eval, "evaluate_affected_metric", one_run)
    args = (
        tmp_path / "source",
        tmp_path / "experiment",
        VerificationResult("verified_bug", reproduction_code="assert True"),
        PatchSite("llm", "fedot/ts.py", 12),
    )
    initial = one_run(*args, seed=42)
    confirmed = affected_eval.confirm_affected_metric(*args, initial=initial)

    assert confirmed["confirmed"] is True
    assert confirmed["improved_seeds"] == 2
    assert confirmed["regressed_task_seed_pairs"] == 0
    assert confirmed["runs"][0] is not initial
    initial["dev_confirmation"] = confirmed
    json.dumps(initial)

    statuses[44] = "regressed"
    rejected = affected_eval.confirm_affected_metric(*args, initial=initial)
    assert rejected["confirmed"] is False
    assert rejected["regressed_task_seed_pairs"] == 1


def test_task_override_changes_evaluator_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from fedotllm.agents.evolve.evaluation import eval as evaluator

    monkeypatch.setattr(evaluator, "source_fingerprint", lambda _path: "source")
    monkeypatch.setattr(evaluator, "source_commit", lambda _path: "commit")
    monkeypatch.setattr(evaluator, "score_protocol_fingerprint", lambda: "protocol")
    monkeypatch.setattr(
        evaluator,
        "load_task",
        lambda _task_id: TaskSpec(task_id="ts", kind="seq", nodes=("lagged",)),
    )

    left = evaluator._env_hash(
        tmp_path,
        "ts",
        task_override={"operation_params": {"lagged": {"window_size": 25}}},
    )
    right = evaluator._env_hash(
        tmp_path,
        "ts",
        task_override={"operation_params": {"lagged": {"window_size": 50}}},
    )

    assert left != right
