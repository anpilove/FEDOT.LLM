from __future__ import annotations

from fedotllm.agents.evolve.evaluation.quality_registry import (
    list_quality_task_ids,
    load_quality_registry,
    select_task_ids_for_job,
)
from fedotllm.agents.evolve.evaluation.quality_scope import (
    analyze_quality_scope,
    extract_change_tags,
    extract_changed_symbols,
)


def test_multifile_patch_unions_problem_families_before_scoring():
    scope = analyze_quality_scope(
        {
            "changed_files": [
                "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/ar.py",
                "fedot/core/operations/evaluation/operation_implementations/models/regression/ridge.py",
            ]
        }
    )
    assert scope["problem_families"] == ["regression", "ts_forecasting"]
    assert scope["confidence"] == "direct"


def test_shared_code_selects_every_problem_family():
    registry = load_quality_registry()
    scope = analyze_quality_scope(
        {"changed_files": ["fedot/core/data/data_split.py"]}, registry
    )
    assert scope["problem_families"] == [
        "classification",
        "regression",
        "ts_forecasting",
    ]
    assert select_task_ids_for_job(
        {"changed_files": ["fedot/core/data/data_split.py"]}, registry=registry
    ) == list_quality_task_ids(registry)
    assert select_task_ids_for_job(
        {"changed_files": ["fedot/core/data/data_split.py"]},
        requested=(),
        registry=registry,
    ) == ()


def test_guarded_categorical_encoder_helper_routes_to_categorical_datasets():
    registry = load_quality_registry()
    job = {
        "changed_files": ["fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py"],
        "changed_symbols": ["OneHotEncodingImplementation._apply_one_hot_encoding"],
        "changed_code_tags": ["categorical"],
        "hypothesis": {
            "claim": "Preserve categorical features for native categorical estimators",
            "expected_effect": "Improve scores on datasets with categorical columns",
        },
    }

    selected = select_task_ids_for_job(job, registry=registry)
    expected = tuple(
        dataset.task_id for dataset in registry.datasets if "categorical" in dataset.tags
    )
    assert selected == expected
    assert len(selected) == 11
    assert all(not task_id.startswith("fedot-ts-") for task_id in selected)


def test_categorical_scope_stays_broad_if_code_or_hypothesis_is_ambiguous():
    registry = load_quality_registry()
    broad = list_quality_task_ids(registry)
    cases = [
        {
            "changed_files": ["fedot/core/data/data.py"],
            "changed_code_tags": ["categorical"],
            "hypothesis": {"claim": "Improve general model quality"},
        },
        {
            "changed_files": ["fedot/core/data/data.py"],
            "hypothesis": {"claim": "Preserve categorical features"},
        },
    ]
    for job in cases:
        assert select_task_ids_for_job(job, registry=registry) == broad


def test_changed_code_tags_require_every_patch_edit_to_match_trait():
    assert extract_change_tags(["if categorical_idx is not None:", "cat_features = raw"])
    assert extract_change_tags(["categorical_idx = idx", "return values"]) == []


def test_comments_and_string_literals_do_not_establish_categorical_code():
    for code in (
        "return np.nan_to_num(self.features)  # preserve categorical features",
        'message = "categorical features"',
        '"""categorical preprocessing"""\nreturn values',
    ):
        assert extract_change_tags([code]) == []


def test_shared_and_unknown_changes_do_not_remove_non_categorical_datasets():
    registry = load_quality_registry()
    for path in ("fedot/core/data/data.py", "fedot/new_component.py"):
        job = {
            "changed_files": [path],
            "changed_symbols": ["InputData.get_not_encoded_data"],
            "changed_code_tags": ["categorical"],
            "hypothesis": {"claim": "Preserve categorical features"},
        }
        assert select_task_ids_for_job(job, registry=registry) == list_quality_task_ids(registry)


def test_dedicated_encoder_general_methods_stay_broad():
    registry = load_quality_registry()
    for symbols in (
        [],
        ["OneHotEncodingImplementation.transform"],
        ["OneHotEncodingImplementation._apply_one_hot_encoding", "OneHotEncodingImplementation.transform"],
    ):
        job = {
            "changed_files": ["fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py"],
            "changed_symbols": symbols,
            "changed_code_tags": ["categorical"],
            "hypothesis": {"claim": "Preserve categorical features"},
        }
        assert select_task_ids_for_job(job, registry=registry) == list_quality_task_ids(registry)


def test_ast_error_prevents_categorical_narrowing():
    registry = load_quality_registry()
    job = {
        "changed_files": ["fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py"],
        "changed_symbols": ["OneHotEncodingImplementation._apply_one_hot_encoding"],
        "changed_code_tags": ["categorical"],
        "ast_errors": ["edit_not_found"],
        "hypothesis": {"claim": "Preserve categorical features"},
    }
    assert select_task_ids_for_job(job, registry=registry) == list_quality_task_ids(registry)


def test_regression_strategy_checks_regression_and_forecasting_datasets():
    registry = load_quality_registry()
    job = {
        "changed_files": ["fedot/core/operations/evaluation/regression.py"],
        "changed_symbols": ["SkLearnRegressionStrategy.predict"],
    }
    selected = select_task_ids_for_job(job, registry=registry)
    assert len(selected) == 16
    assert "openml-regression-airfoil-self-noise" in selected
    assert "fedot-ts-temperature" in selected
    assert "openml-31-credit-g" not in selected


def test_ast_scope_covers_entire_edit_not_only_first_line(tmp_path):
    source = tmp_path / "encoder.py"
    text = (
        "class Encoder:\n"
        "    def categorical_helper(self):\n"
        "        return self.categorical_idx\n"
        "    def transform(self):\n"
        "        return self.features\n"
    )
    source.write_text(text, encoding="utf-8")
    symbols, errors = extract_changed_symbols(
        tmp_path,
        [{"file_path": "encoder.py", "old_code": "        return self.categorical_idx\n    def transform(self):\n"}],
    )
    assert symbols == ["Encoder"]
    assert errors == []


def test_module_level_edit_is_visible_as_unresolved_symbol(tmp_path):
    source = tmp_path / "encoder.py"
    source.write_text("categorical_default = None\n", encoding="utf-8")
    symbols, errors = extract_changed_symbols(
        tmp_path, [{"file_path": "encoder.py", "old_code": "categorical_default = None\n"}],
    )
    assert symbols == []
    assert errors == ["encoder.py:edit_outside_symbol"]


def test_new_code_escaping_categorical_helper_prevents_narrowing(tmp_path):
    source = tmp_path / "encoder.py"
    source.write_text(
        "class OneHotEncodingImplementation:\n"
        "    def _apply_one_hot_encoding(self, features):\n"
        "        return features[:, self.categorical_ids]\n",
        encoding="utf-8",
    )
    symbols, errors = extract_changed_symbols(
        tmp_path,
        [{
            "file_path": "encoder.py",
            "old_code": "        return features[:, self.categorical_ids]\n",
            "new_code": "        return features[:, self.categorical_ids]\ncategorical_default = True\n",
        }],
    )
    assert errors == ["encoder.py:new_edit_outside_symbol"]


def test_queue_preserves_guarded_encoder_helper_trait(tmp_path):
    import json

    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit

    rel = "fedot/core/operations/evaluation/operation_implementations/data_operations/categorical_encoders.py"
    source = tmp_path / "FEDOT"
    target = source / rel
    target.parent.mkdir(parents=True)
    target.write_text(
        "class OneHotEncodingImplementation:\n"
        "    def _apply_one_hot_encoding(self, features):\n"
        "        return features[:, self.categorical_ids]\n",
        encoding="utf-8",
    )
    candidate = PatchCandidate(
        "guarded-encoder", file_path=rel,
        edits=[PatchEdit(rel, "        return features[:, self.categorical_ids]\n",
                         "        return features[:, self.categorical_ids].astype(str)\n")],
    )
    path = enqueue_quality_job(
        tmp_path, candidate=candidate, patch_text="encoder patch",
        hint="", priority="normal", source_root=source,
        hypothesis={"claim": "Preserve categorical features"},
    )
    job = json.loads(path.read_text(encoding="utf-8"))
    assert job["quality_scope"]["dataset_traits"] == ["categorical"]
    assert len(job["task_ids"]) == 11
    assert "openml-regression-abalone" in job["task_ids"]
    assert "fedot-ts-temperature" not in job["task_ids"]


def test_unknown_changed_file_falls_back_to_full_suite_even_with_narrow_hypothesis():
    registry = load_quality_registry()
    job = {
        "changed_files": ["fedot/new_unmapped_component.py"],
        "hypothesis": {
            "claim": "Improve forecasting accuracy",
            "expected_effect": "TS only",
        },
    }
    scope = analyze_quality_scope(job, registry)
    assert scope["problem_families"] == [
        "classification",
        "regression",
        "ts_forecasting",
    ]
    assert scope["confidence"] == "fallback"
    assert select_task_ids_for_job(job, registry=registry) == list_quality_task_ids(
        registry
    )


def test_hypothesis_does_not_widen_task_specific_changed_code_scope():
    registry = load_quality_registry()
    scope = analyze_quality_scope(
        {
            "changed_files": [
                "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
            ],
            "hypothesis": {
                "claim": "The downstream ridge regression node learns from AutoReg predictions",
                "expected_effect": "Improve forecast error on time-series workloads",
            },
        },
        registry,
    )
    assert scope["problem_families"] == ["ts_forecasting"]
    assert select_task_ids_for_job(
        {
            "changed_files": [
                "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
            ],
            "hypothesis": {
                "claim": "The downstream ridge regression node learns from AutoReg predictions",
                "expected_effect": "Improve forecast error on time-series workloads",
            },
        },
        registry=registry,
    ) == tuple(
        dataset.task_id for dataset in registry.datasets if dataset.problem == "ts_forecasting"
    )


def test_autoregression_hypothesis_does_not_select_tabular_regression_tasks():
    registry = load_quality_registry()
    job = {
        "changed_files": [
            "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/statsmodels.py"
        ],
        "hypothesis": {
            "claim": "AutoReg autoregression uses two lagged coefficients and loses short-lag information",
            "expected_effect": "Improve forecast error on time-series forecasting workloads",
        },
    }

    scope = analyze_quality_scope(job, registry)

    assert scope["problem_families"] == ["ts_forecasting"]
    assert select_task_ids_for_job(job, registry=registry) == tuple(
        dataset.task_id for dataset in registry.datasets if dataset.problem == "ts_forecasting"
    )


def test_regress_prefixed_symbol_still_selects_regression_tasks():
    scope = analyze_quality_scope({"changed_files": ["fedot/regress_model.py"]})

    assert scope["problem_families"] == ["regression"]


def test_model_implementation_paths_route_to_their_supported_task_families():
    registry = load_quality_registry()
    cases = {
        "models/discriminant_analysis.py": ["classification"],
        "models/svc.py": ["classification"],
        "models/keras.py": ["classification"],
        "models/knn.py": ["classification", "regression"],
        "models/tabpfn.py": ["classification", "regression"],
        "models/custom_model.py": [
            "classification",
            "regression",
            "ts_forecasting",
        ],
    }

    for suffix, expected_families in cases.items():
        job = {
            "changed_files": [
                "fedot/core/operations/evaluation/operation_implementations/"
                + suffix
            ]
        }
        scope = analyze_quality_scope(job, registry)
        assert scope["problem_families"] == expected_families
        selected = select_task_ids_for_job(job, registry=registry)
        expected_ids = tuple(
            dataset.task_id
            for dataset in registry.datasets
            if dataset.problem in expected_families
        )
        assert selected == expected_ids


def test_ts_prefixed_utility_routes_to_forecasting_only():
    registry = load_quality_registry()
    job = {"changed_files": ["fedot/utilities/ts_gapfilling.py"]}

    assert analyze_quality_scope(job, registry)["problem_families"] == [
        "ts_forecasting"
    ]
    assert select_task_ids_for_job(job, registry=registry) == tuple(
        dataset.task_id
        for dataset in registry.datasets
        if dataset.problem == "ts_forecasting"
    )


def test_task_specific_evaluation_strategies_route_to_supported_families():
    registry = load_quality_registry()
    expected = tuple(
        dataset.task_id
        for dataset in registry.datasets
        if dataset.problem in {"classification", "regression"}
    )
    for path in (
        "fedot/core/operations/evaluation/boostings.py",
        "fedot/core/operations/evaluation/tabpfn.py",
    ):
        job = {"changed_files": [path]}
        assert analyze_quality_scope(job, registry)["problem_families"] == [
            "classification",
            "regression",
        ]
        assert select_task_ids_for_job(job, registry=registry) == expected


def test_ast_finds_enclosing_operation_symbol_for_exact_patch_edit(tmp_path):
    source = tmp_path / "fedot" / "forecasting.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class TimeSeriesForecast:\n"
        "    def predict(self, values):\n"
        "        return values\n",
        encoding="utf-8",
    )
    symbols, errors = extract_changed_symbols(
        tmp_path,
        [{"file_path": "fedot/forecasting.py", "old_code": "        return values\n"}],
    )
    assert symbols == ["TimeSeriesForecast.predict"]
    assert errors == []


def test_ast_parse_failure_is_visible_for_conservative_fallback(tmp_path):
    source = tmp_path / "fedot" / "bad.py"
    source.parent.mkdir(parents=True)
    source.write_text("def broken(:\n", encoding="utf-8")
    symbols, errors = extract_changed_symbols(
        tmp_path,
        [{"file_path": "fedot/bad.py", "old_code": "def broken(:\n"}],
    )
    assert symbols == []
    assert errors == ["fedot/bad.py:SyntaxError"]


def test_queue_freezes_scope_hypothesis_and_selected_tasks_for_all_edits(tmp_path):
    import json

    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit

    source = tmp_path / "FEDOT"
    ts = (
        source
        / "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/ar.py"
    )
    reg = (
        source
        / "fedot/core/operations/evaluation/operation_implementations/models/regression/ridge.py"
    )
    ts.parent.mkdir(parents=True)
    reg.parent.mkdir(parents=True)
    ts.write_text("def forecast(values):\n    return values\n", encoding="utf-8")
    reg.write_text("def fit(values):\n    return values\n", encoding="utf-8")
    candidate = PatchCandidate(
        candidate_id="multi-family-candidate",
        file_path="fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/ar.py",
        edits=[
            PatchEdit(
                ts.relative_to(source).as_posix(),
                "    return values\n",
                "    return list(values)\n",
            ),
            PatchEdit(
                reg.relative_to(source).as_posix(),
                "    return values\n",
                "    return tuple(values)\n",
            ),
        ],
    )
    job_path = enqueue_quality_job(
        tmp_path,
        candidate=candidate,
        patch_text="multi-file patch",
        hint="",
        priority="normal",
        hypothesis={
            "id": "h-1",
            "claim": "Improve forecasting and regression preprocessing",
            "expected_effect": "Improve RMSE on affected workloads",
        },
        source_root=source,
    )
    job = json.loads(job_path.read_text(encoding="utf-8"))
    registry = load_quality_registry()
    assert job["hypothesis"]["id"] == "h-1"
    assert job["changed_files"] == [edit.file_path for edit in candidate.edits]
    assert job["changed_symbols"] == ["forecast", "fit"]
    assert job["quality_scope"]["problem_families"] == ["regression", "ts_forecasting"]
    assert job["task_ids"] == [
        item.task_id
        for item in registry.datasets
        if item.problem in {"regression", "ts_forecasting"}
    ]


def test_queue_keeps_shared_source_broad_despite_categorical_hypothesis(tmp_path):
    import json

    from fedotllm.agents.evolve.controller.quality_queue import enqueue_quality_job
    from fedotllm.agents.evolve.controller.quality_executor import _job_quality_task_ids
    from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit

    source = tmp_path / "FEDOT"
    target = source / "fedot/core/data/data.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def get_not_encoded_data(self):\n    return self.features\n", encoding="utf-8"
    )
    candidate = PatchCandidate(
        candidate_id="categorical-trait-candidate",
        file_path="fedot/core/data/data.py",
        edits=[
            PatchEdit(
                "fedot/core/data/data.py",
                "    return self.features\n",
                "    if categorical_idx is not None:\n        return cat_features\n",
            )
        ],
    )
    path = enqueue_quality_job(
        tmp_path,
        candidate=candidate,
        patch_text="test patch",
        hint="",
        priority="normal",
        hypothesis={"claim": "Preserve categorical features for native estimators"},
        source_root=source,
    )
    job = json.loads(path.read_text(encoding="utf-8"))
    registry = load_quality_registry()
    expected = list(list_quality_task_ids(registry))
    assert job["changed_code_tags"] == ["categorical"]
    assert job["quality_scope"]["dataset_traits"] == []
    assert job["task_ids"] == expected
    assert _job_quality_task_ids(job, list_quality_task_ids(registry), registry) == tuple(expected)


def test_quality_result_message_separates_no_gain_from_infrastructure_failure():
    from fedotllm.agents.evolve.controller.quality_queue import quality_result_message

    accepted = quality_result_message(
        {
            "candidate_id": "candidate-0",
            "status": "keep",
            "reason": "quality_improved",
            "quality_outcome": "broad_positive",
            "improved_task_count": 2,
            "task_ids": ["cls-a", "reg-a", "ts-a"],
            "hypothesis": {"claim": "improve validation score"},
        }
    )
    no_gain = quality_result_message(
        {
            "candidate_id": "candidate-1",
            "status": "drop",
            "reason": "no_quality_gain",
            "quality_outcome": "insufficient_effect",
            "improved_task_count": 1,
            "task_ids": ["fedot-ts-temperature", "fedot-ts-traffic"],
            "hypothesis": {"claim": "stabilize forecast window"},
        }
    )
    infrastructure = quality_result_message(
        {
            "candidate_id": "candidate-2",
            "status": "infrastructure_error",
            "reason": "quality fedot-ts-temperature timeout",
            "task_ids": ["fedot-ts-temperature"],
            "hypothesis": {"claim": "stabilize forecast window"},
        }
    )
    candidate_error = quality_result_message(
        {
            "candidate_id": "candidate-3",
            "status": "candidate_error",
            "reason": "patched_quality_failed:fedot-ts-temperature:invalid:search_ran=False",
            "task_ids": ["fedot-ts-temperature"],
        }
    )
    evaluation_error = quality_result_message(
        {
            "candidate_id": "candidate-4",
            "status": "evaluation_error",
            "reason": "stock_evaluation_failed:fedot-ts-temperature:invalid:search_ran=False",
            "task_ids": ["fedot-ts-temperature"],
        }
    )

    assert "2 из 3" in accepted
    assert "большинстве задач" in accepted
    assert "Не принимаем" in no_gain
    assert "прирост" in no_gain
    assert "1 из 2 задач" in no_gain
    assert "критерий локального" in no_gain
    assert "не завершена" in infrastructure
    assert "не считать" in infrastructure
    assert "автоматически повторять" in candidate_error
    assert "не подтверждённая регрессия" in evaluation_error


def test_skipped_frozen_job_falls_back_to_its_required_datasets():
    from fedotllm.agents.evolve.controller.quality_executor import _job_quality_task_ids

    registry = load_quality_registry()
    pool = tuple(
        item.task_id for item in registry.datasets if item.problem == "classification"
    )
    required = tuple(
        item.task_id for item in registry.datasets if item.problem == "ts_forecasting"
    )
    job = {
        "quality_scope_frozen": True,
        "status": "skipped",
        "task_ids": [],
        "required_task_ids": list(required),
        "quality_scope": {"problem_families": ["ts_forecasting"]},
    }

    assert _job_quality_task_ids(job, pool, registry) == required


def test_skipped_frozen_job_without_recorded_scope_falls_back_to_full_registry():
    from fedotllm.agents.evolve.controller.quality_executor import _job_quality_task_ids

    registry = load_quality_registry()
    job = {"quality_scope_frozen": True, "status": "skipped", "task_ids": []}

    assert _job_quality_task_ids(job, (), registry) == tuple(
        item.task_id for item in registry.datasets
    )
