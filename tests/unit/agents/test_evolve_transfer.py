from dataclasses import replace

import pytest

from fedotllm.agents.evolve.controller import transfer
from fedotllm.agents.evolve.controller.measurement_budget import MeasurementBudget
from fedotllm.agents.evolve.evaluation import tasks
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL
from fedotllm.agents.evolve.types import MatchSite, ScoreResult, TaskSpec


def test_transfer_uses_six_sources_not_temperature_variants():
    ids = transfer.dataset_tasks("ar-shifted-index->ridge")
    assert {tasks.load_task(task).dataset for task in ids} == {
        "temperature", "traffic", "economic", "arctic", "nemo", "metocean"}
    assert len(ids) == 6
    spec = tasks.workload_on_dataset("economic", "ar-shifted-index->ridge")
    assert spec.nodes == ("ar", "ridge")
    assert spec.index_offset == 1000
    assert spec.forecast_horizon == 12
    assert spec.train_file == "time_series/economic_data.csv"
    with pytest.raises(ValueError, match="incompatible"):
        tasks.workload_on_dataset("cancer", "ar->ridge")


@pytest.fixture
def world(monkeypatch):
    specs = {str(i): TaskSpec(str(i), "seq", ("logit",), dataset=str(i),
                             train_file=f"{i}.csv", min_delta=.01) for i in range(4)}
    monkeypatch.setattr(transfer, "all_tasks", lambda: list(specs.values()))
    monkeypatch.setattr(transfer, "load_task", specs.__getitem__)
    from fedotllm.agents.evolve.controller import metric_study
    monkeypatch.setattr(metric_study, "load_task", specs.__getitem__)
    lead = MatchSite("execution", "fedot/a.py", 7)
    effects = {name: .03 for name in specs}
    calls = []
    def score(task, **kwargs):
        calls.append((task, kwargs))
        return ScoreResult(task, "ok", .7, coverage=(
            {"file_path": "fedot/a.py", "line_ranges": [[7, 7]]},), data_evidence={
            "protocol": PROTOCOL, "data_hash": task,
            "validation_ids": [f"{kwargs['split']}:1"],
        })
    def patched(task, **kwargs):
        result = score(task, **kwargs)
        result.score += effects[task]
        return result
    monkeypatch.setattr(transfer, "run_stock", score)
    monkeypatch.setattr(transfer, "run_patched", patched)
    plan = transfer.preregister_transfer(None, "0", lead, {})
    return plan, effects, calls


def test_single_dataset_win_cannot_pass_even_with_large_gain(world):
    plan, effects, calls = world
    effects.update({"0": .3, "1": 0, "2": 0, "3": 0})
    for seed in (42, 43, 44):
        passed, report = transfer.evaluate_transfer(None, None, plan, split="dev", seed=seed)
        assert not passed
        assert report["positive_dataset_count"] == 1
        assert report["required_positive_datasets"] == 3
        assert len(report["datasets"]) == 4


def test_majority_gain_with_neutral_source_passes(world):
    plan, effects, _ = world
    effects["3"] = 0
    passed, report = transfer.evaluate_transfer(None, None, plan, split="dev")
    assert passed and report["positive_dataset_count"] == 3


def test_screen_is_smaller_and_never_claims_full_confirmation(world):
    plan, _, calls = world
    calls.clear()
    passed, report = transfer.evaluate_transfer(None, None, plan, split="dev", scope="screen")
    assert passed and report["reason"] == "transfer_screen_promising"
    assert len(report["datasets"]) == 3
    assert len(calls) == 6  # baseline + patched once for each screen source


def test_screen_sources_are_static_metadata_quantiles_not_hash_order(world):
    plan, _, _ = world
    assert plan["screen_selection"] == "static_task_metadata_quantiles"
    assert len(plan["screen_data_hashes"]) == 3
    # The fixture has equal task metadata, so task id provides the stable tie
    # break. Changing future patch scores cannot change this plan.
    assert plan["screen_data_hashes"] == ["0", "2", "3"]


def test_budget_stops_screen_as_inconclusive_without_a_drop(world):
    plan, _, _ = world
    budget = MeasurementBudget(max_pairs=2, reserve_final_pairs=1)
    passed, report = transfer.evaluate_transfer(
        None, None, plan, split="dev", scope="screen", budget=budget,
    )
    assert not passed and report["reason"] == "inconclusive_budget"
    assert report["measurement_budget"]["pairs_started"] == 1


def test_final_reserve_is_available_only_to_final_stage():
    budget = MeasurementBudget(max_pairs=2, reserve_final_pairs=1)
    first = budget.begin("screen")
    assert first is not None
    budget.finish(first)
    assert budget.begin("screen") is None
    final = budget.begin("final")
    assert final is not None
    budget.finish(final)


def test_regression_cannot_be_hidden_by_other_wins(world):
    plan, effects, _ = world
    effects["3"] = -.01
    passed, report = transfer.evaluate_transfer(None, None, plan, split="dev")
    assert not passed and report["reason"] == "transfer_invalid_or_regressed"
    assert len(report["datasets"]) == 4


def test_copies_of_one_source_cannot_satisfy_transfer(world):
    plan, _, _ = world
    for row in plan["datasets"]:
        row["data_hash"] = "same-content"
    assert transfer.validate_transfer(plan) == "transfer_insufficient_affected_datasets"


def test_unreached_sources_are_guards_not_evidence(world):
    plan, _, _ = world
    for row in plan["datasets"][1:]:
        row["affected"] = False
    assert transfer.validate_transfer(plan) == "transfer_insufficient_affected_datasets"


def test_crashed_baseline_is_incomplete_not_excluded(world):
    plan, _, _ = world
    plan["datasets"][3]["baseline_status"] = "crash"
    assert transfer.validate_transfer(plan) == "transfer_baseline_incomplete"


def test_changed_dataset_inventory_invalidates_plan(world):
    plan, _, _ = world
    plan["datasets"].pop()
    assert transfer.validate_transfer(plan) == "transfer_dataset_inventory_changed"


def test_changed_data_or_missing_branch_rejects(world, monkeypatch):
    plan, _, _ = world
    old = transfer.run_stock
    def changed(task, **kwargs):
        row = old(task, **kwargs)
        if task == "2":
            row.data_evidence["data_hash"] = "changed"
            row.coverage = ()
        return row
    monkeypatch.setattr(transfer, "run_stock", changed)
    passed, report = transfer.evaluate_transfer(None, None, plan, split="shadow")
    assert not passed
    assert "data_changed:2" in report["failures"]
    assert "branch_not_reached:2" in report["failures"]


def test_final_requires_uncertainty_on_multiple_sources(world, monkeypatch):
    from fedotllm.agents.evolve.evaluation import uncertainty
    plan, _, _ = world
    calls = []
    def interval(before, after, **kwargs):
        calls.append(kwargs)
        return {"excludes_zero": before.task_id == "0"}
    monkeypatch.setattr(uncertainty, "paired_interval", interval)
    passed, report = transfer.evaluate_transfer(None, None, plan, split="final")
    assert not passed and report["positive_dataset_count"] == 1
    assert all(call["comparisons"] == 12 for call in calls)


def test_default_operation_is_guarded_on_every_dataset(world, monkeypatch):
    plan, _, _ = world
    plan["operation_params"] = {"logit": {"C": 2}}
    old = transfer.run_patched
    def regress_default(task, **kwargs):
        result = old(task, **kwargs)
        if task == "3" and not kwargs["task_override"]["operation_params"]:
            result.score = .6
        return result
    monkeypatch.setattr(transfer, "run_patched", regress_default)
    passed, report = transfer.evaluate_transfer(None, None, plan, split="dev", params={"logit": {"C": 2}})
    assert not passed
    assert "default:3:regression:3" in report["failures"]


def test_legacy_plan_fails_closed():
    assert transfer.validate_transfer(None) == "transfer_plan_missing_or_stale"


def test_stage_neutral_template_does_not_veto_broad_gain(monkeypatch):
    from fedotllm.agents.evolve.controller import metric_study
    row = ScoreResult('catboost', 'ok', .7, data_evidence={'protocol': PROTOCOL})
    monkeypatch.setattr(metric_study, '_pair', lambda *a, **k: ({'catboost': row}, {'catboost': row}))
    monkeypatch.setattr(transfer, 'evaluate_transfer', lambda *a, **k: (True, {'reason': 'transfer_confirmed'}))
    passed, report, _, _ = metric_study._stage(None, None, ('catboost',), 'catboost', split='dev', transfer={})
    assert passed and report['delta'] == 0


def test_stage_target_win_cannot_bypass_failed_transfer(monkeypatch):
    from fedotllm.agents.evolve.controller import metric_study
    row = ScoreResult('catboost', 'ok', .7, data_evidence={'protocol': PROTOCOL})
    monkeypatch.setattr(metric_study, '_pair', lambda *a, **k: ({'catboost': row}, {'catboost': replace(row, score=.9)}))
    monkeypatch.setattr(transfer, 'evaluate_transfer', lambda *a, **k: (False, {'reason': 'transfer_gain_not_reproduced'}))
    passed, report, _, _ = metric_study._stage(None, None, ('catboost',), 'catboost', split='dev', transfer={})
    assert not passed and report['reason'] == 'transfer_gain_not_reproduced'


def test_ordinary_final_without_transfer_never_evaluates_or_accepts(tmp_path):
    from fedotllm.agents.evolve.controller.finalization import record_final
    from fedotllm.agents.evolve.types import Decision, PatchCandidate
    def forbidden(*args, **kwargs):
        pytest.fail('FINAL opened without cross-dataset plan')
    decision = record_final(tmp_path, tmp_path, tmp_path / 'journal.jsonl', PatchCandidate('x'),
                            Decision(True, 'keep', .1), run_id='x', measure_stock_fn=forbidden,
                            measure_patched_fn=forbidden, verdict_fn=forbidden)
    assert not decision.keep and decision.reason == 'transfer_plan_missing_or_stale'


def test_changed_parameters_do_not_reuse_coverage_plan(world):
    plan, _, _ = world
    passed, report = transfer.evaluate_transfer(None, None, plan, split='dev', params={'logit': {'C': 999}})
    assert not passed and report['reason'] == 'transfer_parameters_changed'


def test_neutral_default_and_improved_shifted_scenario_count_sources_once(world, monkeypatch):
    original, _, _ = world
    old_tasks, old_load = transfer.all_tasks, transfer.load_task
    variant = replace(old_load('0'), task_id='shifted', index_offset=1000)
    monkeypatch.setattr(transfer, 'all_tasks', lambda: [*old_tasks(), variant])
    monkeypatch.setattr(transfer, 'load_task', lambda task: variant if task == 'shifted' else old_load(task))
    old_patched = transfer.run_patched
    def patched(task, **kwargs):
        result = old_patched(task, **kwargs)
        if kwargs['task_override']['pipeline_task'] != 'shifted' or kwargs['task_override'].get('index_offset') == 0:
            result.score = .7
        return result
    monkeypatch.setattr(transfer, 'run_patched', patched)
    plan = transfer.preregister_transfer(None, '0', MatchSite(**original['lead']), {})
    passed, report = transfer.evaluate_transfer(None, None, plan, split='dev')
    assert passed
    assert len(report['datasets']) == 8
    assert report['affected_dataset_count'] == report['positive_dataset_count'] == 4


def test_wall_clock_reserve_blocks_screen_but_keeps_final_available(monkeypatch):
    import time
    budget = MeasurementBudget(
        max_pairs=None,
        max_seconds=10,
        reserve_final_seconds=3,
    )
    # Simulate a worker whose campaign has already consumed eight seconds.
    monkeypatch.setattr(
        budget, "_campaign_started_at", time.monotonic() - 8,
    )
    assert budget.begin("screen") is None
    final = budget.begin("final")
    assert final is not None
    budget.finish(final)
    snapshot = budget.snapshot()
    assert snapshot["remaining_seconds"] <= 2.1
    assert "_campaign_started_at" not in snapshot


def test_traced_stock_crash_recovered_on_multiple_sources_is_accepted(world, monkeypatch):
    plan, _, _ = world
    lead = MatchSite(**plan["lead"])
    original_stock = transfer.run_stock

    def crash_at_lead(task, **kwargs):
        row = original_stock(task, **kwargs)
        row.status = "crash"
        row.score = .5
        row.traceback = f'File "/tmp/{lead.file_path}", line {lead.line}, in fit'
        return row

    monkeypatch.setattr(transfer, "run_stock", crash_at_lead)
    plan = transfer.preregister_transfer(None, "0", lead, {})
    assert transfer.validate_transfer(plan) is None
    passed, report = transfer.evaluate_transfer(None, None, plan, split="final")
    assert passed
    assert report["reason"] == "transfer_recovery_confirmed"
    assert report["evidence_kind"] == "crash_recovery"
    assert report["positive_dataset_count"] == 4


def test_stock_crash_outside_planned_line_remains_incomplete(world, monkeypatch):
    plan, _, _ = world
    lead = MatchSite(**plan["lead"])
    original_stock = transfer.run_stock

    def crash_elsewhere(task, **kwargs):
        row = original_stock(task, **kwargs)
        row.status = "crash"
        row.traceback = 'File "/tmp/fedot/elsewhere.py", line 9, in fit'
        return row

    monkeypatch.setattr(transfer, "run_stock", crash_elsewhere)
    plan = transfer.preregister_transfer(None, "0", lead, {})
    assert transfer.validate_transfer(plan) == "transfer_baseline_incomplete"


def test_identical_traced_crash_is_neutral_guard_not_regression(world, monkeypatch):
    plan, _, _ = world
    lead = MatchSite(**plan["lead"])
    original_stock = transfer.run_stock
    original_patched = transfer.run_patched

    def crash_at_lead(task, **kwargs):
        row = original_stock(task, **kwargs)
        row.status = "crash"
        row.score = .5
        row.detail = "IndexError: planned failure"
        row.traceback = f'File "/tmp/{lead.file_path}", line {lead.line}, in fit'
        return row

    def recover_except_last(task, **kwargs):
        if task != "3":
            return original_patched(task, **kwargs)
        return crash_at_lead(task, **kwargs)

    monkeypatch.setattr(transfer, "run_stock", crash_at_lead)
    monkeypatch.setattr(transfer, "run_patched", recover_except_last)
    plan = transfer.preregister_transfer(None, "0", lead, {})
    passed, report = transfer.evaluate_transfer(None, None, plan, split="dev")
    assert passed
    assert report["positive_dataset_count"] == 3
    assert not report["failures"]
    assert any(row.get("unchanged_baseline_crash_unmeasured") for row in report["datasets"])


def test_json_default_is_reached_by_matching_runtime_operation(world, monkeypatch, tmp_path):
    plan, _, _ = world
    config = tmp_path / "fedot/defaults.json"
    config.parent.mkdir(parents=True)
    config.write_text('{\n  "logit": {"C": 1}\n}\n')
    lead = MatchSite("configuration", "fedot/defaults.json", 2)
    original_stock = transfer.run_stock

    def operation_trace(task, **kwargs):
        row = original_stock(task, **kwargs)
        row.dataflow = ({"operation": "logit"},)
        return row

    monkeypatch.setattr(transfer, "run_stock", operation_trace)
    plan = transfer.preregister_transfer(tmp_path, "0", lead, {})
    assert all(row["configuration_reached"] for row in plan["datasets"])
    assert transfer.validate_transfer(plan) is None


def test_json_default_reachability_is_rechecked_during_transfer(world, monkeypatch, tmp_path):
    _, _, _ = world
    config = tmp_path / "fedot/defaults.json"
    config.parent.mkdir(parents=True)
    config.write_text('{\n  "logit": {"C": 1}\n}\n')
    lead = MatchSite("configuration", "fedot/defaults.json", 2)
    original_stock = transfer.run_stock

    def operation_trace(task, **kwargs):
        row = original_stock(task, **kwargs)
        row.dataflow = ({"operation": "logit"},)
        return row

    monkeypatch.setattr(transfer, "run_stock", operation_trace)
    plan = transfer.preregister_transfer(tmp_path, "0", lead, {})
    passed, report = transfer.evaluate_transfer(tmp_path, tmp_path, plan, split="dev")
    assert passed
    assert not report["failures"]
