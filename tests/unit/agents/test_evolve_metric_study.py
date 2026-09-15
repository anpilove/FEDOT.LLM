from __future__ import annotations

import json

import numpy as np
import pytest

from fedotllm.agents.evolve.controller import metric_study as study
from fedotllm.agents.evolve.evaluation.independent_data import (
    PROTOCOL, partition_indices, select_indices, table_split, ts_indices,
)
from fedotllm.agents.evolve.evaluation.uncertainty import paired_interval
from fedotllm.agents.evolve.types import PatchSite, ScoreResult, TaskSpec
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit, VerificationResult


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("problem", ["classification", "regression", "text"])
def test_training_folds_never_touch_outer_holdouts(external, problem):
    labels = np.tile([0, 1], 150)
    other = np.tile([0, 1], 50) if external else None
    core, holdouts = partition_indices(labels, problem=problem, external_target=other)
    outer = [{f"{ns}:{i}" for i in ids} for ns, ids in holdouts.values()]
    assert all(not a & b for i, a in enumerate(outer) for b in outer[i + 1:])
    outer_ids = set().union(*outer)
    cv_seen = set()
    for fold in (0, 1, 2):
        train, ns, validation, repeated_core = select_indices(
            labels, problem=problem, split="dev", fold=fold, external_target=other,
        )
        ids = {f"{ns}:{i}" for i in validation}
        assert not ids & (outer_ids | cv_seen)
        assert not {f"train:{i}" for i in train} & (outer_ids | ids)
        assert np.array_equal(core, repeated_core)
        cv_seen.update(ids)
    assert cv_seen == {f"train:{i}" for i in core}


@pytest.mark.parametrize("split,fold", [("final", 0), ("shadow", 1), ("dev", 3)])
def test_resampling_cannot_move_outer_holdouts(split, fold):
    with pytest.raises(ValueError):
        select_indices(np.tile([0, 1], 50), problem="classification", split=split, fold=fold)


def test_pair_evidence_identifies_exact_rows_and_dataset():
    x = np.arange(600).reshape(300, 2)
    y = np.tile([0, 1], 150)
    one = table_split(x, y, problem="classification", split="dev")[-1]
    two = table_split(x.copy(), y.copy(), problem="classification", split="dev")[-1]
    assert one == two
    changed = table_split(x + 1, y, problem="classification", split="dev")[-1]
    assert one["data_hash"] != changed["data_hash"]
    cv = table_split(x, y, problem="classification", split="dev", fold=0)[-1]
    assert cv["validation_hash"] != one["validation_hash"]


def test_rolling_origins_precede_outer_windows():
    prior = set()
    last_end = -1
    for split, fold in [("dev", 2), ("dev", 1), ("dev", 0), ("dev", None), ("shadow", None), ("final", None)]:
        train, valid = ts_indices(300, 24, split=split, fold=fold)
        assert min(valid) > last_end
        last_end = max(valid)
        assert max(train) < min(valid)
        assert not set(valid) & prior
        prior.update(valid)
    with pytest.raises(ValueError, match="insufficient history"):
        ts_indices(48, 24, split="dev", fold=2)


@pytest.fixture
def specs(monkeypatch):
    monkeypatch.setattr(study, "load_task", lambda name: TaskSpec(
        name, "seq", ("logit",), metric="holdout_roc_auc", min_delta=0.01,
    ))


def score(task, value, **kwargs):
    return ScoreResult(task, "ok", value, data_evidence={
        "protocol": PROTOCOL, "validation_ids": ["train:1"],
    }, **kwargs)


def test_guard_improvement_cannot_replace_preregistered_target(specs):
    before = {"target": score("target", .7), "guard": score("guard", .6)}
    after = {"target": score("target", .7), "guard": score("guard", .9)}
    passed, result = study.judge_pair(before, after, "target")
    assert not passed
    assert result["reason"] == "target_no_gain"
    assert result["delta"] == 0


def test_subthreshold_guard_move_is_retained_for_later_confirmation(specs):
    before = {"target": score("target", .7), "guard": score("guard", .6)}
    after = {"target": score("target", .73), "guard": score("guard", .599)}
    passed, result = study.judge_pair(before, after, "target")
    assert passed and result["reason"] == "passed"


def test_material_guard_regression_is_rejected(specs):
    before = {"target": score("target", .7), "guard": score("guard", .6)}
    after = {"target": score("target", .73), "guard": score("guard", .589)}
    passed, result = study.judge_pair(before, after, "target")
    assert not passed and result["reason"] == "regression:guard"


def test_data_mismatch_fails_closed(specs):
    before, after = score("target", .7), score("target", .9)
    after.data_evidence["validation_ids"] = ["train:2"]
    passed, result = study.judge_pair({"target": before}, {"target": after}, "target")
    assert not passed and result["reason"] == "data_pair_mismatch:target"


def test_current_protocol_reaches_real_scorer_dispatch(monkeypatch):
    from fedotllm.agents.evolve.evaluation import scorer
    from fedotllm.agents.evolve.controller import campaign
    assert scorer.INDEPENDENT_DATA_PROTOCOL == campaign.INDEPENDENT_DATA_PROTOCOL == PROTOCOL
    monkeypatch.setattr(scorer, "load_task", lambda task: TaskSpec(task, "seq", ("logit",)))
    def strict(*args, **kwargs):
        raise RuntimeError("strict split reached")
    monkeypatch.setattr(scorer, "_make_independent_split", strict)
    result = scorer.score_task("dispatch", task_override={"evaluation_protocol": PROTOCOL})
    assert result["detail"] == "RuntimeError: strict split reached"


def test_parameterized_target_retains_default_workload_guard(monkeypatch, specs):
    def stock(task, **kwargs):
        return score(task, .7)
    def patched(task, **kwargs):
        value = .8 if kwargs["task_override"].get("operation_params") else .69
        return score(task, value)
    monkeypatch.setattr(study, "run_stock", stock)
    monkeypatch.setattr(study, "run_patched", patched)
    before, after = study._pair(None, None, ("target",), split="dev", target="target",
                                operation_params={"logit": {"C": 2}})
    assert set(before) == {"target", "target::default"}
    passed, result = study.judge_pair(before, after, "target")
    assert not passed and result["reason"] == "regression:target::default"


def test_preregistered_target_cannot_change_on_resume(tmp_path, specs):
    lead = PatchSite("execution", "fedot/a.py", 3)
    covered = ({"file_path": "fedot/a.py", "line_ranges": [[2, 4]]},)
    baseline = {"b": score("b", .1, coverage=covered), "a": score("a", .9, coverage=covered)}
    path = tmp_path / "plan.json"
    plan = study.preregister(lead, baseline, ("b", "a"), path)
    assert plan["target_task"] == "a"  # deterministic coverage, not best prospective delta
    assert study.preregister(lead, baseline, ("a", "b"), path) == plan
    with pytest.raises(ValueError, match="cannot be changed"):
        study.preregister(lead, baseline, ("b",), path)


def test_bootstrap_no_effect_is_not_significant():
    y = np.tile([0., 1.], 30)
    a, b = score("x", .5), score("x", .5)
    a.metric_observations = b.metric_observations = {"target": y.tolist(), "prediction": [0.5] * len(y)}
    result = paired_interval(a, b, metric="holdout_roc_auc", repeats=200)
    assert result["status"] == "estimated"
    assert result["lower"] == result["upper"] == 0
    assert not result["excludes_zero"]


def test_bootstrap_pairs_predictions_and_adjusts_for_three_candidates():
    y = np.tile([0., 1.], 30)
    a, b = score("x", .5), score("x", 1)
    a.metric_observations = {"target": y.tolist(), "prediction": [0.5] * len(y)}
    b.metric_observations = {"target": y.tolist(), "prediction": y.tolist()}
    result = paired_interval(a, b, metric="holdout_roc_auc", repeats=200)
    assert result["excludes_zero"] and result["comparisons"] == 3
    b.metric_observations["target"] = (1 - y).tolist()
    assert paired_interval(a, b, metric="holdout_roc_auc")["status"] == "insufficient_or_invalid_paired_observations"


def test_final_requires_a_batch_before_any_score_call(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_stage", lambda *a, **k: pytest.fail("premature FINAL"))
    with pytest.raises(ValueError, match="three frozen"):
        study.finalize_batch(tmp_path / "source", tmp_path / "study")


def test_cli_resumes_existing_final_without_model(tmp_path, monkeypatch, capsys):
    from fedotllm.agents.evolve.__main__ import main
    (tmp_path / "final-batch.json").write_text("{}")
    monkeypatch.setattr(study, "finalize_batch", lambda source, root: {"metric_confirmed_count": 1})
    assert main(["metric-finalize", "--metric-study", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["metric_confirmed_count"] == 1


def test_candidate_stages_keep_target_and_never_query_final_before_batch(tmp_path, monkeypatch, specs):
    from fedotllm.agents.evolve.controller import transfer
    monkeypatch.setattr(transfer, "validate_transfer", lambda plan: None)
    transfer_calls = []
    def transfer_stage(*args, **kwargs):
        transfer_calls.append((kwargs["split"], kwargs["fold"], kwargs["seed"]))
        return True, {"reason": "transfer_confirmed"}
    monkeypatch.setattr(transfer, "evaluate_transfer", transfer_stage)
    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/a.py").write_text("value = 1\n")
    lead = PatchSite("execution", "fedot/a.py", 1)
    candidate = PatchCandidate("candidate", edits=[PatchEdit("fedot/a.py", "value = 1", "value = 2")])
    plan = {"target_task": "target", "data_hashes": {"target": "frozen-data"}, "transfer": {}}
    calls = []
    def pair(src, exp, tasks, *, split, fold=None, seed=42, target=None, operation_params=None):
        assert target == "target" and split != "final"
        calls.append((split, fold, seed))
        # The template source is neutral. The controller must rely on the
        # separately mocked cross-dataset signal, not turn this one source
        # into a hidden veto during seeds or CV folds.
        before, after = score(target, .7), score(target, .7)
        for row in (before, after):
            row.data_evidence.update(data_hash="frozen-data", validation_ids=[f"{split}:{fold}"])
            row.coverage = ({"file_path": "fedot/a.py", "line_ranges": [[1, 1]]},)
        return {target: before}, {target: after}
    monkeypatch.setattr(study, "_pair", pair)
    monkeypatch.setattr(study, "finalize_batch", lambda *a: pytest.fail("premature FINAL"))
    decision = study.evaluate_candidate(source, source, candidate, VerificationResult("verified_bug"), plan,
        lead=lead, tasks=("target",), study=tmp_path / "study", workspace=tmp_path / "case")
    assert decision.reason == "metric_ready_waiting_for_frozen_FINAL_batch"
    assert decision.dev_keep and not decision.keep and not decision.metric_goal_keep
    assert calls == [("dev", None, 42), ("dev", None, 43), ("dev", None, 44),
                     ("dev", 0, 42), ("dev", 1, 42), ("dev", 2, 42), ("shadow", None, 42)]
    assert transfer_calls == [("dev", None, 42), ("shadow", None, 42)]
    assert len(list((tmp_path / "study/ready").glob("*.json"))) == 1


def test_final_resumes_frozen_batch_not_new_ready_files(tmp_path, monkeypatch, specs):
    from fedotllm.agents.evolve.controller import transfer
    from fedotllm.agents.evolve.controller.measurement_budget import MeasurementBudget
    monkeypatch.setattr(transfer, "validate_transfer", lambda plan: None)
    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/a.py").write_text("value = 1\n")
    root = tmp_path / "study"
    root.mkdir()
    manifest = {"source_hash": study.source_fingerprint(source),
                "acceptance_protocol": study.acceptance_protocol_fingerprint(), "tasks": ["target"]}
    (root / "study.json").write_text(json.dumps(manifest))
    frozen = [{"site_key": str(i), "candidate": {"candidate_id": str(i), "edits": [
        {"file_path": "fedot/a.py", "old_code": "value = 1", "new_code": f"value = {i + 2}"}]},
        "plan": {"target_task": "target", "transfer": {}}, "development_evidence": []} for i in range(3)]
    (root / "final-batch.json").write_text(json.dumps({"study": manifest, "candidates": frozen}))
    (root / "final-checkpoint.jsonl").write_text(json.dumps({"candidate": "0", "passed": False, "result": {}}) + "\n")
    calls = []
    budget = MeasurementBudget(max_pairs=10, reserve_final_pairs=0)
    def stage(*args, **kwargs):
        calls.append(kwargs["split"])
        assert kwargs["measurement_budget"] is budget
        return False, {"reason": "target_no_gain"}, {}, {}
    monkeypatch.setattr(study, "_stage", stage)
    result = study.finalize_batch(source, root, measurement_budget=budget)
    assert calls == ["final", "final"]
    assert len(result["cases"]) == 3 and not result["goal_achieved"]
    assert study.finalize_batch(source, root) == result
    assert calls == ["final", "final"]
    manifest["acceptance_protocol"] = "obsolete-single-dataset-policy"
    (root / "study.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="acceptance protocol changed"):
        study.finalize_batch(source, root)
