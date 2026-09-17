"""Regression checks for failures missed by mocked campaign tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from fedotllm.agents.evolve.evaluation.compare import compare_pack
from fedotllm.agents.evolve.evaluation.scorer import rmse_score, roc_auc_score
from fedotllm.agents.evolve.storage.scoreboard import (
    append_attempt,
    append_final,
    summarize,
)
from fedotllm.agents.evolve.types import Decision, MatchSite, ScoreResult


def test_shipped_manifest_is_loadable():
    from fedotllm.agents.evolve.evaluation.manifest import load_manifest

    manifest = load_manifest()
    assert len(manifest["fedot_commit"]) == 40
    assert manifest["files"]


def test_proposal_memory_outlives_exact_site_cooldown(tmp_path):
    from fedotllm.agents.evolve.storage.replay import (
        recent_completed_hypotheses_from_findings,
        recent_completed_sites_from_findings,
    )

    rows = []
    for number in (1, 2):
        common = {"run_id": str(number), "run_number": number, "source_hash": "source"}
        rows.extend(
            [
                {
                    **common,
                    "record_type": "run",
                    "event": "run_start",
                    "campaign_config": {"score_protocol_hash": "score"},
                },
                {
                    **common,
                    "record_type": "finding",
                    "score_protocol_hash": "score",
                    "lead": {
                        "file_path": "fedot/model.py",
                        "line": number,
                        "proposed_change": f"proposal {number}",
                    },
                },
                {
                    **common,
                    "record_type": "run",
                    "event": "run_end",
                    "immutable_source": True,
                },
            ]
        )
    path = tmp_path / "findings.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    scope = {"source_hash": "source", "score_protocol_hash": "score"}
    assert recent_completed_sites_from_findings(path, **scope) == {
        ("fedot/model.py", 2)
    }
    memory = recent_completed_hypotheses_from_findings(path, **scope)
    assert {row["proposed_change"] for row in memory} == {"proposal 1", "proposal 2"}


def test_final_confirmed_task_is_deduplicated_by_behavior_not_patch(tmp_path):
    from fedotllm.agents.evolve.storage.replay import solved_lift_tasks_from_findings

    path = tmp_path / "findings.jsonl"
    rows = [
        {
            "event": "run_start",
            "run_id": "old",
            "source_hash": "same-source",
        },
        {
            "event": "final_outcome",
            "run_id": "old",
            "candidate_id": "fix-in-catboost",
            "final": {
                "keep": True,
                "regression_deltas": {
                    "pca->catboost": 0.099,
                    "rf": 0.0,
                },
            },
        },
        {
            "event": "run_start",
            "run_id": "other-source-run",
            "source_hash": "other-source",
        },
        {
            "event": "final_outcome",
            "run_id": "other-source-run",
            "candidate_id": "unrelated",
            "final": {"keep": True, "regression_deltas": {"rf": 0.5}},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert solved_lift_tasks_from_findings(path, source_hash="same-source") == {
        "pca->catboost"
    }


def test_confirmed_defect_memory_crosses_versions_without_blocking_a_site(tmp_path):
    from fedotllm.agents.evolve.storage.replay import (
        confirmed_hypotheses_from_findings,
    )

    path = tmp_path / "findings.jsonl"
    rows = [
        {
            "record_type": "finding",
            "candidate_id": "old-fix",
            "source_hash": "same-source",
            "outcome": "confirmed_fix_metric_neutral",
            "lead": {
                "file_path": "fedot/preprocessing.py",
                "line": 207,
            },
        },
        {
            "record_type": "finding",
            "candidate_id": "withdrawn-fix",
            "source_hash": "same-source",
            "outcome": "correctness_keep",
            "lead": {"file_path": "fedot/other.py", "line": 10},
        },
        {
            "record_type": "rejudge",
            "candidate_id": "withdrawn-fix",
            "outcome": "semantic_duplicate",
        },
        {
            "record_type": "finding",
            "candidate_id": "other-source-fix",
            "source_hash": "other-source",
            "outcome": "correctness_keep",
            "lead": {"file_path": "fedot/unrelated.py", "line": 1},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    memory = confirmed_hypotheses_from_findings(path)
    assert {row["file_path"] for row in memory} == {
        "fedot/preprocessing.py",
        "fedot/unrelated.py",
    }
    assert all(row["history_kind"] == "confirmed_defect" for row in memory)


def test_confirmed_defect_memory_accepts_append_only_lead_correction(tmp_path):
    from fedotllm.agents.evolve.storage.replay import (
        confirmed_hypotheses_from_findings,
    )

    path = tmp_path / "findings.jsonl"
    rows = [
        {
            "record_type": "finding",
            "candidate_id": "fixed",
            "outcome": "correctness_keep",
            "lead": None,
        },
        {
            "record_type": "rejudge",
            "candidate_id": "fixed",
            "outcome": "correctness_keep",
            "lead": {
                "file_path": "fedot/core/data/merge.py",
                "line": 12,
                "mechanism": "parent rows are joined by position",
                "proposed_change": "align rows by their unique index",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert confirmed_hypotheses_from_findings(path) == [
        {
            "file_path": "fedot/core/data/merge.py",
            "line": 12,
            "mechanism": "parent rows are joined by position",
            "proposed_change": "align rows by their unique index",
            "history_kind": "confirmed_defect",
        }
    ]


def test_confirmed_defect_match_compares_mechanism_not_line_or_site_alone():
    from fedotllm.agents.evolve.discovery.selection import (
        SiteProposal,
        _matches_confirmed_defect_proposal,
    )

    history = [
        {
            "history_kind": "confirmed_defect",
            "file_path": "fedot/preprocessing.py",
            "line": 207,
            "mechanism": (
                "numeric inf target bypasses NaN row removal and reaches the "
                "regression estimator during fit"
            ),
            "proposed_change": (
                "replace non-finite target values with NaN before dropping target rows"
            ),
        }
    ]
    repeated = SiteProposal(
        why="infinite numeric target survives preprocessing",
        mechanism=(
            "inf target bypasses NaN row removal and reaches the regression "
            "estimator during fit"
        ),
        proposed_change=(
            "replace non-finite target values with NaN before dropping target rows"
        ),
    )
    different = SiteProposal(
        why="fitted schema is mutated at prediction time",
        mechanism=(
            "a reordered prediction table overwrites stored column types and makes "
            "the next prediction depend on the previous request"
        ),
        proposed_change="copy the prediction schema before correcting column types",
    )

    assert _matches_confirmed_defect_proposal(
        repeated, "fedot/preprocessing.py", history
    )
    assert not _matches_confirmed_defect_proposal(
        different, "fedot/preprocessing.py", history
    )


def test_known_mechanism_registry_contains_all_seven_benchmark_defects(tmp_path):
    from fedotllm.agents.evolve.benchmark.known_mechanisms import (
        combined_known_mechanisms,
        load_known_mechanisms,
    )

    rows = load_known_mechanisms()
    assert {row["case_id"] for row in rows} == {
        "partial_poly_params",
        "lda_effective_solver",
        "lagged_reproducibility",
        "polyfit_parameter_identity",
        "nonfinite_target_preprocessing",
        "merge_parent_index_alignment",
        "single_column_multits_lagged",
    }
    assert len({row["contract_id"] for row in rows}) == 7

    findings = tmp_path / "findings.jsonl"
    findings.write_text(
        json.dumps(
            {
                "record_type": "finding",
                "outcome": "correctness_keep",
                "candidate_id": "new",
                "lead": {
                    "file_path": "fedot/new.py",
                    "line": 8,
                    "mechanism": "a separate lifecycle defect",
                    "proposed_change": "preserve fitted state",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    combined = combined_known_mechanisms(findings)
    assert len(combined) == 8
    assert any(row["file_path"] == "fedot/new.py" for row in combined)


def test_atomic_checkpoint_preserves_scout_candidates_and_history(tmp_path):
    from dataclasses import asdict

    from fedotllm.agents.evolve.storage.checkpoint import (
        checkpoint_leads,
        load_checkpoint,
        save_checkpoint,
    )

    lead = MatchSite(
        "llm",
        "fedot/core/example.py",
        12,
        mechanism="rows are joined by position",
        proposed_change="align by unique index",
        expected_metric_effect="restore row identity",
        hypothesis_kind="correctness",
    )
    first = save_checkpoint(
        tmp_path,
        stage="scout_pick",
        run_id="run-a",
        selected_leads=[asdict(lead)],
    )
    second = save_checkpoint(
        tmp_path,
        stage="hypothesis",
        run_id="run-a",
        hypothesis={"id": "h-1"},
    )

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert load_checkpoint(tmp_path)["selected_leads"][0]["file_path"] == lead.file_path
    assert checkpoint_leads(tmp_path) == [lead]
    history = [
        json.loads(line)
        for line in (tmp_path / "checkpoint_history.jsonl").read_text().splitlines()
    ]
    assert [row["stage"] for row in history] == ["scout_pick", "hypothesis"]


def test_discovery_memory_survives_test_only_protocol_changes(tmp_path):
    from fedotllm.agents.evolve.storage.replay import (
        recent_completed_sites_from_findings,
        tried_patch_hashes_from_findings,
        recent_completed_hypotheses_from_findings,
    )

    path = tmp_path / "history.jsonl"
    rows = []
    for number, score, complete in (
        (1, "score-a", True),
        (2, "score-b", True),
        (3, "score-a", False),
    ):
        common = {"run_id": str(number), "run_number": number, "source_hash": "source"}
        rows.extend(
            [
                {
                    **common,
                    "record_type": "run",
                    "event": "run_start",
                    "campaign_config": {
                        "score_protocol_hash": score,
                        "evaluation_protocol_hash": "old-tests",
                    },
                },
                {
                    **common,
                    "record_type": "finding",
                    "score_protocol_hash": score,
                    "evaluation_protocol_hash": "old-tests",
                    "patch_hash": str(number),
                    "final_score": "PRIVATE_FINAL_SCORE",
                    "decision": "PRIVATE_FINAL_VERDICT",
                    "lead": {
                        "file_path": "fedot/model.py",
                        "line": number,
                        "mechanism": f"source mechanism {number}",
                        "proposed_change": "preserve fitted state",
                        "evidence": ["DO_NOT_FORWARD_RUNTIME_OR_METRIC_FEEDBACK"],
                    },
                },
            ]
        )
        if complete:
            rows.append(
                {
                    **common,
                    "record_type": "run",
                    "event": "run_end",
                    "immutable_source": True,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows))
    scope = {"source_hash": "source", "evaluation_protocol_hash": "new-tests"}
    assert recent_completed_sites_from_findings(
        path, **scope, score_protocol_hash="score-a"
    ) == {
        ("fedot/model.py", 1),
    }
    assert not recent_completed_sites_from_findings(path, **scope)
    assert not recent_completed_sites_from_findings(
        path, **scope, score_protocol_hash="unknown"
    )
    assert not tried_patch_hashes_from_findings(path, **scope)
    memory = recent_completed_hypotheses_from_findings(
        path, **scope, score_protocol_hash="score-a"
    )
    assert memory == [
        {
            "file_path": "fedot/model.py",
            "line": 1,
            "mechanism": "source mechanism 1",
            "proposed_change": "preserve fitted state",
        }
    ]
    assert "PRIVATE" not in json.dumps(memory)
    assert "DO_NOT_FORWARD" not in json.dumps(memory)


@pytest.mark.parametrize(
    "first,second,expected,calls",
    [
        ("missing", "changed", "changed", 2),
        ("invalid", "changed", "changed", 2),
        ("no_change", "changed", "changed", 2),
        ("no_change", "no_change", "no_change", 2),
        ("no_change", "patched_error", "no_change", 2),
        ("patched_error", "changed", "patched_error", 1),
        ("changed", "changed", "changed", 1),
    ],
)
def test_prior_probe_is_rerun_and_cannot_waive_a_runtime_regression(
    first, second, expected, calls
):
    from fedotllm.agents.evolve.controller.probes import compare_with_prior_probe

    observed = []

    def compare(source, patched, code):
        observed.append((source, patched, code))
        return {"status": first if len(observed) == 1 else second, "code": code}

    result = compare_with_prior_probe(
        "stock",
        "new-patch",
        "print(1)",
        prior_probe=("earlier-candidate", "print(2)"),
        compare_fn=compare,
    )
    assert result["status"] == expected
    assert len(observed) == calls
    assert all(row[:2] == ("stock", "new-patch") for row in observed)
    if expected == "changed" and calls == 2:
        assert result["submitted_probe_result"]["status"] == first
        assert result["reused_probe_from_candidate"] == "earlier-candidate"


def test_prior_probe_does_not_repeat_an_identical_diagnostic():
    from fedotllm.agents.evolve.controller.probes import compare_with_prior_probe

    calls = []

    def compare(*args):
        calls.append(args)
        return {"status": "no_change"}

    compare_with_prior_probe(
        "stock",
        "patch",
        "print(1)",
        prior_probe=("old", "print( 1 ) # same code"),
        compare_fn=compare,
    )
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_contract_repair_is_isolated_and_preserves_failures(tmp_path, failure):
    from fedotllm.agents.evolve.evaluation.test_contracts import (
        TEST_PATH,
        run_with_test_repairs,
    )
    from fedotllm.agents.evolve.types import TestResult

    original = (
        Path(__file__).parent / "fixtures/frozen_knn_parameter_test.txt"
    ).read_text()
    path = tmp_path / "source" / TEST_PATH
    path.parent.mkdir(parents=True)
    path.write_text(original)
    runtime = path.parents[3] / "fedot/a.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("VALUE = 1\n")
    copies = []

    def runner(checkout):
        copies.append(checkout)
        repaired = (checkout / TEST_PATH).read_text()
        assert "actual_params =" in repaired
        assert "assert not np.array_equal" not in repaired
        assert (checkout / "fedot/a.py").read_text() == "VALUE = 1\n"
        if failure:
            raise RuntimeError("test process failed")
        return TestResult("test_failures", 1, failed_nodes={"another_test"})

    if failure:
        with pytest.raises(RuntimeError, match="test process failed"):
            run_with_test_repairs(path.parents[3], runner)
    else:
        result = run_with_test_repairs(path.parents[3], runner)
        assert result.failed_nodes == {"another_test"}
        assert result.exit_code == 1
    assert copies and not copies[0].exists()
    assert path.read_text() == original
    assert runtime.read_text() == "VALUE = 1\n"


def test_contract_repair_does_not_override_changed_upstream_test():
    from fedotllm.agents.evolve.evaluation.test_contracts import (
        repaired_contract_source,
    )

    original = (
        Path(__file__).parent / "fixtures/frozen_knn_parameter_test.txt"
    ).read_text()
    assert repaired_contract_source(original) != original
    changed = original.replace("p=1", "p=2")
    assert repaired_contract_source(changed) == changed
    repaired = repaired_contract_source(original)
    assert repaired_contract_source(repaired) == repaired


def test_pytest_feedback_keeps_assertion_instead_of_captured_training_logs():
    from fedotllm.agents.evolve.discovery.signals import pytest_failure_excerpt

    node = "test/unit/test_model.py::test_explicit_params"
    output = (
        "________________ test_explicit_params ________________\n"
        "test/unit/test_model.py:40: in test_explicit_params\n"
        "    assert not np.array_equal(custom, default)\n"
        "E   AssertionError: both predictions are equal\n"
        "---------------- Captured log call ----------------\n"
        + "DEBUG training log with no assertion\n" * 1000
        + "================ short test summary info ================\n"
        + f"FAILED {node} - AssertionError\n"
    )
    feedback = pytest_failure_excerpt(output, {node}, max_chars=600)
    assert "assert not np.array_equal(custom, default)" in feedback
    assert "AssertionError: both predictions are equal" in feedback
    assert "DEBUG training log" not in feedback
    assert len(feedback) <= 600


@pytest.mark.parametrize("metric", [rmse_score, roc_auc_score])
@pytest.mark.parametrize(
    "target,prediction",
    [
        ([0, 1], [0]),
        ([], []),
        ([0, 1], [np.nan, 1]),
        ([0, 1], [0, np.inf]),
        ([0, np.inf], [0, 1]),
    ],
)
def test_metric_rejects_incomplete_or_nonfinite_predictions(metric, target, prediction):
    with pytest.raises(ValueError):
        metric(target, prediction)


def test_metric_agrees_with_sklearn_on_ties_and_column_vectors():
    from sklearn.metrics import mean_squared_error, roc_auc_score as sklearn_auc

    target = np.array([0, 1, 0, 1]).reshape(-1, 1)
    prediction = np.array([0.2, 0.5, 0.5, 0.9])
    assert roc_auc_score(target, prediction) == pytest.approx(
        sklearn_auc(target, prediction)
    )
    assert rmse_score(target, prediction) == pytest.approx(
        np.sqrt(mean_squared_error(target, prediction))
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_protect_score_cannot_pass(bad):
    stock = {key: ScoreResult(key, "ok", 0.7) for key in ("rf", "lgbm")}
    patched = {
        "rf": ScoreResult("rf", "ok", 0.8),
        "lgbm": ScoreResult("lgbm", "ok", bad),
    }
    result = compare_pack(stock, patched, lift_ids=("rf",), protect_ids=("rf", "lgbm"))
    assert not result.keep
    assert result.infrastructure_error


def test_scoreboard_preserves_small_relative_keep_and_last_drop(tmp_path):
    stock = {"arctic": ScoreResult("arctic", "ok", 0.1)}
    patched = {"arctic": ScoreResult("arctic", "ok", 0.098)}
    keep = compare_pack(stock, patched, lift_ids=("arctic",), protect_ids=("arctic",))
    assert keep.keep
    append_attempt(
        tmp_path, lead=None, candidate=None, stock=stock, patched=patched, decision=keep
    )
    append_final(tmp_path, stock=stock, patched=patched, decision=keep)
    append_attempt(
        tmp_path,
        lead=None,
        candidate=None,
        stock=stock,
        patched=stock,
        decision=Decision(False, "neutral", 0.0),
    )
    summary = summarize(tmp_path)
    assert summary["getting_better"]
    assert summary["getting_better_final"]
    assert not summary["last_keep"]


def test_correctness_keep_cannot_borrow_a_rejected_metric_gain(tmp_path):
    append_attempt(
        tmp_path,
        lead=None,
        candidate=None,
        stock={},
        patched={},
        decision=Decision(False, "regression", 0.2),
    )
    append_attempt(
        tmp_path,
        lead=None,
        candidate=None,
        stock={},
        patched={},
        decision=Decision(
            False, "confirmed_fix_metric_neutral", 0.0, correctness_keep=True
        ),
    )
    assert not summarize(tmp_path)["getting_better"]


def test_scout_skip_does_not_become_static_hypothesis(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.discovery import discover
    from fedotllm.agents.evolve.discovery.discover import SiteProposal

    rel = "fedot/core/operations/model.py"
    source = tmp_path / rel
    source.parent.mkdir(parents=True)
    source.write_text("def transform(x):\n    return x + 1\n")
    monkeypatch.setattr(
        discover, "static_leads", lambda *_: [MatchSite("static", rel, 1)]
    )

    class Skip:
        def create(self, *_args, **_kwargs):
            return SiteProposal(status="skip", file_path=rel)

    assert discover.discover_leads(tmp_path, inference=Skip(), limit=5) == []


def test_importing_logger_does_not_truncate_previous_run(tmp_path):
    import os
    import subprocess
    import sys

    (tmp_path / "fedotllm.log").write_text("previous campaign\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    subprocess.run(
        [sys.executable, "-c", "import fedotllm.log"], cwd=tmp_path, env=env, check=True
    )
    assert "previous campaign" in (tmp_path / "fedotllm.log").read_text()


def test_campaign_root_resolves_scoreboard_and_replay_after_move(tmp_path):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import load_replay

    run = tmp_path / "runs" / "first"
    append_attempt(
        run,
        lead=None,
        candidate=None,
        stock={},
        patched={},
        decision=Decision(True, "keep", 0.02),
    )
    append_journal(
        run / "journal.jsonl", {"event": "decision", "keep": True, "reason": "keep"}
    )
    (tmp_path / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": "first",
                "workspace": "/old/moved/workspace/runs/first",
            }
        )
    )
    assert summarize(tmp_path)["attempts"] == 1
    assert summarize(tmp_path)["getting_better"]
    assert load_replay(tmp_path)["keep"]


def test_worker_thread_defaults_and_explicit_override(tmp_path, monkeypatch):
    from fedotllm.agents.evolve.execution.process import clean_subprocess_env

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    env = clean_subprocess_env(tmp_path)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "2"


def test_dev_feedback_reveals_moved_crash_without_leaking_unchanged_protect_crash():
    from fedotllm.agents.evolve.controller.feedback import _dev_feedback

    def failure(root, method, line):
        return ScoreResult(
            "pca->catboost",
            "crash",
            0.5,
            detail="IndexError: index 1",
            traceback=f'  File "{root}/fedot/models.py", line {line}, in {method}\n'
            '  File "' + root + '/fedot/data.py", line 718, in get_not_encoded_data\n',
        )

    stock = {"pca->catboost": failure("/stock", "fit", 10)}
    unchanged = {"pca->catboost": failure("/experiment", "fit", 25)}
    moved = {"pca->catboost": failure("/experiment", "predict_proba", 40)}
    decision = Decision(
        False,
        "target_delta 0 below threshold",
        0.0,
        regression_deltas={"pca->catboost": 0.0},
    )
    assert "pca->catboost" not in _dev_feedback(decision, unchanged, stock=stock)
    feedback = _dev_feedback(decision, moved, stock=stock)
    assert "predict_proba" in feedback
    assert "IndexError" in feedback
    assert "/experiment" not in feedback


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_controller_observed_crash(tmp_path, legacy):
    from dataclasses import asdict
    from fedotllm.agents.evolve.agents.verifier import (
        verification_from_observed_crash,
        is_controller_observed_crash,
    )
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import load_resume_branch

    lead = MatchSite(
        "execution",
        "fedot/core/data.py",
        10,
        evidence=(
            "stock runtime crash: IndexError: index 1",
            "FEDOT frame chain: fedot/core/data.py:10 in fit",
            "workload operation: catboost",
        ),
    )
    verification = verification_from_observed_crash(lead)
    assert verification is not None
    row = asdict(verification)
    if legacy:
        row.pop("evidence")
    append_journal(
        tmp_path / "journal.jsonl",
        {
            **row,
            "event": "verification",
            "hypothesis_id": "h",
            "lead": asdict(lead),
        },
    )
    append_journal(
        tmp_path / "journal.jsonl",
        {
            "event": "decision",
            "candidate": "c",
            "hypothesis_id": "h",
            "lead": asdict(lead),
        },
    )
    resumed = load_resume_branch(tmp_path, candidate_id="c")
    assert resumed is not None
    assert is_controller_observed_crash(resumed["verification"])


def test_resume_recovers_candidate_artifact_without_decision(tmp_path):
    from dataclasses import asdict
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import load_resume_branch

    lead = MatchSite(
        "invariant",
        "fedot/core/example.py",
        17,
        why="preserve row identity",
    )
    append_journal(
        tmp_path / "journal.jsonl",
        {
            "event": "verification",
            "hypothesis_id": "h-interrupted",
            "lead": asdict(lead),
            "status": "quality_hypothesis",
            "claim": "rows keep their ids",
            "expected": "merged values follow ids",
            "observed": "possible positional merge",
        },
    )
    folder = tmp_path / "candidates" / "saved-candidate"
    folder.mkdir(parents=True)
    (folder / "candidate.json").write_text(
        json.dumps(
            {
                "candidate_id": "saved-candidate",
                "hypothesis_id": "h-interrupted",
                "lead": asdict(lead),
                "status": "awaiting_probe_repair",
                "edits": [
                    {
                        "file_path": "fedot/core/example.py",
                        "old_code": "return values",
                        "new_code": "return values[idx]",
                    }
                ],
                "rationale": "align values to ids",
                "contract": "row identity is stable",
                "behavior_probe": "broken()",
            }
        ),
        encoding="utf-8",
    )
    (folder / "probe_preflight_failed.txt").write_text(
        "status=runtime_error\nstderr_tail=NameError: broken",
        encoding="utf-8",
    )

    resumed = load_resume_branch(tmp_path, candidate_id="saved-candidate")

    assert resumed is not None
    assert resumed["candidate"].candidate_id == "saved-candidate"
    assert resumed["candidate"].edits[0].new_code == "return values[idx]"
    assert resumed["candidate_status"] == "awaiting_probe_repair"
    assert "NameError: broken" in resumed["resume_diagnostic"]
    assert resumed["hypothesis_id"] == "h-interrupted"


@pytest.mark.parametrize(
    "status,expected_confirmations", [("verified_bug", 2), ("quality_hypothesis", 0)]
)
def test_ablation_uses_the_same_evidence_gate_as_campaign(
    tmp_path, monkeypatch, status, expected_confirmations
):
    from fedotllm.agents.evolve.controller import ablation
    from fedotllm.agents.evolve.types import (
        PatchCandidate,
        PatchEdit,
        TestResult,
        VerificationResult,
    )

    monkeypatch.setattr(
        ablation, "create_experiment_checkout", lambda *_a, **_k: tmp_path
    )
    monkeypatch.setattr(ablation, "discard_experiment_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(ablation, "apply_patch", lambda *_a: True)
    monkeypatch.setattr(ablation, "import_error", lambda *_a: None)
    confirmations = []

    def confirm(*_args, **_kwargs):
        confirmations.append(True)
        return False, {}

    ablation.ablate_candidate(
        tmp_path,
        tmp_path,
        "run",
        PatchCandidate(
            "c",
            edits=[
                PatchEdit("fedot/a.py", "a", "b"),
                PatchEdit("fedot/b.py", "x", "y"),
            ],
            behavior_probe="an optional invalid synthetic probe",
        ),
        ("rf",),
        ("rf",),
        ("rf",),
        tmp_path / "ablation.jsonl",
        baseline_tests=TestResult("passed", 0),
        verification=VerificationResult(status),
        compare_behavior_probe_fn=lambda *_a: {"status": "invalid"},
        behavior_probe_blocks_fn=lambda _r: True,
        measure_fedot_tests_fn=lambda *_a: TestResult("passed", 0),
        confirm_dev_fn=confirm,
    )
    assert len(confirmations) == expected_confirmations


def test_symbol_tool_empty_query_does_not_return_unrelated_code(tmp_path):
    from fedotllm.agents.evolve.discovery.research_tools import symbol_runtime

    root = tmp_path / "fedot/core/operations"
    root.mkdir(parents=True)
    (root / "example.py").write_text("class Unrelated:\n    pass\n")
    output = symbol_runtime(tmp_path, "")
    assert "empty symbol query" in output
    assert "Unrelated" not in output


def test_controller_correctness_replay_uses_only_crashes_that_reached_lead():
    from fedotllm.agents.evolve.controller.campaign import (
        _controller_crashes_for_lead,
    )

    lead = MatchSite("execution", "fedot/core/target.py", 10)
    stock = {
        "trace_hit": ScoreResult(
            "trace_hit",
            "crash",
            0.5,
            traceback='File "/checkout/fedot/core/target.py", line 10',
        ),
        "coverage_hit": ScoreResult(
            "coverage_hit",
            "crash",
            0.5,
            coverage=({"file_path": "fedot/core/target.py", "line": 4},),
        ),
        "unrelated": ScoreResult(
            "unrelated",
            "crash",
            0.5,
            traceback='File "/checkout/fedot/core/other.py", line 10',
        ),
        "healthy": ScoreResult("healthy", "ok", 0.8),
    }

    assert _controller_crashes_for_lead(stock, lead) == (
        "trace_hit",
        "coverage_hit",
    )


def test_scout_symbol_alias_retrieves_requested_class(tmp_path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot/core/operations"
    root.mkdir(parents=True)
    (root / "knn.py").write_text(
        "class Knn:\n    def fit(self, data):\n        return data\n"
    )
    prompts = []

    class Inference:
        def create(self, prompt, _schema):
            prompts.append(prompt)
            if len(prompts) == 1:
                return SiteProposal(status="symbol", symbol="Knn")
            assert "action=symbol query='Knn'" in prompt
            assert "exact class Knn" in prompt
            return SiteProposal(status="skip")

    discover_leads(tmp_path, inference=Inference(), limit=1, max_picks=1)
    assert len(prompts) == 2


def test_scout_long_read_keeps_tool_header_and_source_start():
    from fedotllm.agents.evolve.discovery.selection import _recent_tool_context

    header = "Step 3 action=read Opened fedot/core/foo.py:\n    1|class Needed:\n"
    latest = header + "x" * 8000 + "\n  200|return value"
    output = _recent_tool_context(["old output", latest])
    assert len(output) <= 6000
    assert output.startswith(header)
    assert "old output" not in output
    assert "middle truncated" in output
    assert output.endswith("return value")


def test_runtime_class_priority_does_not_promote_a_shared_name_prefix():
    from fedotllm.agents.evolve.discovery.selection import _execution_causal_priority
    from fedotllm.agents.evolve.types import MatchSite

    evidence = (
        "runtime operation instances: lgbm/FedotLightGBMClassificationImplementation fit params={}",
    )
    dispatcher = MatchSite(
        "execution",
        "fedot/api/main.py",
        100,
        "executed symbol Fedot.fit",
        evidence=evidence,
        signals=("executed", "method"),
    )
    concrete = MatchSite(
        "execution",
        "fedot/core/operations/evaluation/operation_implementations/models.py",
        20,
        "executed symbol FedotLightGBMClassificationImplementation.fit",
        evidence=evidence,
        signals=("executed", "method"),
    )
    transform = MatchSite(
        "execution",
        "fedot/core/operations/evaluation/operation_implementations/data_operations/ts.py",
        20,
        "executed symbol LaggedImplementation.transform",
        evidence=evidence,
        signals=("executed", "method"),
    )
    assert _execution_causal_priority(concrete) == 0
    assert _execution_causal_priority(concrete) < _execution_causal_priority(transform)
    assert _execution_causal_priority(transform) < _execution_causal_priority(
        dispatcher
    )


def test_stage_clients_preserve_explicit_reasoning_output_budget(monkeypatch):
    from fedotllm.agents.evolve.__main__ import _stage_inferences
    from types import SimpleNamespace
    from fedotllm.configs.schema import EvolveConfig, LLMConfig
    from fedotllm.configs import loader
    import fedotllm.llm as llm

    config = SimpleNamespace(
        llm=LLMConfig(
            api_key="test",
            fallback_models="another-model",
            completion_params={"max_tokens": 8000},
        ),
        evolve=EvolveConfig(),
    )
    monkeypatch.setenv("FEDOTLLM_LLM_API_KEY", "test")
    monkeypatch.setattr(loader, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(llm, "AIInference", lambda cfg: cfg)
    scout, verifier, fixer = _stage_inferences("test")
    assert scout.completion_params["max_tokens"] == 8000
    assert verifier.completion_params["max_tokens"] == 8000
    assert fixer.completion_params["max_tokens"] == 8000
    assert scout.fallback_models == ""
    assert verifier.fallback_models == ""
    assert fixer.fallback_models == ""
    scout.completion_params["max_tokens"] = 123
    assert fixer.completion_params["max_tokens"] == 8000
    assert config.llm.completion_params["max_tokens"] == 8000


def test_runtime_context_preserves_training_rows_and_transformed_shape():
    from fedotllm.agents.evolve.controller.feedback import _runtime_operation_evidence

    row = {
        "operation": "lagged",
        "implementation": "LaggedImplementation",
        "stage": "fit",
        "input": {"features_shape": [2000], "active_width": 1},
        "output": {
            "features_shape": [2000],
            "predict_shape": [1880, 97],
            "active_width": 97,
        },
    }
    context = _runtime_operation_evidence(
        (row,), symbol="LaggedImplementation.transform"
    )
    assert "shape=[2000]->[1880, 97]" in context
    assert "width=1->97" in context
    assert "target" not in context
