from __future__ import annotations

from pathlib import Path

import pytest

from fedotllm.agents.evolve.evaluation.compare import compare_pack
from fedotllm.agents.evolve.discovery.context import context_from_lead, inspect_trace, show_source
from fedotllm.agents.evolve.discovery.discover import parse_pytest_output
from fedotllm.agents.evolve.execution.guard import guard_path
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.evaluation.tasks import load_task
from fedotllm.agents.evolve.types import (
    PatchCandidate,
    MatchSite,
    ScoreResult,
    SnippetResult,
    TestResult,
    EvolveRunPolicy,
)

FAST_RUN_POLICY = EvolveRunPolicy(
    verify_manifest=False,
    confirm_and_ablate=False,
    confirm_small_signals=False,
    evaluate_final=False,
    fedot_quality_jobs=False,
)


def _score(task_id: str, status: str, score: float) -> ScoreResult:
    return ScoreResult(task_id=task_id, status=status, score=score)


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


def test_guard_denies_evaluator_paths():
    assert guard_path("data/cases.json") == "deny"
    assert guard_path("research/evolve/evolve_agent/scorer.py") == "deny"
    assert guard_path("fedotllm/agents/evolve/evaluation/scorer.py") == "deny"
    assert guard_path("fedotllm/llm.py") == "deny"
    assert guard_path("_local_fedot_patches/replacements/pca_keep_cats.py") == "deny"


def test_guard_allows_fedot_copy_inside_disposable_checkout(tmp_path):
    from fedotllm.agents.evolve.execution.guard import deny_write, repo_root

    checkout = tmp_path / "fedot-checkout"
    target = (
        checkout
        / "fedot/core/operations/evaluation/operation_implementations/data_operations"
        / "categorical_encoders.py"
    )
    target.parent.mkdir(parents=True)
    target.write_text("# disposable checkout fixture\n", encoding="utf-8")
    assert deny_write(target, checkout=checkout) is None
    assert guard_path(target, checkout=checkout) == "allow"
    assert deny_write(repo_root() / "fedotllm/llm.py", checkout=checkout)


def test_quality_suite_covers_real_cases():
    from fedotllm.agents.evolve.evaluation.tasks import (
        DEFAULT_COVERAGE_TASKS,
        coverage_task_limit,
        final_exam,
        hidden_exam,
        quality_suite,
    )

    suite = quality_suite()
    assert suite[:3] == (
        "imputation->scaling->logit",
        "normalization->knn",
        "poly_features->logit",
    )
    assert coverage_task_limit() == DEFAULT_COVERAGE_TASKS == 12
    covered_operations = {
        node for task_id in suite[:coverage_task_limit()] for node in load_task(task_id).nodes
    }
    assert {
        "simple_imputation",
        "scaling",
        "normalization",
        "poly_features",
        "smoothing",
        "gaussian_filter",
        "lagged",
        "logit",
        "knn",
        "lasso",
        "ridge",
        "catboost",
        "lgbm",
        "rf",
    } <= covered_operations
    assert "pca->catboost" in suite and "fast_ica->lgbm" in suite
    assert "cancer" in suite and "metocean" in suite
    assert "cancer->lgbm" in suite and "kc2->lgbm" in suite
    lift, protect = hidden_exam()
    assert lift == protect == suite
    assert final_exam() == (suite, suite)
    spec = load_task("catboost")
    assert spec.min_delta == 0.01
    assert spec.nodes == ("catboost",)
    rmse = load_task("cholesterol")
    assert rmse.higher_is_better is False
    assert rmse.min_delta_mode == "relative"
    # A library-wide LGBM default is protected on three independent frozen
    # datasets instead of two pipelines over only the scoring dataset.
    lgbm_datasets = {
        load_task(task_id).dataset
        for task_id in suite
        if "lgbm" in load_task(task_id).nodes
    }
    assert {"scoring", "cancer", "kc2"} <= lgbm_datasets


def test_inspect_trace_keeps_checkout_frames_only(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "pca.py"
    src.parent.mkdir(parents=True)
    src.write_text("def transform():\n    raise IndexError('x')\n", encoding="utf-8")
    traceback = (
        'Traceback (most recent call last):\n'
        f'  File "/not/gym/scorer.py", line 149, in score_task\n'
        f'    pipeline.fit(train)\n'
        f'  File "{src}", line 2, in transform\n'
        f'    raise IndexError("x")\n'
        "IndexError: x\n"
    )
    frames = inspect_trace(traceback, checkout=tmp_path)
    assert len(frames) == 1
    assert frames[0]["file"] == "fedot/core/pca.py"
    assert frames[0]["line"] == 2


def test_show_source_is_enclosing_function_not_file_window(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "data.py"
    src.parent.mkdir(parents=True)
    other = "\n".join(f"def other_{i}():\n    return {i}\n" for i in range(25))
    src.write_text(other + "\ndef get_not_encoded_data():\n    raise IndexError('x')\n", encoding="utf-8")
    lines = src.read_text(encoding="utf-8").splitlines()
    crash = next(i + 1 for i, line in enumerate(lines) if "IndexError" in line)
    text = show_source(src, checkout=tmp_path, around=crash)
    assert "get_not_encoded_data" in text
    assert "IndexError" in text
    assert "other_0" not in text
    assert "other_24" not in text


def test_show_source_denies_cases_json(tmp_path: Path):
    from fedotllm.agents.evolve.execution.guard import repo_root

    assert show_source(repo_root() / "data" / "cases.json", checkout=tmp_path) == ""


def test_apply_patch_search_replace(tmp_path: Path):
    target = tmp_path / "fedot" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("def f(x):\n    return x\n", encoding="utf-8")
    ok = apply_patch(
        tmp_path,
        PatchCandidate(
            candidate_id="c1",
            file_path="fedot/foo.py",
            old_code="    return x\n",
            new_code="    return x + 1\n",
        ),
    )
    assert ok is True
    assert "return x + 1" in target.read_text(encoding="utf-8")


def test_apply_patch_denies_evaluator(tmp_path: Path):
    import os

    from fedotllm.agents.evolve.execution.guard import repo_root

    escape = os.path.relpath(repo_root() / "data" / "cases.json", tmp_path)
    with pytest.raises(PermissionError):
        apply_patch(
            tmp_path,
            PatchCandidate(
                candidate_id="c2",
                file_path=escape,
                old_code="{",
                new_code="{ ",
            ),
        )


def test_parse_pytest_uses_fedot_frames_not_gym(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "data.py"
    src.parent.mkdir(parents=True)
    src.write_text("def get_x():\n    raise IndexError('x')\n", encoding="utf-8")
    text = (
        "FAILED test/unit/data/test_data.py::test_idx - IndexError: x\n"
        "Traceback (most recent call last):\n"
        f'  File "{tmp_path / "test" / "unit" / "data" / "test_data.py"}", line 4, in test_idx\n'
        "    get_x()\n"
        f'  File "{src}", line 2, in get_x\n'
        "    raise IndexError('x')\n"
        "IndexError: x\n"
    )
    leads = parse_pytest_output(text, tmp_path)
    assert leads
    assert leads[0].channel == "fedot_test"
    assert leads[0].file_path == "fedot/core/data.py"
    assert leads[0].line == 2
    assert "gym" not in leads[0].why
    assert "pca->catboost" not in leads[0].why


def test_signal_text_normalizes_bytes():
    from fedotllm.agents.evolve.discovery.signals import _as_text

    assert _as_text(None) == ""
    assert _as_text("ok") == "ok"
    assert _as_text(b"FAILED test/unit/x.py\n") == "FAILED test/unit/x.py\n"


def test_context_from_lead_includes_field_usage(tmp_path: Path):
    data = tmp_path / "fedot" / "core" / "data.py"
    pca = tmp_path / "fedot" / "core" / "operations" / "pca.py"
    data.parent.mkdir(parents=True)
    pca.parent.mkdir(parents=True)
    data.write_text(
        "def get_not_encoded_data(self):\n    return self.features[:, self.numerical_idx]\n",
        encoding="utf-8",
    )
    pca.write_text(
        "def transform(data):\n    data.numerical_idx = np.arange(data.features.shape[1])\n",
        encoding="utf-8",
    )
    ctx = context_from_lead(
        MatchSite(channel="trace", file_path="fedot/core/data.py", line=2, why="IndexError in get_not_encoded_data"),
        tmp_path,
    )
    assert "get_not_encoded_data" in ctx
    assert "Field numerical_idx" in ctx
    assert "fedot/core/operations/pca.py" in ctx


def test_context_from_lead(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "data.py"
    src.parent.mkdir(parents=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = context_from_lead(MatchSite(channel="lint", file_path="fedot/core/data.py", line=2, why="F821 x"), tmp_path)
    assert "fedot/core/data.py" in ctx
    assert "Channel:" not in ctx
    assert "pca->catboost" not in ctx
    assert "cases.json" not in ctx


def test_compare_pack_keep_and_protect():
    stock = {
        "catboost": _score("catboost", "ok", 0.85),
        "lgbm": _score("lgbm", "ok", 0.80),
        "rf": _score("rf", "ok", 0.78),
    }
    patched = {
        "catboost": _score("catboost", "ok", 0.87),
        "lgbm": _score("lgbm", "ok", 0.80),
        "rf": _score("rf", "ok", 0.78),
    }
    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost", "lgbm", "rf"),
        protect_ids=("catboost", "lgbm", "rf"),
    )
    assert decision.keep is True
    assert decision.target_delta == pytest.approx(0.02)


def test_compare_pack_uses_per_task_rmse_threshold():
    stock = {
        "catboost": _score("catboost", "ok", 0.85),
        "cholesterol": _score("cholesterol", "ok", 50.0),
    }
    patched = {
        "catboost": _score("catboost", "ok", 0.85),
        "cholesterol": _score("cholesterol", "ok", 49.0),
    }
    decision = compare_pack(
        stock,
        patched,
        lift_ids=("catboost", "cholesterol"),
        protect_ids=("catboost", "cholesterol"),
    )
    assert decision.keep is True
    assert decision.regression_deltas["cholesterol"] == pytest.approx(1.0)


def test_compare_pack_distinguishes_small_metric_signal_from_practical_keep():
    stock = {"cholesterol": _score("cholesterol", "ok", 50.0)}
    patched = {"cholesterol": _score("cholesterol", "ok", 49.8)}

    practical = compare_pack(
        stock,
        patched,
        lift_ids=("cholesterol",),
        protect_ids=("cholesterol",),
    )
    evidence = compare_pack(
        stock,
        patched,
        lift_ids=("cholesterol",),
        protect_ids=("cholesterol",),
        evidence_only=True,
    )

    assert practical.keep is False
    assert practical.reason.endswith("below per-task threshold")
    assert evidence.keep is True
    assert evidence.reason == "metric_signal"
    assert evidence.target_delta == pytest.approx(0.2)


def test_scoreboard_summarize_getting_better(tmp_path: Path):
    from fedotllm.agents.evolve.storage.scoreboard import append_attempt, summarize
    from fedotllm.agents.evolve.types import Decision as Dec

    stock = {"pca->catboost": _score("pca->catboost", "crash", 0.5)}
    patched = {"pca->catboost": _score("pca->catboost", "ok", 0.85)}
    append_attempt(
        tmp_path,
        lead=MatchSite(channel="fedot_test", file_path="fedot/core/data.py", line=2),
        candidate=None,
        stock=stock,
        patched=patched,
        decision=Dec(keep=True, reason="keep", target_delta=0.35, regression_deltas={}),
    )
    summary = summarize(tmp_path)
    assert summary["attempts"] == 1
    assert summary["keeps"] == 1
    assert summary["getting_better"] is True
    assert summary["getting_better_final"] is False
    assert summary["best_delta"] == pytest.approx(0.35)


def test_scoreboard_final_is_the_claim(tmp_path: Path):
    from fedotllm.agents.evolve.storage.scoreboard import append_attempt, append_final, summarize
    from fedotllm.agents.evolve.types import Decision as Dec

    stock = {"pca->catboost": _score("pca->catboost", "crash", 0.5)}
    patched = {"pca->catboost": _score("pca->catboost", "ok", 0.85)}
    append_attempt(
        tmp_path,
        lead=MatchSite(channel="fedot_test", file_path="fedot/core/data.py", line=2),
        candidate=None,
        stock=stock,
        patched=patched,
        decision=Dec(keep=True, reason="keep", target_delta=0.35, regression_deltas={}),
    )
    assert summarize(tmp_path)["getting_better_final"] is False
    append_final(
        tmp_path,
        decision=Dec(keep=True, reason="keep", target_delta=0.12, regression_deltas={}),
        stock={"knn": _score("knn", "ok", 0.70)},
        patched={"knn": _score("knn", "ok", 0.82)},
    )
    summary = summarize(tmp_path)
    assert summary["getting_better"] is True
    assert summary["getting_better_final"] is True
    assert summary["final_delta"] == pytest.approx(0.12)


def test_eval_passes_fixed_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.evaluation import eval as ev

    checkout = tmp_path / "fedot_src"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("EVOLVE_AGENT_SEED", "7")
    monkeypatch.setenv("EVOLVE_AGENT_TMP", str(tmp_path))

    from fedotllm.agents.evolve.execution.process import WorkerOutcome

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_k):
        captured["cmd"] = list(cmd)
        return WorkerOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ev, "run_worker", fake_run)
    result = ev.run_stock("catboost", checkout=checkout)
    assert "--seed" in captured["cmd"]
    assert "7" in captured["cmd"]
    assert result.seed == 7


def test_propose_prompt_hides_exam_catalog():
    from fedotllm.agents.evolve.agents.propose import build_prompt

    prompt = build_prompt("# fedot/core/data.py\n    return x")
    assert "pca->catboost" not in prompt
    assert "cases.json" not in prompt
    assert "leftover" not in prompt.lower()
    assert "data/cases.json" not in prompt


def test_propose_cannot_fix():
    from fedotllm.agents.evolve.agents.propose import PatchProposal, propose_patch

    class Inf:
        def create(self, _prompt, _model):
            return PatchProposal(file_path="fedot/core/a.py", status="cannot_fix")

    errors: list[str] = []
    assert propose_patch(inference=Inf(), context="# fedot/core/a.py\npass\n", errors=errors) is None
    assert errors == ["cannot_fix"]


def test_scout_stack_source_has_no_cases_catalog():
    from fedotllm.agents.evolve.execution.guard import repo_root

    folder = repo_root() / "fedotllm" / "agents" / "evolve"
    sources = (
        "discovery/discover.py",
        "agents/scout.py",
        "agents/fixer.py",
        "agents/propose.py",
        "discovery/context.py",
        "discovery/repo_map.py",
        "discovery/invariants.py",
        "execution/run_code.py",
    )
    for name in sources:
        text = (folder / name).read_text(encoding="utf-8")
        assert "from fedotllm.agents.evolve.evaluation.tasks" not in text
        assert "from fedotllm.agents.evolve.commands.recall" not in text
        assert "from fedotllm.agents.evolve.commands.repair" not in text
        assert "pca->catboost" not in text


def test_leads_from_scores_uses_trace_not_task_id(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import leads_from_scores

    src = tmp_path / "fedot" / "core" / "data.py"
    src.parent.mkdir(parents=True)
    src.write_text("def get_not_encoded_data():\n    raise IndexError('oob')\n", encoding="utf-8")
    traceback = (
        "Traceback (most recent call last):\n"
        f'  File "{src}", line 2, in get_not_encoded_data\n'
        "    raise IndexError('oob')\n"
        "IndexError: oob\n"
    )
    stock = {
        "pca->catboost": ScoreResult(
            task_id="pca->catboost",
            status="crash",
            score=0.5,
            traceback=traceback,
            detail="IndexError: oob",
            coverage=(
                {
                    "file_path": "fedot/core/operations/sklearn_transformations.py",
                    "symbol": "PCAImplementation",
                    "line": 130,
                    "count": 1,
                },
            ),
            dataflow=(
                {
                    "operation": "pca",
                    "stage": "fit",
                    "input": {"active_width": 3, "metadata_within_width": True},
                    "output": {"active_width": 1, "metadata_within_width": False},
                },
            ),
        )
    }
    leads = leads_from_scores(
        stock,
        tmp_path,
        operation_hints={"pca->catboost": ("pca",)},
    )
    assert leads
    assert leads[0].channel == "operation"
    assert leads[0].file_path == "fedot/core/operations/sklearn_transformations.py"
    trace_lead = next(lead for lead in leads if lead.channel == "trace")
    assert trace_lead.file_path == "fedot/core/data.py"
    assert trace_lead.line == 2
    assert "pca->catboost" not in trace_lead.why
    assert "IndexError" in trace_lead.why
    assert "pca->catboost" not in " ".join(leads[0].evidence)
    assert "FEDOT frame chain" in leads[0].evidence[1]
    assert "workload operation pca" in leads[0].evidence[2]
    assert "sklearn_transformations.py:130" in leads[0].evidence[2]
    assert "runtime data flow pca/fit" in leads[0].evidence[3]
    assert '"metadata_within_width": false' in leads[0].evidence[3]


def test_worker_data_snapshot_detects_metadata_outside_transformed_width():
    from types import SimpleNamespace

    import numpy as np

    from fedotllm.agents.evolve.evaluation._worker import _data_snapshot

    output = SimpleNamespace(
        features=np.zeros((5, 3)),
        predict=np.zeros((5, 1)),
        numerical_idx=np.array([0, 2]),
        categorical_idx=np.array([], dtype=int),
        encoded_idx=None,
    )
    snapshot = _data_snapshot(output, output=True)

    assert snapshot["active_width"] == 1
    assert snapshot["numerical_idx"]["max"] == 2
    assert snapshot["numerical_idx"]["within_width"] is False
    assert snapshot["metadata_within_width"] is False


def test_discover_prioritizes_crash_trace_over_coverage(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    trace_path = tmp_path / "fedot" / "core" / "trace_target.py"
    coverage_path = tmp_path / "fedot" / "core" / "coverage_target.py"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("def fit():\n    return 1\n", encoding="utf-8")
    coverage_path.write_text("def fit():\n    return 2\n", encoding="utf-8")
    trace_lead = MatchSite(
        "trace",
        "fedot/core/trace_target.py",
        1,
        "IndexError in fit",
        evidence=("stock runtime crash: IndexError",),
    )
    leads = discover_leads(
        tmp_path,
        limit=2,
        execution=[
            {
                "file_path": "fedot/core/coverage_target.py",
                "line": 1,
                "symbol": "fit",
                "count": 100,
            }
        ],
        trace_leads=[trace_lead],
    )

    assert leads[0].channel == "trace"
    assert leads[0].file_path == trace_lead.file_path
    assert leads[0].evidence == trace_lead.evidence


def test_failed_pytest_nodes_and_gate():
    from fedotllm.agents.evolve.discovery.discover import (
        failed_pytest_nodes,
        pytest_failure_excerpt,
    )
    from fedotllm.agents.evolve.evaluation.judge import tests_regressed

    text = "FAILED test/unit/data/test_data.py::test_idx - IndexError: x\n"
    nodes = failed_pytest_nodes(text)
    assert nodes == {"test/unit/data/test_data.py::test_idx"}
    assert tests_regressed(nodes, nodes) is None
    blocked = tests_regressed(set(), nodes)
    assert blocked is not None
    assert blocked.keep is False
    assert "fedot_tests_regressed" in blocked.reason
    assert "pca->catboost" not in blocked.reason

    detailed = (
        "============================= FAILURES =============================\n"
        "______________________________ test_idx ______________________________\n"
        "    assert actual == expected\n"
        "E   AssertionError: expected a fresh index\n"
        "====================== short test summary info ======================\n"
        "FAILED test/unit/data/test_data.py::test_idx - AssertionError: expected a fresh index\n"
        "99 warnings in 1.2s\n"
    )
    excerpt = pytest_failure_excerpt(detailed, nodes)
    assert "assert actual == expected" in excerpt
    assert "expected a fresh index" in excerpt
    assert "99 warnings" not in excerpt


def test_existing_failed_node_must_keep_the_same_failure_signature():
    from fedotllm.agents.evolve.discovery.discover import (
        failed_pytest_nodes,
        pytest_failure_excerpt,
    )
    from fedotllm.agents.evolve.evaluation.judge import tests_regressed
    from fedotllm.agents.evolve.types import TestResult

    node = "test/unit/data/test_data.py::test_idx"
    before = TestResult(
        "test_failures",
        1,
        {node},
        output=(
            "============================= FAILURES =============================\n"
            "______________________________ test_idx ______________________________\n"
            "../../.venv/lib/python3.11/site-packages/pandas/core/frame.py:12: in get\n"
            "  File \"/tmp/stock/fedot/a.py\", line 12\n"
            "E   AssertionError: expected metadata index\n"
            f"FAILED {node} - AssertionError: expected metadata index\n"
        ),
    )
    same_failure_other_checkout = TestResult(
        "test_failures",
        1,
        {node},
        output=(
            "============================= FAILURES =============================\n"
            "______________________________ test_idx ______________________________\n"
            "/Users/example/project/.venv/lib/python3.11/site-packages/pandas/core/frame.py:99: in get\n"
            "  File \"/tmp/work/experiments/run/candidate/fedot/a.py\", line 99\n"
            "E   AssertionError: expected metadata index\n"
            f"FAILED {node} - AssertionError: expected metadata index\n"
        ),
    )
    changed_failure = TestResult(
        "test_failures",
        1,
        {node},
        output=(
            "============================= FAILURES =============================\n"
            "______________________________ test_idx ______________________________\n"
            "E   TypeError: incompatible public contract\n"
            f"FAILED {node} - TypeError: incompatible public contract\n"
        ),
    )

    # A baseline-known node is allowed only if its underlying failure is still
    # the same. A different exception cannot hide behind the same node id.
    assert tests_regressed(before, same_failure_other_checkout) is None
    same_numeric_drift = TestResult(
        "test_failures",
        1,
        {node},
        output=(
            "============================= FAILURES =============================\n"
            "______________________________ test_idx ______________________________\n"
            "E   AssertionError: 0.847 vs 0.844 at 0x10abcdef\n"
            f"FAILED {node} - AssertionError: 0.847 vs 0.844 at 0x10abcdef\n"
        ),
    )
    same_numeric_drift_other_run = TestResult(
        "test_failures",
        1,
        {node},
        output=(
            "============================= FAILURES =============================\n"
            "______________________________ test_idx ______________________________\n"
            "E   AssertionError: 0.851 vs 0.849 at 0x7f3310aa\n"
            f"FAILED {node} - AssertionError: 0.851 vs 0.849 at 0x7f3310aa\n"
        ),
    )
    assert tests_regressed(same_numeric_drift, same_numeric_drift_other_run) is None
    blocked = tests_regressed(before, changed_failure)
    assert blocked is not None
    assert blocked.reason.startswith("fedot_tests_changed_failure")

    parametrized = (
        "________ test_value[first] ________\n"
        "E   AssertionError: first contract\n"
        "________ test_value[second] ________\n"
        "E   AssertionError: second contract\n"
        "================ warnings summary ================\n"
        "irrelevant dependency warning\n"
        "FAILED test/unit/test_value.py::test_value[first]\n"
        "FAILED test/unit/test_value.py::test_value[second]\n"
    )
    parameter_nodes = failed_pytest_nodes(parametrized)
    parameter_excerpt = pytest_failure_excerpt(parametrized, parameter_nodes)
    assert "first contract" in parameter_excerpt
    assert "second contract" in parameter_excerpt
    assert "irrelevant dependency warning" not in parameter_excerpt


def test_repo_map_ranks_symbol_and_field(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.repo_map import repo_map, search_callers, search_field_usage

    data = tmp_path / "fedot" / "core" / "data.py"
    pca = tmp_path / "fedot" / "core" / "operations" / "pca.py"
    data.parent.mkdir(parents=True)
    pca.parent.mkdir(parents=True)
    data.write_text(
        "class InputData:\n"
        "    def get_not_encoded_data(self):\n"
        "        return self.categorical_idx\n",
        encoding="utf-8",
    )
    pca.write_text(
        "def transform(data):\n"
        "    return data.get_not_encoded_data()\n",
        encoding="utf-8",
    )
    mapped = repo_map(tmp_path, ["get_not_encoded_data"])
    assert mapped
    assert mapped[0].name == "get_not_encoded_data"
    assert mapped[0].file_path == "fedot/core/data.py"
    callers = search_callers(tmp_path, "get_not_encoded_data")
    assert callers
    assert callers[0].file_path == "fedot/core/operations/pca.py"
    fields = search_field_usage(tmp_path, "categorical_idx")
    assert fields
    assert fields[0].file_path == "fedot/core/data.py"


def test_field_usage_prefers_writes_in_operations(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.repo_map import search_field_usage

    data = tmp_path / "fedot" / "core" / "data.py"
    pca = tmp_path / "fedot" / "core" / "operations" / "pca.py"
    data.parent.mkdir(parents=True)
    pca.parent.mkdir(parents=True)
    data.write_text(
        "def get_not_encoded_data(self):\n    return self.features[:, self.categorical_idx]\n",
        encoding="utf-8",
    )
    pca.write_text(
        "def transform(data):\n    data.categorical_idx = np.arange(3)\n",
        encoding="utf-8",
    )
    hits = search_field_usage(tmp_path, "categorical_idx")
    assert hits[0].file_path == "fedot/core/operations/pca.py"
    assert hits[0].kind == "field_write"


def test_repo_map_spreads_across_core_packages(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.repo_map import repo_map

    models = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "models"
        / "a.py"
    )
    dataops = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "z.py"
    )
    models.parent.mkdir(parents=True)
    dataops.parent.mkdir(parents=True)
    models.write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    dataops.write_text("class Z:\n    def h(self):\n        return 1\n", encoding="utf-8")
    mapped = repo_map(tmp_path, (), limit=2)
    blob = " ".join(item.file_path for item in mapped)
    assert "/models/" in blob
    assert "/data_operations/" in blob


def test_repo_map_prefers_fit_inside_package(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.repo_map import repo_map

    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text(
        "def helper():\n    return 1\n\nclass Knn:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    mapped = repo_map(tmp_path, (), limit=1)
    assert mapped[0].name == "fit"


def test_repo_map_skips_abstract_stubs(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.repo_map import repo_map

    iface = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "implementation_interfaces.py"
    )
    knn = tmp_path / "fedot" / "core" / "operations" / "evaluation" / "operation_implementations" / "models" / "knn.py"
    iface.parent.mkdir(parents=True)
    knn.parent.mkdir(parents=True)
    iface.write_text(
        "from abc import abstractmethod\n"
        "class Base:\n"
        "    @abstractmethod\n"
        "    def fit(self, data):\n"
        "        raise NotImplementedError\n",
        encoding="utf-8",
    )
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    mapped = repo_map(tmp_path, (), limit=8)
    assert any(item.name == "fit" and item.file_path.endswith("knn.py") for item in mapped)
    assert not any(item.name == "fit" and item.file_path.endswith("implementation_interfaces.py") for item in mapped)


def test_discover_walks_whole_core_not_one_field(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    data = tmp_path / "fedot" / "core" / "data.py"
    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    data.parent.mkdir(parents=True)
    knn.parent.mkdir(parents=True)
    data.write_text(
        "class InputData:\n"
        "    def get_not_encoded_data(self):\n"
        "        return self.features\n",
        encoding="utf-8",
    )
    knn.write_text(
        "class KnnImplementation:\n"
        "    def fit(self, data):\n"
        "        return data.get_not_encoded_data()\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=5)
    assert leads
    assert all(lead.channel == "repo_map" for lead in leads)
    paths = {lead.file_path for lead in leads}
    assert "fedot/core/operations/knn.py" in paths
    assert "fedot/core/data.py" in paths
    assert all(lead.file_path.startswith("fedot/core/") for lead in leads)


def test_discover_skips_cache_infra(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    cache = tmp_path / "fedot" / "core" / "caching" / "base_cache.py"
    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    cache.parent.mkdir(parents=True)
    knn.parent.mkdir(parents=True)
    cache.write_text(
        "def _is_expected_db_error(exc):\n    return True\n",
        encoding="utf-8",
    )
    knn.write_text(
        "class KnnImplementation:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=5)
    paths = {lead.file_path for lead in leads}
    assert "fedot/core/operations/knn.py" in paths
    assert not any("/caching/" in path for path in paths)


def test_scan_skips_only_non_training_folders(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.discovery.repo_map import in_metric_scan, iter_symbols

    knn = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "models"
        / "knn.py"
    )
    keep = {
        tmp_path / "fedot" / "core" / "pipelines" / "node.py": (
            "class PipelineNode:\n    def fit(self, data):\n        return data\n"
        ),
        tmp_path / "fedot" / "core" / "optimisers" / "objective" / "metrics_objective.py": (
            "def metric_value(x):\n    return x\n"
        ),
        tmp_path / "fedot" / "core" / "composer" / "composer.py": (
            "class Composer:\n    def compose(self, data):\n        return data\n"
        ),
        tmp_path / "fedot" / "preprocessing" / "preprocessing.py": (
            "class DataPreprocessor:\n    def fit(self, data):\n        return data\n"
        ),
    }
    skip = {
        tmp_path / "fedot" / "remote" / "remote_evaluator.py": "def run():\n    return 0\n",
        tmp_path / "fedot" / "explainability" / "explainers.py": "def explain(m):\n    return m\n",
        tmp_path / "fedot" / "core" / "visualisation" / "pipeline_specific_visuals.py": (
            "def plot_pipeline(g):\n    return g\n"
        ),
    }
    knn.parent.mkdir(parents=True)
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    for path, text in {**keep, **skip}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    knn_rel = knn.relative_to(tmp_path).as_posix()
    assert in_metric_scan(knn_rel)
    assert in_metric_scan("fedot/core/pipelines/node.py")
    assert in_metric_scan("fedot/core/optimisers/objective/metrics_objective.py")
    assert in_metric_scan("fedot/core/composer/composer.py")
    assert in_metric_scan("fedot/preprocessing/preprocessing.py")
    assert not in_metric_scan("fedot/remote/remote_evaluator.py")
    assert not in_metric_scan("fedot/explainability/explainers.py")
    files = {item.file_path for item in iter_symbols(tmp_path)}
    assert knn_rel in files
    assert "fedot/core/pipelines/node.py" in files
    assert "fedot/core/optimisers/objective/metrics_objective.py" in files
    assert not any("/remote/" in path for path in files)
    assert not any("explainability" in path for path in files)
    paths = {lead.file_path for lead in discover_leads(tmp_path, limit=8)}
    assert knn_rel in paths
    assert "fedot/remote/remote_evaluator.py" not in paths


def test_discover_skips_plots_keeps_quality_helpers(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    plots = tmp_path / "fedot" / "core" / "data" / "visualisation.py"
    enc = tmp_path / "fedot" / "core" / "operations" / "encode.py"
    plots.parent.mkdir(parents=True)
    enc.parent.mkdir(parents=True)
    plots.write_text(
        "def plot_pipeline(graph):\n    return graph\n",
        encoding="utf-8",
    )
    enc.write_text(
        "class Encoder:\n    def encode(self, data):\n        return data\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=5)
    paths = {lead.file_path for lead in leads}
    assert "fedot/core/operations/encode.py" in paths
    assert not any("visual" in path or "plot" in path for path in paths)


def test_discover_runtime_methods_before_helpers(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    data = tmp_path / "fedot" / "core" / "data" / "array_utilities.py"
    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    data.parent.mkdir(parents=True)
    knn.parent.mkdir(parents=True)
    data.write_text(
        "def find_common_elements(arrays):\n    return arrays\n",
        encoding="utf-8",
    )
    knn.write_text(
        "class Knn:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=1)
    assert leads[0].file_path == "fedot/core/operations/knn.py"
    assert leads[0].why.endswith("fit")


def test_run_once_scout_then_reproduce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.controller.campaign import run_once

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    stock = {"pca->catboost": _score("pca->catboost", "crash", 0.5)}
    order: list[str] = []

    def fake_tests(*_a, **_k):
        order.append("fedot_tests")
        return TestResult("passed", 0)

    def fake_scout(*_a, **kwargs):
        order.append("scout")
        assert "stock" not in kwargs
        return []

    def fake_stock(*_a, **_k):
        order.append("stock")
        return stock

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_fedot_tests", fake_tests)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.scout", fake_scout)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_stock", fake_stock)
    run_once(
        checkout=checkout,
        workspace=tmp_path,
        max_leads=1,
        inference=None,
        policy=FAST_RUN_POLICY,
    )
    assert set(order) == {"scout", "stock"}
    import json

    scout_row = json.loads((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert scout_row["logging_version"] == 2
    assert "pool_rows_static" in scout_row
    assert "llm_pick_raw" in scout_row
    assert "llm_pick_rounds" in scout_row
    assert "llm_picks" in scout_row


def test_run_once_refreshes_coverage_for_a_later_crashing_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.controller.campaign import run_once

    checkout = tmp_path / "fedot-src"
    leaf = checkout / "fedot" / "core" / "data.py"
    operation = checkout / "fedot" / "core" / "sklearn_transformations.py"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("def consume():\n    raise IndexError('oob')\n", encoding="utf-8")
    operation.write_text("class PCAImplementation:\n    pass\n", encoding="utf-8")
    traceback = (
        "Traceback (most recent call last):\n"
        f'  File "{leaf}", line 2, in consume\n'
        "    raise IndexError('oob')\n"
        "IndexError: oob\n"
    )
    without_coverage = ScoreResult(
        "pca->catboost", "crash", 0.5, traceback=traceback, detail="IndexError: oob"
    )
    with_coverage = ScoreResult(
        "pca->catboost",
        "crash",
        0.5,
        traceback=traceback,
        detail="IndexError: oob",
        coverage=(
            {
                "file_path": "fedot/core/sklearn_transformations.py",
                "symbol": "PCAImplementation",
                "line": 1,
                "count": 1,
            },
        ),
    )
    calls: list[tuple[str, ...]] = []

    def fake_stock(ids, **_kwargs):
        calls.append(tuple(ids))
        result = without_coverage if len(calls) == 1 else with_coverage
        return {"pca->catboost": result}

    def fake_scout(*_args, **kwargs):
        leads = kwargs["trace_leads"]
        assert leads[0].channel == "operation"
        assert leads[0].file_path.endswith("sklearn_transformations.py")
        return []

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_stock", fake_stock)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.scout", fake_scout)

    run_once(
        checkout=checkout,
        workspace=tmp_path / "work",
        inference=object(),
        lift_ids=("pca->catboost",),
        protect_ids=("pca->catboost",),
        max_leads=1,
        policy=FAST_RUN_POLICY,
    )

    assert calls == [("pca->catboost",), ("pca->catboost",)]


def test_run_once_drops_on_new_fedot_test_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    _accept_behavior_probe(monkeypatch)
    calls = {"n": 0}

    def fake_tests(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return TestResult("passed", 0)
        return TestResult(
            "test_failures",
            1,
            {"test/unit/new.py::test_x"},
            "FAILED test/unit/new.py::test_x",
        )

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_fedot_tests", fake_tests)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout",
        lambda *_a, **_k: [MatchSite(channel="lint", file_path="fedot/a.py", line=1, why="B006")],
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.fix_lead",
        lambda *_a, **_k: PatchCandidate("c1", "fedot/a.py", "old", "new"),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: {"pca->catboost": _score("pca->catboost", "crash", 0.5)},
    )
    patched_called = {"n": 0}

    def boom(*_a, **_k):
        patched_called["n"] += 1
        raise AssertionError("hidden exam must not run if FEDOT tests regressed")

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_patched", boom)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.snapshot_diff", lambda *_a, **_k: "")
    decision = run_once(
        checkout=checkout,
        workspace=tmp_path,
        max_leads=1,
        inference=object(),
        policy=FAST_RUN_POLICY,
    )
    assert decision.keep is False
    assert "fedot_tests_regressed" in decision.reason
    assert patched_called["n"] == 0


def test_run_once_drops_unimportable_before_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    _accept_behavior_probe(monkeypatch)
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "fedot" / "a.py").write_text("broken = definitely_not_defined\n", encoding="utf-8")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.scout",
        lambda *_a, **_k: [MatchSite(channel="repo_map", file_path="fedot/a.py", line=1, why="x")],
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.fix_lead",
        lambda *_a, **_k: PatchCandidate("c1", "fedot/a.py", "old", "new"),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    patched_called = {"n": 0}

    def boom(*_a, **_k):
        patched_called["n"] += 1
        raise AssertionError("holdout must not run if patch does not import")

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_patched", boom)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.snapshot_diff", lambda *_a, **_k: "diff")
    decision = run_once(
        checkout=checkout,
        workspace=tmp_path,
        max_leads=1,
        inference=object(),
        policy=FAST_RUN_POLICY,
    )
    assert decision.keep is False
    assert "patch_unimportable" in decision.reason
    assert patched_called["n"] == 0


def test_run_once_tries_second_lead_and_writes_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.storage.scoreboard import summarize
    from fedotllm.agents.evolve.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    _accept_behavior_probe(monkeypatch)
    leads = [
        MatchSite(channel="lint", file_path="fedot/a.py", line=1, why="B006"),
        MatchSite(channel="fedot_test", file_path="fedot/b.py", line=2, why="test/unit/x.py"),
    ]
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.hidden_exam",
        lambda: (("catboost", "lgbm", "rf"), ("catboost", "lgbm", "rf")),
    )
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.scout", lambda *a, **k: leads)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.snapshot_diff", lambda *a, **k: "--- a\n+++ b")
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *a, **k: TestResult("passed", 0),
    )

    stock = {
        "catboost": _score("catboost", "ok", 0.85),
        "lgbm": _score("lgbm", "ok", 0.80),
        "rf": _score("rf", "ok", 0.78),
    }
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.measure_stock", lambda *a, **k: stock)

    cands = [
        PatchCandidate("c1", "fedot/a.py", "old", "new"),
        PatchCandidate("c2", "fedot/b.py", "old2", "new2"),
    ]
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", lambda *a, **k: cands.pop(0))

    patched_runs = [
        {
            "catboost": _score("catboost", "ok", 0.85),
            "lgbm": _score("lgbm", "ok", 0.80),
            "rf": _score("rf", "ok", 0.78),
        },
        {
            "catboost": _score("catboost", "ok", 0.87),
            "lgbm": _score("lgbm", "ok", 0.80),
            "rf": _score("rf", "ok", 0.78),
        },
    ]
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *a, **k: patched_runs.pop(0),
    )

    decision = run_once(
        checkout=checkout,
        workspace=tmp_path,
        max_leads=2,
        max_revisions=1,
        inference=object(),
        lift_ids=("catboost", "lgbm", "rf"),
        protect_ids=("catboost", "lgbm", "rf"),
        policy=FAST_RUN_POLICY,
    )
    assert decision.keep is True
    summary = summarize(tmp_path)
    assert summary["attempts"] == 2
    assert summary["keeps"] == 1
    assert summary["getting_better"] is True
    journal = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert '"stock"' in journal
    assert '"patched"' in journal
    assert "catboost" in (tmp_path / "scoreboard.jsonl").read_text(encoding="utf-8")


def test_replay_reads_cmd_log_and_diff(tmp_path: Path):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import load_replay

    append_journal(
        tmp_path / "journal.jsonl",
        {
            "event": "decision",
            "candidate": "abc",
            "file": "fedot/core/foo.py",
            "keep": False,
            "reason": "no_lift",
            "diff": "--- a\n+++ b",
            "stock": {"t": {"cmd": "python -m worker", "log_tail": "boom", "status": "crash"}},
            "patched": {"t": {"cmd": "python -m worker", "log_tail": "still boom", "status": "crash"}},
        },
    )
    row = load_replay(tmp_path, candidate="abc")
    assert row is not None
    assert row["diff"] == "--- a\n+++ b"
    assert row["stock"]["t"]["cmd"] == "python -m worker"
    assert row["patched"]["t"]["log_tail"] == "still boom"


def test_discover_skips_pipeline_and_helpers_for_max_leads(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    knn = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "models"
        / "knn.py"
    )
    scale = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "scale.py"
    )
    node = tmp_path / "fedot" / "core" / "pipelines" / "node.py"
    helper = tmp_path / "fedot" / "core" / "data" / "array_utilities.py"
    for path in (knn, scale, node, helper):
        path.parent.mkdir(parents=True, exist_ok=True)
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    scale.write_text(
        "class Scale:\n    def transform(self, data):\n        return data\n",
        encoding="utf-8",
    )
    node.write_text(
        "class PipelineNode:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    helper.write_text("def find_common_elements(arrays):\n    return arrays\n", encoding="utf-8")
    leads = discover_leads(tmp_path, limit=2)
    paths = [lead.file_path for lead in leads]
    assert len(leads) == 2
    assert all("/operation_implementations/" in path for path in paths)
    assert not any("array_utilities" in path for path in paths)
    assert not any("/pipelines/" in path for path in paths)
    names = {_lead_why_name(lead) for lead in leads}
    assert names <= {"fit", "transform"}


def _lead_why_name(lead: MatchSite) -> str:
    return lead.why.strip().split()[-1].rsplit(".", 1)[-1]


def test_skip_tried_sites_from_scoreboard(tmp_path: Path):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import skip_tried

    append_journal(
        tmp_path / "scoreboard.jsonl",
        {
            "event": "attempt",
            "keep": False,
            "candidate_id": "applied1",
            "lead": {"channel": "repo_map", "file_path": "fedot/core/pipelines/node.py", "line": 185},
        },
    )
    leads = [
        MatchSite(
            channel="repo_map",
            file_path="fedot/core/pipelines/node.py",
            line=185,
            why="method PipelineNode.fit",
        ),
        MatchSite(
            channel="repo_map",
            file_path="fedot/core/operations/knn.py",
            line=53,
            why="method Knn.fit",
        ),
    ]
    kept = skip_tried(leads, tmp_path)
    assert [lead.file_path for lead in kept] == ["fedot/core/operations/knn.py"]


def test_skip_tried_skips_no_patch(tmp_path: Path):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import skip_tried

    append_journal(
        tmp_path / "scoreboard.jsonl",
        {
            "event": "attempt",
            "keep": False,
            "reason": "no_patch",
            "candidate_id": None,
            "lead": {"channel": "repo_map", "file_path": "fedot/core/operations/encoders.py", "line": 31},
        },
    )
    leads = [
        MatchSite(
            channel="repo_map",
            file_path="fedot/core/operations/encoders.py",
            line=31,
            why="method OneHot.fit",
        )
    ]
    assert skip_tried(leads, tmp_path) == []


def test_skip_tried_does_not_permanently_hide_cross_workspace_location(tmp_path: Path):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.storage.replay import skip_tried

    findings = tmp_path / "research" / "findings.jsonl"
    append_journal(
        findings,
        {
            "record_type": "finding",
            "source_hash": "frozen-a",
            "lead": {
                "file_path": "fedot/core/operations/encoders.py",
                "line": 31,
            },
        },
    )
    append_journal(
        findings,
        {
            "record_type": "finding",
            "source_hash": "other-source",
            "lead": {
                "file_path": "fedot/core/operations/knn.py",
                "line": 53,
            },
        },
    )
    leads = [
        MatchSite("repo_map", "fedot/core/operations/encoders.py", 31),
        MatchSite("repo_map", "fedot/core/operations/knn.py", 53),
    ]

    kept = skip_tried(
        leads,
        tmp_path / "new-workspace",
    )

    assert [(lead.file_path, lead.line) for lead in kept] == [
        ("fedot/core/operations/encoders.py", 31),
        ("fedot/core/operations/knn.py", 53),
    ]


def test_fit_tree_follows_both_branches_skips_plots(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.discovery.repo_map import reachable_from_fit

    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "encode.py"
    )
    plots = tmp_path / "fedot" / "core" / "data" / "visualisation.py"
    api = tmp_path / "fedot" / "api" / "main.py"
    impl.parent.mkdir(parents=True)
    plots.parent.mkdir(parents=True)
    api.parent.mkdir(parents=True)
    impl.write_text(
        "class Encoder:\n"
        "    def fit(self, data):\n"
        "        if data is None:\n"
        "            return self.encode(data)\n"
        "        return self.other(data)\n"
        "    def encode(self, data):\n"
        "        return data\n"
        "    def other(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    plots.write_text(
        "def plot_pipeline(graph):\n    return graph\n",
        encoding="utf-8",
    )
    api.write_text(
        "class Fedot:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    names = {item.name for item in reachable_from_fit(tmp_path)}
    assert "fit" in names
    assert "encode" in names
    assert "other" in names
    assert "plot_pipeline" not in names
    leads = discover_leads(tmp_path, limit=10)
    paths = {lead.file_path for lead in leads}
    assert impl.relative_to(tmp_path).as_posix() in paths
    assert not any("visual" in path for path in paths)


def test_discover_drops_timer_that_fit_calls(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "encode.py"
    )
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "class Encoder:\n"
        "    def fit(self, data):\n"
        "        self.set_seed()\n"
        "        self.tick_timer()\n"
        "        return self.encode(data)\n"
        "    def encode(self, data):\n"
        "        return data.features\n"
        "    def set_seed(self):\n"
        "        return 1\n"
        "    def tick_timer(self):\n"
        "        return 0\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=10)
    names = {lead.why.split()[-1].rsplit(".", 1)[-1] for lead in leads}
    assert "fit" in names
    assert "tick_timer" not in names
    assert impl.relative_to(tmp_path).as_posix() in {lead.file_path for lead in leads}


def test_neighborhood_keeps_same_class_not_in_call_graph(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "encode.py"
    )
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "class Encoder:\n"
        "    def fit(self, data):\n"
        "        return data\n"
        "    def reshape(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=10)
    names = {lead.why.split()[-1].rsplit(".", 1)[-1] for lead in leads}
    assert "fit" in names
    assert impl.relative_to(tmp_path).as_posix() in {lead.file_path for lead in leads}


def test_registry_maps_json_strategy_to_impl(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.discovery.registry import registry_files

    json_dir = tmp_path / "fedot" / "core" / "repository" / "data"
    strat = tmp_path / "fedot" / "core" / "operations" / "evaluation" / "boostings.py"
    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "models"
        / "boostings_implementations.py"
    )
    other = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "topological"
        / "fast_topological_extractor.py"
    )
    json_dir.mkdir(parents=True)
    strat.parent.mkdir(parents=True)
    impl.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    (json_dir / "model_repository.json").write_text(
        '{"metadata": {"boosting_class": {"strategies": ["fedot.core.operations.evaluation.boostings", "BoostingStrategy"]}}, "operations": {"catboost": {"meta": "boosting_class"}}}',
        encoding="utf-8",
    )
    strat.write_text(
        "from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import C\n"
        "class BoostingStrategy:\n"
        "    def fit(self, data):\n"
        "        return C().fit(data)\n",
        encoding="utf-8",
    )
    impl.write_text(
        "class C:\n    def fit(self, data):\n        return data\n    def predict(self, data):\n        return data\n",
        encoding="utf-8",
    )
    other.write_text(
        "class Topo:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    files = registry_files(tmp_path)
    assert "fedot/core/operations/evaluation/boostings.py" in files
    assert impl.relative_to(tmp_path).as_posix() in files
    assert other.relative_to(tmp_path).as_posix() not in files
    leads = discover_leads(tmp_path, limit=10)
    paths = {lead.file_path for lead in leads}
    assert impl.relative_to(tmp_path).as_posix() in paths
    assert other.relative_to(tmp_path).as_posix() in paths
    by_path = {lead.file_path: lead.channel for lead in leads}
    assert by_path[impl.relative_to(tmp_path).as_posix()] == "registry"
    assert by_path[other.relative_to(tmp_path).as_posix()] == "core_scan"


def test_discover_spreads_by_method_name(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    base = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
    )
    base.mkdir(parents=True)
    (base / "a.py").write_text(
        "class A:\n"
        "    def fit(self, data):\n"
        "        return self.encode(data)\n"
        "    def transform(self, data):\n"
        "        return data\n"
        "    def encode(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    (base / "b.py").write_text(
        "class B:\n"
        "    def fit(self, data):\n"
        "        return data\n"
        "    def transform(self, data):\n"
        "        return data\n",
        encoding="utf-8",
    )
    leads = discover_leads(tmp_path, limit=3)
    paths = {lead.file_path for lead in leads}
    assert len(paths) == 2
    assert {lead.why.split()[-1].rsplit(".", 1)[-1] for lead in leads} == {"fit"}


def test_context_includes_callee_in_other_file(tmp_path: Path):
    a = tmp_path / "fedot" / "core" / "operations" / "a.py"
    b = tmp_path / "fedot" / "core" / "operations" / "b.py"
    a.parent.mkdir(parents=True)
    a.write_text("def fit(data):\n    return helper(data)\n", encoding="utf-8")
    b.write_text("def helper(data):\n    return data.features\n", encoding="utf-8")
    ctx = context_from_lead(
        MatchSite(channel="repo_map", file_path="fedot/core/operations/a.py", line=1, why="function fit"),
        tmp_path,
    )
    assert "Called from this function" in ctx
    assert "def helper" in ctx


def test_invariant_flags_stale_column_metadata(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.discovery.invariants import invariant_leads

    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "proj.py"
    )
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "class Proj:\n"
        "    def transform(self, data):\n"
        "        width = max(data.col_idx) if data.col_idx else 0\n"
        "        data.features = data.features[:, :1]\n"
        "        return data\n"
        "    def fit(self, data):\n"
        "        data.col_idx = [0]\n"
        "        data.features = data.features\n"
        "        return data\n",
        encoding="utf-8",
    )
    hints = invariant_leads(tmp_path)
    assert hints
    assert all(lead.channel == "invariant" for lead in hints)
    assert all("leftover" not in lead.why.lower() for lead in hints)
    names = {lead.why.rsplit(" in ", 1)[-1] for lead in hints}
    assert any(name.endswith("transform") for name in names)
    assert not any(name.endswith("fit") for name in names)
    trace: dict = {}
    leads = discover_leads(tmp_path, limit=5, trace=trace)
    assert all(lead.channel != "invariant" for lead in leads)
    assert any(hit["why"].endswith("transform") for hit in trace["metadata_stale_hits"])
    blob = " ".join(lead.why for lead in leads).lower()
    assert "leftover" not in blob


def test_row_identity_invariant_ranks_unaligned_parent_join(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import static_leads
    from fedotllm.agents.evolve.discovery.invariants import row_identity_leads

    impl = tmp_path / "fedot" / "core" / "data" / "join.py"
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "import numpy as np\n"
        "class Joiner:\n"
        "    def select(self, idx, values):\n"
        "        mask = np.isin(idx, self.common)\n"
        "        return values[mask]\n"
        "    def collect(self):\n"
        "        return [self.select(parent.idx, parent.values) "
        "for parent in self.parents]\n"
        "    def merge(self, values):\n"
        "        return np.concatenate(values, axis=-1)\n",
        encoding="utf-8",
    )

    hints = row_identity_leads(tmp_path)
    assert len(hints) == 1
    assert hints[0].line == 6
    assert "canonical index order" in hints[0].why
    assert hints[0].signals == ("data_plane", "row_identity_contract")
    assert static_leads(tmp_path)[0].file_path == "fedot/core/data/join.py"


def test_row_identity_invariant_accepts_explicit_position_map(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.invariants import row_identity_leads

    impl = tmp_path / "fedot" / "core" / "data" / "join.py"
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "import numpy as np\n"
        "class Joiner:\n"
        "    def select(self, idx, values):\n"
        "        mask = np.isin(idx, self.common)\n"
        "        return values[mask]\n"
        "    def collect(self):\n"
        "        rows = {key: pos for pos, key in enumerate(self.common)}\n"
        "        return [self.select(parent.idx, parent.values) "
        "for parent in self.parents]\n"
        "    def merge(self, values):\n"
        "        return np.concatenate(values, axis=-1)\n",
        encoding="utf-8",
    )

    assert row_identity_leads(tmp_path) == []


def test_discover_keeps_private_helper_on_fit_path(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import discover_leads

    impl = (
        tmp_path
        / "fedot"
        / "core"
        / "operations"
        / "evaluation"
        / "operation_implementations"
        / "data_operations"
        / "proj.py"
    )
    impl.parent.mkdir(parents=True)
    impl.write_text(
        "class Proj:\n"
        "    def transform(self, data):\n"
        "        return self._prepare_features(data)\n"
        "    def _prepare_features(self, data):\n"
        "        data.features = data.features * 2\n"
        "        return data\n",
        encoding="utf-8",
    )
    lead = next(item for item in discover_leads(tmp_path, limit=10) if item.why.endswith("transform"))
    ctx = context_from_lead(lead, tmp_path)
    assert "def transform" in ctx
    assert "def _prepare_features" in ctx


def test_apply_patch_strips_line_gutter_and_rejects_noop(tmp_path: Path):
    from fedotllm.agents.evolve.execution.patch import apply_patch, same_runtime

    target = tmp_path / "fedot" / "foo.py"
    target.parent.mkdir(parents=True)
    target.write_text("def f(x):\n    return x\n", encoding="utf-8")
    ok = apply_patch(
        tmp_path,
        PatchCandidate(
            candidate_id="gutter",
            file_path="fedot/foo.py",
            old_code="     2|    return x\n",
            new_code="     2|    return x + 1\n",
        ),
    )
    assert ok is True
    assert "return x + 1" in target.read_text(encoding="utf-8")
    assert same_runtime("    return x  # a\n", "    return x  # b\n")
    blocked = apply_patch(
        tmp_path,
        PatchCandidate(
            candidate_id="noop",
            file_path="fedot/foo.py",
            old_code="    return x + 1\n",
            new_code="    return x + 1  # same\n",
        ),
    )
    assert blocked is False


def test_propose_rejects_noop_and_strips_gutter():
    from fedotllm.agents.evolve.agents.propose import PatchProposal, propose_patch

    class _Inf:
        def __init__(self, proposal: PatchProposal):
            self.proposal = proposal

        def create(self, _prompt, _model):
            return self.proposal

    noop = propose_patch(
        inference=_Inf(PatchProposal(file_path="fedot/core/x.py", old_code="return x", new_code="return x")),
        context="# fedot/core/x.py\n    return x",
    )
    assert noop is None
    cand = propose_patch(
        inference=_Inf(
            PatchProposal(
                file_path="fedot/core/x.py",
                old_code="     2|    return x\n",
                new_code="     2|    return x + 1\n",
            )
        ),
        context="# fedot/core/x.py\n    return x",
    )
    assert cand is not None
    assert cand.old_code == "    return x"
    assert cand.new_code == "    return x + 1"
    assert "|" not in cand.old_code


def test_unique_keeps_llm_pick_for_pool_rows():
    from fedotllm.agents.evolve.discovery.discover import _unique, pool_rows

    structural = MatchSite(
        channel="repo_map",
        file_path="fedot/core/operations/impute.py",
        line=10,
        why="method Imputer.fit",
        signals=("reachable",),
    )
    picked = MatchSite(
        channel="llm",
        file_path="fedot/core/operations/impute.py",
        line=10,
        why="imputation affects fit/predict quality",
    )
    merged = _unique([picked, structural])
    assert merged[0].channel == "llm"
    assert "imputation affects" in merged[0].why
    assert merged[0].signals == ("reachable",)
    assert pool_rows(merged)[0]["llm_picked"] is True
    reversed_merge = _unique([structural, picked])
    assert reversed_merge[0].channel == "llm"
    assert "imputation affects" in reversed_merge[0].why


def test_calibrate_stock_summarizes_ok_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.commands.calibrate import calibrate_stock

    scores = {1: 0.80, 2: 0.82, 3: 0.81}

    def fake_stock(task_id, *, checkout, seed, timeout_s=None):
        return ScoreResult(task_id=task_id, status="ok", score=scores[seed])

    monkeypatch.setattr("fedotllm.agents.evolve.commands.calibrate.run_stock", fake_stock)
    payload = calibrate_stock(
        checkout=tmp_path,
        task_ids=("catboost",),
        seeds=(1, 2, 3),
        workspace=tmp_path,
    )
    stats = payload["by_task"]["catboost"]
    assert stats["n_ok"] == 3
    assert stats["range"] == pytest.approx(0.02)
    assert stats["mean"] == pytest.approx(0.81)
    assert (tmp_path / "noise.jsonl").is_file()


def test_discover_trace_keeps_static_pool_and_raw_llm_pick(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")

    class _Inf:
        def create(self, prompt, _schema):
            assert "pca->catboost" not in prompt
            assert "Pick one site that can change model quality" not in prompt
            assert "operations, models, preprocessing, data handling, defaults" not in prompt
            assert "status=skip" in prompt
            assert "status=read" in prompt
            assert "status=run" in prompt
            return SiteProposal(
                file_path="fedot/core/operations/knn.py",
                line=2,
                why="scale features before knn",
                **_causal_fields(2),
            )

    trace: dict = {}
    leads = discover_leads(tmp_path, inference=_Inf(), limit=5, trace=trace)
    assert trace["pool_rows_static"]
    assert "scale features before knn" not in trace["pool_rows_static"][0]["symbol"]
    assert trace["llm_pick"]["file_path"].endswith("knn.py")
    assert trace["llm_pick"]["why"] == "scale features before knn"
    assert leads[0].why == "scale features before knn"
    assert leads[0].channel == "llm"
    assert trace["llm_pick_raw"] is None
    from fedotllm.agents.evolve.discovery.discover import localization

    loc = localization(
        leads[0].file_path,
        leads[0].line,
        static_rows=trace["pool_rows_static"],
        final_rows=[{"rank": i, "file_path": x.file_path, "line": x.line} for i, x in enumerate(leads, start=1)],
        llm_pick=trace["llm_pick"],
    )
    assert loc["picked_by_llm"] is True
    assert loc["final_rank"] == 1


def test_localization_rank_delta_and_raw_pick():
    from fedotllm.agents.evolve.discovery.discover import localization

    static = [
        {"rank": 1, "file_path": "fedot/a.py", "line": 1},
        {"rank": 2, "file_path": "fedot/b.py", "line": 2},
    ]
    final = [
        {"rank": 1, "file_path": "fedot/b.py", "line": 2},
        {"rank": 2, "file_path": "fedot/a.py", "line": 1},
    ]
    pick = {"file_path": "fedot/b.py", "line": 2, "why": "quality"}
    up = localization("fedot/b.py", 2, static_rows=static, final_rows=final, llm_pick=pick)
    assert up == {"static_rank": 2, "final_rank": 1, "rank_delta": 1, "picked_by_llm": True}
    down = localization("fedot/a.py", 1, static_rows=static, final_rows=final, llm_pick=pick)
    assert down["rank_delta"] == -1
    assert down["picked_by_llm"] is False


def test_discover_captures_llm_pick_raw(tmp_path: Path):
    import json

    from fedotllm.agents.evolve.discovery.discover import discover_leads

    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    raw = json.dumps({
        "file_path": "fedot/core/operations/knn.py",
        "line": 2,
        "why": "scale before knn extra note",
        **_causal_fields(2),
    })

    class _Inf:
        def query(self, _messages, *args, **kwargs):
            return raw

        def create(self, prompt, schema):
            return schema.model_validate_json(self.query(prompt))

    trace: dict = {}
    discover_leads(tmp_path, inference=_Inf(), limit=3, trace=trace)
    assert trace["llm_pick_raw"] == raw
    assert "extra note" in trace["llm_pick"]["why"]


def test_evolve_llm_audit_records_exact_prompt_raw_and_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json

    from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create
    from fedotllm.agents.evolve.discovery.discover import SiteProposal

    audit = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("EVOLVE_AGENT_LLM_AUDIT", str(audit))

    class _Config:
        provider = "openrouter"
        model_name = "z-ai/glm-5.3-flash"

    class _Inf:
        config = _Config()

        def __init__(self):
            self.usage = {"requests": 0, "prompt_tokens": 0}

        def query(self, messages, *args, **kwargs):
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += 7
            return '{"status":"skip","why":"raw answer without truncation"}'

        def create(self, prompt, schema):
            exact = prompt + "\n<STRUCTURED SCHEMA>"
            return schema.model_validate_json(self.query(exact))

    parsed, raw = capture_structured_create(
        _Inf(),
        "GENERAL SEARCH OBJECTIVE",
        SiteProposal,
        stage="scout",
        metadata={"catalog_file": "fedot/a.py"},
    )
    assert parsed.status == "skip"
    assert raw == '{"status":"skip","why":"raw answer without truncation"}'
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == [
        "llm_structured_started",
        "llm_query_started",
        "llm_query",
        "llm_structured_result",
    ]
    query = rows[2]
    result = rows[3]
    assert query["messages"] == "GENERAL SEARCH OBJECTIVE\n<STRUCTURED SCHEMA>"
    assert query["response"] == raw
    assert query["usage_delta"] == {"prompt_tokens": 7, "requests": 1}
    assert query["model"] == "openrouter/z-ai/glm-5.3-flash"
    assert result["parsed"]["status"] == "skip"
    assert result["metadata"]["catalog_file"] == "fedot/a.py"


def test_fixer_prompt_does_not_leak_known_pca_case():
    from fedotllm.agents.evolve.agents.propose import build_prompt

    prompt = build_prompt("# arbitrary FEDOT source", max_edits=4).lower()
    assert "pcaimplementation" not in prompt
    assert "metadata_within_width" not in prompt
    assert "stale column indexes" not in prompt
    assert "numeric-projection" not in prompt


def test_llm_pick_rejects_file_outside_shown_context(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot" / "core" / "operations"
    root.mkdir(parents=True)
    (root / "aa.py").write_text(
        "class Impl:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )

    class _Inf:
        def create(self, prompt, _schema):
            assert "# fedot/core/operations/aa.py" in prompt
            return SiteProposal(
                file_path="fedot/core/operations/ae.py",
                line=2,
                why="outside shown files",
            )

    trace: dict = {}
    leads = discover_leads(tmp_path, inference=_Inf(), limit=8, trace=trace)
    assert trace["llm_pick"] is None
    assert all(lead.channel != "llm" for lead in leads)


def test_context_includes_sibling_runtime_methods(tmp_path: Path):
    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text(
        "class Knn:\n"
        "    def fit(self, data):\n"
        "        return data\n"
        "    def predict(self, data):\n"
        "        return data.features\n",
        encoding="utf-8",
    )
    ctx = context_from_lead(
        MatchSite(channel="repo_map", file_path="fedot/core/operations/knn.py", line=2, why="method Knn.fit"),
        tmp_path,
    )
    assert "def fit" in ctx
    assert "def predict" in ctx


def test_context_from_lead_sends_whole_file(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "operations" / "pca.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def helper():\n    return 0\n\nclass Pca:\n    def transform(self, data):\n        return data\n",
        encoding="utf-8",
    )
    ctx = context_from_lead(
        MatchSite(channel="repo_map", file_path="fedot/core/operations/pca.py", line=5, why="method Pca.transform"),
        tmp_path,
    )
    assert "def helper" in ctx
    assert "def transform" in ctx
    assert "Read this whole file" in ctx


def test_context_slices_long_file(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "operations" / "wide.py"
    src.parent.mkdir(parents=True)
    body = "x = 1\n" * 450 + "class P:\n    def transform(self, data):\n        return data\n"
    src.write_text(body, encoding="utf-8")
    line = body.splitlines().index("    def transform(self, data):") + 1
    ctx = context_from_lead(
        MatchSite(channel="repo_map", file_path="fedot/core/operations/wide.py", line=line, why="method P.transform"),
        tmp_path,
    )
    assert "def transform" in ctx
    assert "File is long" in ctx
    assert ctx.count("x = 1") < 50


def test_bounded_context_prioritizes_measured_symbol_over_repeated_runtime_rows(
    tmp_path: Path,
):
    src = tmp_path / "fedot" / "core" / "operations" / "medium.py"
    src.parent.mkdir(parents=True)
    padding = "\n".join(f"PADDING_{index} = {index}" for index in range(205))
    src.write_text(
        padding
        + "\n\ndef metric_path(data):\n"
        + "    important_contract = data.features\n"
        + "    return important_contract\n",
        encoding="utf-8",
    )
    lead_line = src.read_text(encoding="utf-8").splitlines().index(
        "def metric_path(data):"
    ) + 1
    repeated = tuple(
        f'runtime operation instances: model_{index}/Wrapper fit params={{"very_long": "{("x" * 500)}"}} | '
        f'model_{index}/Wrapper predict params={{"very_long": "{("y" * 500)}"}}'
        for index in range(12)
    )
    ctx = context_from_lead(
        MatchSite(
            channel="execution",
            file_path="fedot/core/operations/medium.py",
            line=lead_line,
            why="executed symbol metric_path",
            evidence=(
                "runtime line hits: 9",
                f"executed lines in this symbol: {lead_line}-{lead_line + 2}",
                *repeated,
            ),
        ),
        tmp_path,
        max_chars=8_000,
    )

    assert "def metric_path(data):" in ctx
    assert "important_contract = data.features" in ctx
    assert "runtime implementations observed:" in ctx
    assert '"very_long"' not in ctx
    assert len(ctx) <= 8_000


def test_run_once_skips_replayed_lead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.types import Decision as Dec
    from fedotllm.agents.evolve.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    append_journal(
        tmp_path / "scoreboard.jsonl",
        {
            "event": "attempt",
            "keep": False,
            "candidate_id": "already",
            "lead": {"channel": "repo_map", "file_path": "fedot/a.py", "line": 1},
        },
    )
    leads = [
        MatchSite(channel="repo_map", file_path="fedot/a.py", line=1, why="method A.fit"),
        MatchSite(channel="repo_map", file_path="fedot/b.py", line=2, why="method B.fit"),
    ]
    seen: list[str] = []

    def fake_fix(_checkout, lead, **_k):
        seen.append(lead.file_path)
        return PatchCandidate("c1", lead.file_path, "old", "new")

    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.scout", lambda *_a, **_k: leads)
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.fix_lead", fake_fix)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_stock",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_fedot_tests",
        lambda *_a, **_k: TestResult("passed", 0),
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.measure_patched",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.controller.campaign.verdict",
        lambda *_a, **_k: Dec(keep=False, reason="target_delta 0.0000 < 0.01", target_delta=0.0, regression_deltas={}),
    )
    monkeypatch.setattr("fedotllm.agents.evolve.controller.campaign.snapshot_diff", lambda *_a, **_k: "diff")
    run_once(
        checkout=checkout,
        workspace=tmp_path,
        max_leads=2,
        max_revisions=1,
        inference=object(),
        policy=FAST_RUN_POLICY,
    )
    assert seen == ["fedot/b.py"]


def test_recall_metrics_and_split_disjoint():
    from fedotllm.agents.evolve.commands.recall import metrics, split_gold, unique_files
    from fedotllm.agents.evolve.types import MatchSite

    ranks = [1, 4, None]
    row = metrics(ranks)
    assert row["recall@1"] == pytest.approx(1 / 3)
    assert row["recall@3"] == pytest.approx(1 / 3)
    assert row["recall@5"] == pytest.approx(2 / 3)
    assert row["mrr"] == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    parts = split_gold()
    assert parts["dev"] and parts["test"]
    assert set(parts["dev"]).isdisjoint(parts["test"])
    leads = [
        MatchSite(channel="repo_map", file_path="fedot/a.py", line=1),
        MatchSite(channel="repo_map", file_path="fedot/a.py", line=2),
        MatchSite(channel="repo_map", file_path="fedot/b.py", line=1),
    ]
    assert unique_files(leads) == ["fedot/a.py", "fedot/b.py"]


def test_recall_skips_llm_variant_without_inference(tmp_path: Path):
    from fedotllm.agents.evolve.commands.recall import measure_localization

    (tmp_path / "fedot" / "core" / "operations").mkdir(parents=True)
    (tmp_path / "fedot" / "core" / "operations" / "model.py").write_text(
        "class Operation:\n    def fit(self):\n        pass\n",
        encoding="utf-8",
    )
    payload = measure_localization(tmp_path, inference=None)
    assert "structural_invariant_llm" not in payload["by_split"]["dev"]
    assert payload["llm_pick"] is None


def test_oracle_lead_uses_runtime_line(tmp_path: Path):
    from fedotllm.agents.evolve.commands.repair import oracle_lead

    path = tmp_path / "fedot" / "core" / "data.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "def helper():\n    return 1\n\nclass Data:\n    def fit(self, x):\n        return x\n",
        encoding="utf-8",
    )
    lead = oracle_lead(tmp_path, "fedot/core/data.py")
    assert lead is not None
    assert lead.channel == "oracle"
    assert lead.line >= 4
    assert "fit" in lead.why


def test_measure_repair_no_inference_is_no_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.commands.repair import measure_repair

    rel = "fedot/api/time.py"
    src = tmp_path / "stock"
    (src / "fedot" / "api").mkdir(parents=True)
    (src / "fedot" / "api" / "time.py").write_text(
        "def determine_operation_timeout():\n    return 1\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    (work / "fedot" / "api").mkdir(parents=True)
    (work / "fedot" / "api" / "time.py").write_text(
        "def determine_operation_timeout():\n    return 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.resolve_fedot_src", lambda: src)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.split_gold",
        lambda: {"dev": [rel], "test": []},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.static_leads",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.unique_files",
        lambda *_a, **_k: [],
    )
    payload = measure_repair(inference=None, checkout=work, workspace=tmp_path / "ws", limit=1)
    assert payload["n"] == 1
    assert payload["patch_rate"] == 0
    assert payload["rows"][0]["status"] == "no_patch"


def test_measure_repair_files_override_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.commands.repair import measure_repair

    rel = "fedot/core/operations/knn.py"
    src = tmp_path / "stock"
    (src / "fedot" / "core" / "operations").mkdir(parents=True)
    (src / "fedot" / "core" / "operations" / "knn.py").write_text(
        "class K:\n    def fit(self):\n        return 1\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    (work / "fedot" / "core" / "operations").mkdir(parents=True)
    (work / "fedot" / "core" / "operations" / "knn.py").write_text(
        "class K:\n    def fit(self):\n        return 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.resolve_fedot_src", lambda: src)
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.split_gold", lambda: {"dev": ["fedot/api/time.py"], "test": []})
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.static_leads", lambda *_a, **_k: [])
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.unique_files", lambda *_a, **_k: [])
    payload = measure_repair(
        inference=None,
        checkout=work,
        workspace=tmp_path / "ws",
        files=(rel,),
    )
    assert payload["n"] == 1
    assert payload["rows"][0]["file_path"] == rel


def test_gate_saved_repairs_flags_new_pytest_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.commands.repair import gate_saved_repairs

    workspace = tmp_path / "ws"
    cand = workspace / "candidates" / "abc"
    cand.mkdir(parents=True)
    (cand / "old.py").write_text("x = 1\n", encoding="utf-8")
    (cand / "new.py").write_text("x = 2\n", encoding="utf-8")
    journal = workspace / "repair.jsonl"
    journal.write_text(
        '{"event":"oracle_attempt","file_path":"fedot/a.py","candidate_id":"abc"}\n',
        encoding="utf-8",
    )
    checkout = tmp_path / "fedot-src"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot" / "a.py").write_text("x = 1\n", encoding="utf-8")
    stock = tmp_path / "stock"
    (stock / "fedot").mkdir(parents=True)
    (stock / "fedot" / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.resolve_fedot_src", lambda: stock)
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.apply_patch", lambda *_a, **_k: True)
    calls = {"n": 0}

    def fake_tests(_checkout):
        calls["n"] += 1
        if calls["n"] == 1:
            return TestResult("passed", 0)
        return TestResult(
            "test_failures",
            1,
            {"test/unit/test_a.py::test_x"},
        )

    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.measure_fedot_tests", fake_tests)
    payload = gate_saved_repairs(workspace=workspace, checkout=checkout)
    assert payload["tests_ok_rate"] == 0
    assert payload["rows"][0]["status"] == "tests_fail"
    assert "fedot_tests_regressed" in (payload["rows"][0]["reason"] or "")


def test_holdout_saved_repairs_records_drop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.commands.repair import holdout_saved_repairs
    from fedotllm.agents.evolve.types import Decision as Dec

    workspace = tmp_path / "ws"
    cand = workspace / "candidates" / "abc"
    cand.mkdir(parents=True)
    (cand / "old.py").write_text("x = 1\n", encoding="utf-8")
    (cand / "new.py").write_text("x = 2\n", encoding="utf-8")
    (workspace / "repair.jsonl").write_text(
        '{"event":"oracle_attempt","file_path":"fedot/a.py","candidate_id":"abc"}\n',
        encoding="utf-8",
    )
    checkout = tmp_path / "fedot-src"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot" / "a.py").write_text("x = 1\n", encoding="utf-8")
    stock = tmp_path / "stock"
    (stock / "fedot").mkdir(parents=True)
    (stock / "fedot" / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.resolve_fedot_src", lambda: stock)
    monkeypatch.setattr("fedotllm.agents.evolve.commands.repair.apply_patch", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.measure_stock",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.measure_patched",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr(
        "fedotllm.agents.evolve.commands.repair.verdict",
        lambda *_a, **_k: Dec(keep=False, reason="target_delta 0.0000 < 0.01", target_delta=0.0, regression_deltas={}),
    )
    payload = holdout_saved_repairs(workspace=workspace, checkout=checkout)
    assert payload["keep_rate"] == 0
    assert payload["rows"][0]["status"] == "drop"


def test_run_fedot_snippet_blocks_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.execution import run_code as rc

    out = rc.run_fedot_snippet(tmp_path, "from fedotllm.agents.evolve.evaluation.scorer import score")
    assert out.status == "blocked"
    assert out.detail.startswith("<blocked")


def test_run_fedot_snippet_blocks_importlib_harness(tmp_path: Path):
    from fedotllm.agents.evolve.execution import run_code as rc

    code = (
        "import importlib\n"
        "importlib.import_module('fedotllm.agents.evolve.evaluation.scorer')"
    )
    out = rc.run_fedot_snippet(tmp_path, code)
    assert out.status == "blocked"


def test_run_fedot_snippet_redacts_secret_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys

    from fedotllm.agents.evolve.execution import run_code as rc

    monkeypatch.setattr(
        rc,
        "fedot_python",
        lambda _checkout: sys.executable,
    )
    out = rc.run_fedot_snippet(tmp_path, 'print("token=supersecret123")')
    if out.status == "ok":
        assert "supersecret123" not in out.stdout
        assert "REDACTED" in out.stdout


def test_llm_pick_runs_then_picks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text("class Knn:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    calls = {"n": 0}

    class _Inf:
        def create(self, prompt, _schema):
            calls["n"] += 1
            if calls["n"] == 1:
                assert "status=run" in prompt
                return SiteProposal(status="run", run_code="print(1)")
            assert "Output:" in prompt
            assert "ran-ok" in prompt
            return SiteProposal(
                status="pick",
                file_path="fedot/core/operations/knn.py",
                line=2,
                why="checked at runtime",
                **_causal_fields(2),
            )

    def _independent_probe(_checkout, _code, history=None):
        assert history is None
        return SnippetResult(status="ok", code=_code, stdout="ran-ok")

    monkeypatch.setattr(
        "fedotllm.agents.evolve.execution.run_code.run_fedot_snippet",
        _independent_probe,
    )
    trace: dict = {}
    leads = discover_leads(tmp_path, inference=_Inf(), limit=3, trace=trace)
    assert calls["n"] == 2
    assert trace["llm_pick"]["why"] == "checked at runtime"
    assert trace["llm_pick_rounds"][0]["status"] == "run"
    assert leads[0].channel == "llm"


def test_llm_pick_has_global_action_and_per_file_run_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot" / "core" / "operations"
    root.mkdir(parents=True)
    (root / "aa.py").write_text(
        "class A:\n    def fit(self, data):\n        return data\n", encoding="utf-8"
    )
    (root / "bb.py").write_text(
        "class B:\n    def fit(self, data):\n        return data\n", encoding="utf-8"
    )
    calls = {"llm": 0, "run": 0}

    class _Inf:
        def create(self, _prompt, _schema):
            calls["llm"] += 1
            return SiteProposal(status="run", run_code="print(1)")

    def _run(*_args, **_kwargs):
        calls["run"] += 1
        return SnippetResult(status="ok", code="print(1)", stdout="ok")

    monkeypatch.setattr("fedotllm.agents.evolve.execution.run_code.run_fedot_snippet", _run)
    trace: dict = {}
    discover_leads(
        tmp_path,
        inference=_Inf(),
        limit=8,
        max_picks=1,
        max_actions=5,
        max_runs_per_file=2,
        trace=trace,
    )

    assert calls == {"llm": 5, "run": 4}
    assert trace["scout_actions"] == 5
    assert trace["scout_action_limit"] == 5
    assert trace["scout_budget_exhausted"] is True
    assert sum(row["status"] == "run_limit" for row in trace["llm_pick_rounds"]) == 1


def test_llm_pick_reads_neighbor_then_picks(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot" / "core" / "operations"
    root.mkdir(parents=True)
    (root / "knn.py").write_text(
        "class Knn:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    (root / "helper.py").write_text(
        "def scale(x):\n    return x * 2\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    class _Inf:
        def create(self, prompt, _schema):
            calls["n"] += 1
            if "Opened fedot/core/operations/helper.py" in prompt:
                assert "def scale" in prompt
                return SiteProposal(
                    status="pick",
                    file_path="fedot/core/operations/helper.py",
                    line=1,
                    why="scale changes features",
                    **_causal_fields(1),
                )
            if "helper.py" in prompt or "knn.py" in prompt:
                return SiteProposal(
                    status="read",
                    file_path="fedot/core/operations/helper.py",
                    line=1,
                )
            return SiteProposal(status="skip")

    trace: dict = {}
    leads = discover_leads(tmp_path, inference=_Inf(), limit=3, max_picks=1, trace=trace)
    assert calls["n"] == 2
    assert trace["llm_pick"]["file_path"].endswith("helper.py")
    assert trace["llm_pick_rounds"][0]["status"] == "read"
    assert leads[0].channel == "llm"


@pytest.mark.parametrize("action_budget,expected_picks", [(1, 0), (2, 1)])
def test_unread_pick_target_requires_source_review_and_reconfirmation(tmp_path, action_budget, expected_picks):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot/core/operations"
    root.mkdir(parents=True)
    (root / "aa.py").write_text("class A:\n    def fit(self, data):\n        return data\n")
    (root / "bb.py").write_text("class B:\n    def fit(self, data):\n        return data * 2\n")
    prompts = []

    class Inference:
        def create(self, prompt, _schema):
            prompts.append(prompt)
            return SiteProposal(status="pick", file_path="fedot/core/operations/bb.py",
                                line=2, why="inspect another implementation", **_causal_fields(2))

    trace = {}
    leads = discover_leads(tmp_path, inference=Inference(), limit=1, max_picks=1,
                           max_actions=action_budget, trace=trace)
    assert len(leads) == expected_picks
    if expected_picks:
        assert len(prompts) == 2
        assert "has NOT been accepted" in prompts[1]
        assert "return data * 2" in prompts[1]
        assert _causal_fields(2)["proposed_change"] in prompts[1]
        assert trace["llm_pick_rounds"][0]["status"] == "pick_target_read"
    else:
        assert len(prompts) == 1


def test_llm_pick_continues_after_first_pick(tmp_path: Path):
    from fedotllm.agents.evolve.discovery.discover import SiteProposal, discover_leads

    root = tmp_path / "fedot" / "core" / "operations"
    root.mkdir(parents=True)
    (root / "aa.py").write_text("class A:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    (root / "bb.py").write_text("class B:\n    def fit(self, data):\n        return data\n", encoding="utf-8")
    catalog: list[str] = []
    prompts: list[str] = []

    class _Inf:
        def create(self, prompt, _schema):
            prompts.append(prompt)
            for line in prompt.splitlines():
                if line.startswith("Catalog site:"):
                    path = line.split()[2].split(":")[0]
                    catalog.append(path)
                    return SiteProposal(
                        status="pick",
                        file_path=path,
                        line=2,
                        why="site",
                        **_causal_fields(2),
                    )
            return SiteProposal(status="skip")

    trace: dict = {}
    discover_leads(tmp_path, inference=_Inf(), limit=8, max_picks=2, trace=trace,
                   prior_hypotheses=[{"file_path": "fedot/old.py", "line": 1,
                                      "mechanism": "previous source mechanism",
                                      "proposed_change": "previous source proposal"}])
    assert len(catalog) >= 2
    assert catalog[0] != catalog[1]
    assert len(trace["llm_picks"]) == 2
    assert "Already selected causal hypotheses" in prompts[-1]
    assert _causal_fields(2)["mechanism"] in prompts[-1]
    assert all("previous source proposal" in prompt for prompt in prompts)
    assert "adjacent line does not make it new" in prompts[-1]


def test_propose_patch_runs_then_patches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fedotllm.agents.evolve.agents.propose import PatchProposal, propose_patch

    calls = {"n": 0}

    class _Inf:
        def create(self, prompt, _schema):
            calls["n"] += 1
            if calls["n"] == 1:
                return PatchProposal(status="run", run_code="print(1)", file_path="fedot/core/a.py")
            assert "ran-ok" in prompt
            return PatchProposal(
                file_path="fedot/core/a.py",
                old_code="return x",
                new_code="return x + 1",
            )

    monkeypatch.setattr(
        "fedotllm.agents.evolve.execution.run_code.run_fedot_snippet",
        lambda *_a, **_k: SnippetResult(
            status="ok", code="print(1)", stdout="ran-ok"
        ),
    )
    cand = propose_patch(
        inference=_Inf(),
        context="# fedot/core/a.py\nreturn x\n",
        checkout=tmp_path,
    )
    assert calls["n"] == 2
    assert cand is not None
    assert cand.new_code == "return x + 1"


def test_propose_patch_has_dedicated_synthesis_after_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.execution import run_code
    from fedotllm.agents.evolve.agents.propose import PatchProposal, propose_patch

    source = tmp_path / "checkout"
    target = source / "fedot" / "core" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(run_code, "MAX_STEPS", 2)

    class _Inf:
        def __init__(self):
            self.prompts: list[str] = []

        def create(self, prompt, _schema):
            self.prompts.append(prompt)
            if len(self.prompts) <= 2:
                return PatchProposal(
                    status="read",
                    file_path="fedot/core/a.py",
                    line=1,
                )
            assert "FINAL SYNTHESIS STEP" in prompt
            assert "action=read" in prompt
            return PatchProposal(
                status="patch",
                file_path="fedot/core/a.py",
                old_code="return 1",
                new_code="return 2",
                rationale="the collected evidence supports the alternative",
                behavior_probe="print('EVOLVE_OBSERVATION=2')",
            )

    inference = _Inf()
    candidate = propose_patch(
        inference=inference,
        context="# fedot/core/a.py\ndef value():\n    return 1\n",
        checkout=source,
    )

    assert len(inference.prompts) == 3
    assert candidate is not None
    assert candidate.new_code == "return 2"


def test_propose_policy_failure_is_not_downgraded_to_no_patch():
    from fedotllm.agents.evolve.agents.failures import AgentModelFailure
    from fedotllm.agents.evolve.agents.propose import propose_patch

    class Inference:
        calls = 0

        def create(self, _prompt, _schema):
            self.calls += 1
            raise RuntimeError("Access denied by security policy")

    inference = Inference()
    with pytest.raises(AgentModelFailure) as caught:
        propose_patch(inference=inference, context="fedot source")

    assert caught.value.category == "provider_policy"
    assert caught.value.infrastructure is True
    assert inference.calls == 1


def test_public_contract_failure_becomes_correctness_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fedotllm.agents.evolve.discovery import contracts

    first = contracts._CONTRACTS[0]
    target = tmp_path / first.candidate_files[0]
    target.parent.mkdir(parents=True)
    target.write_text("pass\n", encoding="utf-8")

    def failed(_checkout, code):
        return SnippetResult(
            status="runtime_error",
            code=code,
            stdout='EVOLVE_OBSERVATION={"outcome": "InvalidParameterError"}\n',
            stderr="AssertionError: public contract failed",
            exit_code=1,
        )

    monkeypatch.setattr(contracts, "_CONTRACTS", (first,))
    leads, rows = contracts.discover_contract_violations(tmp_path, run_fn=failed)

    assert rows[0]["status"] == "violated"
    assert leads[0].hypothesis_kind == "correctness"
    verification = contracts.verification_from_contract_lead(leads[0])
    assert verification is not None
    assert verification.status == "verified_bug"
    assert verification.reproduction_code == first.probe


def test_behavioral_repair_does_not_require_private_symbol_name():
    from fedotllm.agents.evolve.benchmark.micro_discovery import _case_repaired

    stages = [
        {"stage": "stock_probe", "status": "passed"},
        {"stage": "file_localization", "status": "passed"},
        {"stage": "symbol_localization", "status": "failed"},
        {"stage": "patch", "status": "passed"},
        {"stage": "import_gate", "status": "passed"},
        {"stage": "behavior_probe", "status": "passed"},
    ]
    assert _case_repaired(stages) is True


def test_resume_loads_saved_candidate_before_new_model_work(tmp_path: Path):
    import json

    from fedotllm.agents.evolve.storage.replay import load_resume_branch

    candidate_id = "saved-candidate"
    lead = {
        "channel": "public_contract",
        "file_path": "fedot/core/a.py",
        "line": 4,
        "why": "rows stay aligned",
        "hypothesis_kind": "correctness",
    }
    (tmp_path / "journal.jsonl").write_text(
        json.dumps(
            {
                "event": "verification",
                "hypothesis_id": "h-1",
                "lead": lead,
                "status": "verified_bug",
                "claim": "rows stay aligned",
                "reproduction_code": "assert False",
                "evidence": ["controller_public_contract_probe"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    folder = tmp_path / "candidates" / candidate_id
    folder.mkdir(parents=True)
    folder.joinpath("candidate.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "hypothesis_id": "h-1",
                "lead": lead,
                "status": "awaiting_probe_repair",
                "behavior_probe": "print('EVOLVE_OBSERVATION=x')",
                "edits": [
                    {
                        "file_path": "fedot/core/a.py",
                        "old_code": "return 1",
                        "new_code": "return 2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    branch = load_resume_branch(tmp_path, candidate_id=candidate_id)

    assert branch is not None
    assert branch["candidate"] is not None
    assert branch["candidate"].new_code == "return 2"
    assert branch["candidate_status"] == "awaiting_probe_repair"
    assert branch["lead"].hypothesis_kind == "correctness"


def test_llm_summary_keeps_started_calls_without_a_result(tmp_path: Path):
    import json

    from fedotllm.agents.evolve.controller.session import _llm_attempt_summary

    events = [
        {"event": "llm_structured_started", "call_group_id": "a"},
        {"event": "llm_query_started", "call_group_id": "a", "query_number": 1},
        {"event": "llm_query", "call_group_id": "a", "status": "ok"},
        {"event": "llm_structured_started", "call_group_id": "b"},
        {"event": "llm_query_started", "call_group_id": "b", "query_number": 1},
    ]
    (tmp_path / "llm_calls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )

    summary = _llm_attempt_summary(tmp_path)

    assert summary["provider_queries_started"] == 2
    assert summary["provider_queries_succeeded"] == 1
    assert summary["provider_queries_failed"] == 0
    assert summary["provider_queries_unfinished"] == 1
    assert summary["structured_calls_started"] == 2
    assert summary["structured_calls_succeeded"] == 0
    assert summary["structured_calls_unfinished"] == 2


def test_llm_summary_prefers_actual_provider_attempts_and_counts_retries(tmp_path: Path):
    import json

    from fedotllm.agents.evolve.controller.session import _llm_attempt_summary

    events = [
        {"event": "llm_query_started"},
        {"event": "provider_attempt_started", "attempt_id": "a", "is_retry": False},
        {"event": "provider_attempt_result", "attempt_id": "a", "status": "error", "error_type": "TimeoutError"},
        {"event": "provider_attempt_started", "attempt_id": "b", "is_retry": True},
        {"event": "provider_attempt_result", "attempt_id": "b", "status": "ok"},
        {"event": "provider_local_rejection", "error_type": "EvolveBudgetExhausted"},
        {"event": "llm_query", "status": "ok"},
    ]
    (tmp_path / "llm_calls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )

    summary = _llm_attempt_summary(tmp_path)

    assert summary["provider_queries_started"] == 2
    assert summary["provider_queries_succeeded"] == 1
    assert summary["provider_queries_failed"] == 1
    assert summary["provider_queries_unfinished"] == 0
    assert summary["provider_retries"] == 1
    assert summary["provider_local_rejections"] == 1


def test_findings_count_defect_identity_across_different_patches(tmp_path: Path):
    import json

    from fedotllm.agents.evolve.storage.findings import append_finding

    findings = tmp_path / "findings.jsonl"

    def record(run_id: str, patch_hash: str, code: str, *, resumed: bool = False):
        append_finding(
            findings,
            run_number=1,
            run_id=run_id,
            source_commit="commit",
            source_hash="source",
            workspace=tmp_path,
            row={
                "candidate": f"candidate-{run_id}",
                "patch_hash": patch_hash,
                "lead": {
                    "channel": "execution",
                    "file_path": "fedot/core/a.py",
                    "line": 1,
                    "why": "public result must stay aligned",
                },
                "reproduction": {
                    "stock": "failed_as_predicted",
                    "patched": "resolved",
                    "code": code,
                },
                "fedot_test_status": "passed",
                "reason": "correctness_keep",
                "keep": False,
                "resumed_branch": resumed,
            },
        )

    record("first", "patch-a", "assert public_call() == 2")
    record("second", "patch-b", "assert public_call() == 2")
    record("third", "patch-c", "assert another_call() == 4", resumed=True)

    rows = [
        json.loads(line)
        for line in findings.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["novelty"]["category"] == "new_unique"
    assert rows[1]["novelty"]["category"] == "repeat"
    assert rows[1]["novelty"]["repeat"] is True
    assert rows[2]["novelty"]["category"] == "continuation"
    assert rows[2]["novelty"]["continuation"] is True
