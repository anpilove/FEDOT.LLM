from __future__ import annotations

from pathlib import Path

import pytest

from research.evolve.metric_agent.compare import compare, compare_pack
from research.evolve.metric_agent.context import context_from_lead, context_from_traceback, inspect_trace, show_source
from research.evolve.metric_agent.discover import format_lint_for_llm, parse_lint, parse_pytest_output
from research.evolve.metric_agent.guard import guard_path
from research.evolve.metric_agent.patch import apply_patch
from research.evolve.metric_agent.tasks import list_task_metadata, load_task
from research.evolve.metric_agent.types import Lead, PatchCandidate, ScoreResult


def _score(task_id: str, status: str, score: float) -> ScoreResult:
    return ScoreResult(task_id=task_id, status=status, score=score)


def test_guard_denies_evaluator_paths():
    assert guard_path("data/cases.json") == "deny"
    assert guard_path("research/evolve/metric_agent/scorer.py") == "deny"
    assert guard_path("fedotllm/llm.py") == "deny"
    assert guard_path("_local_fedot_patches/replacements/pca_keep_cats.py") == "deny"


def test_list_tasks_strips_bug_field():
    rows = list_task_metadata()
    ids = {row["task_id"] for row in rows}
    assert "pca->catboost" in ids
    assert "catboost" in ids
    assert "fast_ica->lgbm" in ids
    for row in rows:
        assert "bug" not in row
        assert "role" not in row


def test_fail_task_must_not_regress_fast_ica():
    spec = load_task("pca->catboost")
    assert spec.must_not_regress == ("fast_ica->lgbm",)
    assert spec.min_delta == 0.01
    assert spec.sentinel == 0.5


def test_compare_keep_crash_to_metric():
    decision = compare(
        _score("pca->catboost", "crash", 0.5),
        _score("pca->catboost", "ok", 0.85),
            [(_score("fast_ica->lgbm", "ok", 0.80), _score("fast_ica->lgbm", "ok", 0.80))],
        min_delta=0.01,
        sentinel=0.5,
    )
    assert decision.keep is True
    assert decision.target_delta == pytest.approx(0.35)


def test_compare_drop_regression():
    decision = compare(
        _score("pca->catboost", "crash", 0.5),
        _score("pca->catboost", "ok", 0.85),
        [(_score("fast_ica->lgbm", "ok", 0.80), _score("fast_ica->lgbm", "ok", 0.70))],
        min_delta=0.01,
        sentinel=0.5,
    )
    assert decision.keep is False
    assert "regression" in decision.reason


def test_timeout_is_not_sentinel():
    decision = compare(
        _score("pca->catboost", "ok", 0.5),
        _score("pca->catboost", "timeout", float("nan")),
        [],
        min_delta=0.01,
        sentinel=0.5,
    )
    assert decision.keep is False
    assert decision.target_delta is None
    assert decision.reason == "timeout_or_invalid"


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
    ctx = context_from_traceback(
        ScoreResult(task_id="t", status="crash", score=0.5, traceback=traceback, detail="IndexError: x"),
        tmp_path,
    )
    assert "scorer.py" not in ctx
    assert "fedot/core/pca.py" in ctx
    assert "IndexError: x" in ctx


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
    from research.evolve.metric_agent.guard import repo_root

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

    from research.evolve.metric_agent.guard import repo_root

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


def test_parse_lint_and_prompt_has_no_exam_ids():
    row = parse_lint("fedot/core/data/data.py:718:12: F821 undefined name 'x'")
    assert row is not None
    assert row["file"] == "fedot/core/data/data.py"
    assert row["line"] == 718
    text = format_lint_for_llm(
        [Lead(channel="lint", file_path=row["file"], line=row["line"], why=f"{row['rule']} {row['message']}")]
    )
    assert "pca->catboost" not in text
    assert "cases.json" not in text
    assert "leftover" not in text.lower()


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


def test_rank_leads_prefers_core_over_api():
    from research.evolve.metric_agent.discover import _rank_leads, parse_lint

    ranked = _rank_leads(
        [
            Lead(channel="lint", file_path="fedot/api/api_utils/api_data.py", line=1, why="x"),
            Lead(channel="lint", file_path="fedot/core/data/data.py", line=2, why="y"),
        ]
    )
    assert ranked[0].file_path.startswith("fedot/core/")
    row = parse_lint("fedot/api/api_utils/api_composer.py:185:12: RUF010 Use explicit conversion flag")
    assert row is not None
    from research.evolve.metric_agent.discover import _LINT_NOISE

    assert _LINT_NOISE.match(row["rule"])
    from research.evolve.metric_agent.discover import _as_text

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
        Lead(channel="trace", file_path="fedot/core/data.py", line=2, why="IndexError in get_not_encoded_data"),
        tmp_path,
    )
    assert "get_not_encoded_data" in ctx
    assert "Field numerical_idx" in ctx
    assert "fedot/core/operations/pca.py" in ctx


def test_context_from_lead(tmp_path: Path):
    src = tmp_path / "fedot" / "core" / "data.py"
    src.parent.mkdir(parents=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = context_from_lead(Lead(channel="lint", file_path="fedot/core/data.py", line=2, why="F821 x"), tmp_path)
    assert "fedot/core/data.py" in ctx
    assert "Channel:" not in ctx
    assert "pca->catboost" not in ctx
    assert "cases.json" not in ctx


def test_compare_pack_keep_and_protect():
    stock = {
        "pca->catboost": _score("pca->catboost", "crash", 0.5),
        "catboost": _score("catboost", "ok", 0.85),
        "fast_ica->lgbm": _score("fast_ica->lgbm", "ok", 0.80),
    }
    patched = {
        "pca->catboost": _score("pca->catboost", "ok", 0.85),
        "catboost": _score("catboost", "ok", 0.85),
        "fast_ica->lgbm": _score("fast_ica->lgbm", "ok", 0.80),
    }
    decision = compare_pack(
        stock,
        patched,
        lift_ids=("pca->catboost",),
        protect_ids=("catboost", "fast_ica->lgbm"),
    )
    assert decision.keep is True
    assert decision.target_delta == pytest.approx(0.35)


def test_scoreboard_summarize_getting_better(tmp_path: Path):
    from research.evolve.metric_agent.scoreboard import append_attempt, summarize
    from research.evolve.metric_agent.types import Decision as Dec

    stock = {"pca->catboost": _score("pca->catboost", "crash", 0.5)}
    patched = {"pca->catboost": _score("pca->catboost", "ok", 0.85)}
    append_attempt(
        tmp_path,
        lead=Lead(channel="fedot_test", file_path="fedot/core/data.py", line=2),
        candidate=None,
        stock=stock,
        patched=patched,
        decision=Dec(keep=True, reason="keep", target_delta=0.35, regression_deltas={}),
    )
    summary = summarize(tmp_path)
    assert summary["attempts"] == 1
    assert summary["keeps"] == 1
    assert summary["getting_better"] is True
    assert summary["best_delta"] == pytest.approx(0.35)


def test_propose_prompt_hides_exam_catalog():
    from research.evolve.metric_agent.propose import build_prompt

    prompt = build_prompt("# fedot/core/data.py\n    return x")
    assert "pca->catboost" not in prompt
    assert "cases.json" not in prompt
    assert "leftover" not in prompt.lower()
    assert "data/cases.json" not in prompt


def test_scout_stack_source_has_no_cases_catalog():
    from research.evolve.metric_agent.guard import repo_root

    folder = repo_root() / "research" / "evolve" / "metric_agent"
    for name in ("discover.py", "scout.py", "fixer.py", "propose.py", "context.py", "repo_map.py"):
        text = (folder / name).read_text(encoding="utf-8")
        assert "from research.evolve.metric_agent.tasks" not in text
        assert "pca->catboost" not in text


def test_leads_from_scores_uses_trace_not_task_id(tmp_path: Path):
    from research.evolve.metric_agent.discover import leads_from_scores

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
        )
    }
    leads = leads_from_scores(stock, tmp_path)
    assert leads
    assert leads[0].channel == "trace"
    assert leads[0].file_path == "fedot/core/data.py"
    assert leads[0].line == 2
    assert "pca->catboost" not in leads[0].why
    assert "IndexError" in leads[0].why


def test_failed_pytest_nodes_and_gate():
    from research.evolve.metric_agent.discover import failed_pytest_nodes
    from research.evolve.metric_agent.judge import tests_regressed

    text = "FAILED test/unit/data/test_data.py::test_idx - IndexError: x\n"
    nodes = failed_pytest_nodes(text)
    assert nodes == {"test/unit/data/test_data.py::test_idx"}
    assert tests_regressed(nodes, nodes) is None
    blocked = tests_regressed(set(), nodes)
    assert blocked is not None
    assert blocked.keep is False
    assert "fedot_tests_regressed" in blocked.reason
    assert "pca->catboost" not in blocked.reason


def test_repo_map_ranks_symbol_and_field(tmp_path: Path):
    from research.evolve.metric_agent.repo_map import repo_map, search_callers, search_field_usage

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
    from research.evolve.metric_agent.repo_map import search_field_usage

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
    from research.evolve.metric_agent.repo_map import repo_map

    aaa = tmp_path / "fedot" / "core" / "aaa" / "a.py"
    zzz = tmp_path / "fedot" / "core" / "zzz" / "z.py"
    aaa.parent.mkdir(parents=True)
    zzz.parent.mkdir(parents=True)
    aaa.write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    zzz.write_text("class Z:\n    def h(self):\n        return 1\n", encoding="utf-8")
    mapped = repo_map(tmp_path, (), limit=2)
    areas = {item.file_path.split("/")[2] for item in mapped}
    assert "aaa" in areas
    assert "zzz" in areas


def test_repo_map_prefers_fit_inside_package(tmp_path: Path):
    from research.evolve.metric_agent.repo_map import repo_map

    knn = tmp_path / "fedot" / "core" / "operations" / "knn.py"
    knn.parent.mkdir(parents=True)
    knn.write_text(
        "def helper():\n    return 1\n\nclass Knn:\n    def fit(self, data):\n        return data\n",
        encoding="utf-8",
    )
    mapped = repo_map(tmp_path, (), limit=1)
    assert mapped[0].name == "fit"


def test_discover_walks_whole_core_not_one_field(tmp_path: Path):
    from research.evolve.metric_agent.discover import discover_leads

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
        "        return data\n",
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
    from research.evolve.metric_agent.discover import discover_leads

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


def test_discover_runtime_methods_before_helpers(tmp_path: Path):
    from research.evolve.metric_agent.discover import discover_leads

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
    from research.evolve.metric_agent.loop import run_once

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    stock = {"pca->catboost": _score("pca->catboost", "crash", 0.5)}
    order: list[str] = []

    def fake_tests(*_a, **_k):
        order.append("fedot_tests")
        return "", [], set()

    def fake_scout(*_a, **kwargs):
        order.append("scout")
        assert "stock" not in kwargs
        return []

    def fake_stock(*_a, **_k):
        order.append("stock")
        return stock

    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_fedot_tests", fake_tests)
    monkeypatch.setattr("research.evolve.metric_agent.loop.scout", fake_scout)
    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_stock", fake_stock)
    run_once(checkout=checkout, workspace=tmp_path, max_leads=1, inference=None)
    assert order == ["scout", "stock"]


def test_run_once_drops_on_new_fedot_test_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from research.evolve.metric_agent.loop import run_once
    from research.evolve.metric_agent.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    calls = {"n": 0}

    def fake_tests(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "", [], set()
        return "FAILED test/unit/new.py::test_x", [], {"test/unit/new.py::test_x"}

    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_fedot_tests", fake_tests)
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.scout",
        lambda *_a, **_k: [Lead(channel="lint", file_path="fedot/a.py", line=1, why="B006")],
    )
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.fix_lead",
        lambda *_a, **_k: PatchCandidate("c1", "fedot/a.py", "old", "new"),
    )
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.measure_stock",
        lambda *_a, **_k: {"pca->catboost": _score("pca->catboost", "crash", 0.5)},
    )
    patched_called = {"n": 0}

    def boom(*_a, **_k):
        patched_called["n"] += 1
        raise AssertionError("hidden exam must not run if FEDOT tests regressed")

    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_patched", boom)
    monkeypatch.setattr("research.evolve.metric_agent.loop.revert_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr("research.evolve.metric_agent.loop.snapshot_diff", lambda *_a, **_k: "")
    decision = run_once(checkout=checkout, workspace=tmp_path, max_leads=1, inference=object())
    assert decision.keep is False
    assert "fedot_tests_regressed" in decision.reason
    assert patched_called["n"] == 0


def test_run_once_drops_unimportable_before_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from research.evolve.metric_agent.loop import run_once
    from research.evolve.metric_agent.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    (checkout / "fedot").mkdir(parents=True)
    (checkout / "fedot" / "__init__.py").write_text("", encoding="utf-8")
    (checkout / "fedot" / "a.py").write_text("broken = definitely_not_defined\n", encoding="utf-8")

    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.scout",
        lambda *_a, **_k: [Lead(channel="repo_map", file_path="fedot/a.py", line=1, why="x")],
    )
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.fix_lead",
        lambda *_a, **_k: PatchCandidate("c1", "fedot/a.py", "old", "new"),
    )
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.measure_stock",
        lambda *_a, **_k: {"t": _score("t", "crash", 0.5)},
    )
    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_fedot_tests", lambda *_a, **_k: ("", [], set()))
    patched_called = {"n": 0}

    def boom(*_a, **_k):
        patched_called["n"] += 1
        raise AssertionError("holdout must not run if patch does not import")

    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_patched", boom)
    monkeypatch.setattr("research.evolve.metric_agent.loop.revert_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr("research.evolve.metric_agent.loop.snapshot_diff", lambda *_a, **_k: "diff")
    decision = run_once(checkout=checkout, workspace=tmp_path, max_leads=1, inference=object())
    assert decision.keep is False
    assert "patch_unimportable" in decision.reason
    assert patched_called["n"] == 0


def test_run_once_tries_second_lead_and_writes_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from research.evolve.metric_agent.loop import run_once
    from research.evolve.metric_agent.scoreboard import summarize
    from research.evolve.metric_agent.types import PatchCandidate

    checkout = tmp_path / "fedot-src"
    checkout.mkdir()
    leads = [
        Lead(channel="lint", file_path="fedot/a.py", line=1, why="B006"),
        Lead(channel="fedot_test", file_path="fedot/b.py", line=2, why="test/unit/x.py"),
    ]
    monkeypatch.setattr("research.evolve.metric_agent.loop.scout", lambda *a, **k: leads)
    monkeypatch.setattr("research.evolve.metric_agent.loop.revert_checkout", lambda *a, **k: None)
    monkeypatch.setattr("research.evolve.metric_agent.loop.snapshot_diff", lambda *a, **k: "--- a\n+++ b")
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.measure_fedot_tests",
        lambda *a, **k: ("", [], set()),
    )

    stock = {
        "pca->catboost": _score("pca->catboost", "crash", 0.5),
        "catboost": _score("catboost", "ok", 0.85),
        "fast_ica->lgbm": _score("fast_ica->lgbm", "ok", 0.80),
    }
    monkeypatch.setattr("research.evolve.metric_agent.loop.measure_stock", lambda *a, **k: stock)

    cands = [
        PatchCandidate("c1", "fedot/a.py", "old", "new"),
        PatchCandidate("c2", "fedot/b.py", "old2", "new2"),
    ]
    monkeypatch.setattr("research.evolve.metric_agent.loop.fix_lead", lambda *a, **k: cands.pop(0))

    patched_runs = [
        {
            "pca->catboost": _score("pca->catboost", "crash", 0.5),
            "catboost": _score("catboost", "ok", 0.85),
            "fast_ica->lgbm": _score("fast_ica->lgbm", "ok", 0.80),
        },
        {
            "pca->catboost": _score("pca->catboost", "ok", 0.85),
            "catboost": _score("catboost", "ok", 0.85),
            "fast_ica->lgbm": _score("fast_ica->lgbm", "ok", 0.80),
        },
    ]
    monkeypatch.setattr(
        "research.evolve.metric_agent.loop.measure_patched",
        lambda *a, **k: patched_runs.pop(0),
    )

    decision = run_once(checkout=checkout, workspace=tmp_path, max_leads=2, inference=object())
    assert decision.keep is True
    summary = summarize(tmp_path)
    assert summary["attempts"] == 2
    assert summary["keeps"] == 1
    assert summary["getting_better"] is True
    journal = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert '"stock"' in journal
    assert '"patched"' in journal
    assert "pca->catboost" in (tmp_path / "scoreboard.jsonl").read_text(encoding="utf-8")


def test_replay_reads_cmd_log_and_diff(tmp_path: Path):
    from research.evolve.metric_agent.journal import append_journal
    from research.evolve.metric_agent.replay import load_replay

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
