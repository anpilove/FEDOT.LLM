import json

from fedotllm.agents.evolve.evaluation.early_screen import (
    frozen_screen_tasks,
    screen_pair_archive,
)
from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry


class _Dataset:
    def __init__(self, task_id: str, problem: str):
        self.task_id = task_id
        self.problem = problem


def test_screen_triple_follows_registry_order_and_ignores_pair_status():
    datasets = [
        _Dataset("first-class", "classification"),
        _Dataset("late-class", "classification"),
        _Dataset("first-reg", "regression"),
        _Dataset("first-ts", "ts_forecasting"),
        _Dataset("late-ts", "ts_forecasting"),
    ]
    # Registry order fixes the triple. A later dataset of the same problem
    # cannot replace it after an archive is opened.
    assert frozen_screen_tasks(datasets) == {
        "classification": "first-class",
        "regression": "first-reg",
        "ts_forecasting": "first-ts",
    }

    registry = load_quality_registry()
    assert frozen_screen_tasks(registry.datasets) == {
        "classification": "openml-31-credit-g",
        "regression": "openml-regression-moneyball",
        "ts_forecasting": "fedot-ts-beer",
    }


def _pairs(registry, proposal: str, statuses: dict[str, str] | None = None, default: str = "improved"):
    statuses = statuses or {}
    return [
        {
            "proposal": proposal,
            "task_id": dataset.task_id,
            "problem": dataset.problem,
            "status": statuses.get(dataset.task_id, default),
        }
        for dataset in registry.datasets
    ]


def test_early_screen_separates_improvement_regression_and_timeout():
    registry = load_quality_registry()
    screen = frozen_screen_tasks(registry.datasets)
    late_regression = next(
        dataset.task_id
        for dataset in registry.datasets
        if dataset.task_id not in screen.values()
    )
    pairs = []
    pairs.extend(_pairs(registry, "gain"))
    pairs.extend(_pairs(registry, "screen-regression", {screen["classification"]: "regression"}))
    pairs.extend(_pairs(registry, "late-regression", {late_regression: "regression"}))
    pairs.extend(_pairs(registry, "timeout", {screen["ts_forecasting"]: "timeout"}))
    pairs.extend(_pairs(registry, "missing", {late_regression: "missing_metric"}))

    report = screen_pair_archive(pairs, registry.datasets)

    by_proposal = {row["proposal"]: row for row in report["proposal_decisions"]}
    assert by_proposal["gain"] == {
        "proposal": "gain",
        "early": "continue",
        "full": "keep",
        "pairs_saved": 0,
        "early_reject": False,
        "keep": True,
    }
    assert by_proposal["screen-regression"]["early"] == "regression"
    assert by_proposal["screen-regression"]["full"] == "regression"
    assert by_proposal["screen-regression"]["pairs_saved"] == 37
    assert by_proposal["late-regression"]["early"] == "continue"
    assert by_proposal["late-regression"]["full"] == "regression"
    assert by_proposal["late-regression"]["pairs_saved"] == 0
    assert by_proposal["timeout"]["early"] == "timeout"
    assert by_proposal["timeout"]["full"] == "timeout"
    assert by_proposal["timeout"]["early_reject"] is False
    assert by_proposal["missing"]["early"] == "continue"
    assert by_proposal["missing"]["full"] == "missing_metric"
    assert by_proposal["missing"]["keep"] is False

    assert report["pairs_saved"] == 37
    assert report["early_rejections"] == 1
    assert report["timeout_pairs"] == 1
    assert report["missing_metric_pairs"] == 1
    assert report["early_keeps"] == []
    assert report["full_keeps"] == ["gain"]
    assert report["screen_fixed_before_results"] is True
    assert report["disagreements"] == [
        {
            "proposal": "late-regression",
            "early": "continue",
            "full": "regression",
            "pairs_saved": 0,
            "early_reject": False,
            "keep": False,
            "kind": "screen_missed_regression",
        }
    ]


def test_timeout_and_mislabeled_problem_are_not_valid_regressions():
    registry = load_quality_registry()
    screen = frozen_screen_tasks(registry.datasets)
    pairs = _pairs(registry, "labeled-timeout", {screen["classification"]: "timeout"})
    pairs.append(
        {
            "proposal": "labeled-timeout",
            "task_id": screen["classification"],
            "problem": "regression",
            "status": "regression",
        }
    )

    report = screen_pair_archive(pairs, registry.datasets)
    decision = report["proposal_decisions"][0]

    assert decision["early"] == "timeout"
    assert decision["early_reject"] is False
    assert decision["full"] == "timeout"
    assert report["pairs_saved"] == 0
    assert report["problem_mismatches"] == 1
    assert report["disagreements"] == []


def test_early_screen_command_prints_the_archive_report(tmp_path, capsys):
    from fedotllm.agents.evolve.__main__ import main

    registry = load_quality_registry()
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps({"pairs": _pairs(registry, "gain")}),
        encoding="utf-8",
    )

    assert main(["early-screen", "--pairs", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["full_keeps"] == ["gain"]
    assert payload["pairs_saved"] == 0
    assert payload["early_keeps"] == []
