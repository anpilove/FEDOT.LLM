"""Unit tests for the return_best budget prototype (no FEDOT fit, no datasets)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("return_best_budget.py")
_SPEC = importlib.util.spec_from_file_location("return_best_budget", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ReturnBestBudget = _MODULE.ReturnBestBudget
validate_worker_timeout_kwargs = _MODULE.validate_worker_timeout_kwargs


def test_registry_like_profile_keeps_final_fit_reserve_out_of_search():
    budget = ReturnBestBudget(
        side_timeout_seconds=5400,
        fedot_search_budget_seconds=4500,
        final_fit_reserve_seconds=1500,
        cv_folds=5,
    )
    assert budget.search_deadline_seconds == 3000
    assert budget.composing_budget_seconds == pytest.approx(1800)
    assert budget.tuning_budget_seconds == pytest.approx(1200)
    assert budget.max_pipeline_evaluation_time_seconds == 900
    assert budget.max_pipeline_fit_time_seconds == 180


def test_exhausted_search_skips_candidates_but_keeps_final_reserve():
    budget = ReturnBestBudget(
        side_timeout_seconds=5400,
        fedot_search_budget_seconds=4500,
        final_fit_reserve_seconds=1500,
        cv_folds=5,
    )
    assert budget.may_start_candidate(2999.0) is True
    assert budget.may_start_candidate(3000.0) is False
    assert budget.may_start_final_fit(3000.0) is True
    assert budget.remaining_search_seconds(3100.0) == 0.0


def test_invalid_budgets_fail_closed():
    with pytest.raises(ValueError):
        ReturnBestBudget(5400, 4500, 4500, 5)
    with pytest.raises(ValueError):
        ReturnBestBudget(4500, 4500, 0, 5)
    with pytest.raises(ValueError):
        ReturnBestBudget(5400, 4500, -1, 5)


def test_validate_worker_kwargs_accepts_return_best_and_rejects_unknown():
    ok = validate_worker_timeout_kwargs(
        {
            "timeout_policy": "return_best",
            "final_fit_reserve_seconds": 1500,
            "max_pipeline_evaluation_time": 15.0,
        }
    )
    assert ok["timeout_policy"] == "return_best"
    with pytest.raises(KeyError):
        validate_worker_timeout_kwargs({"timeout_policy": "return_best"})
    with pytest.raises(KeyError):
        validate_worker_timeout_kwargs({"timeout_policy": "mystery"})
