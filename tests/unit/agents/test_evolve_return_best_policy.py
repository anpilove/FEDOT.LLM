"""Return_best compatibility policy: refuse by default, research worktree optional."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedotllm.agents.evolve.evaluation.return_best_policy import (
    POLICY_REFUSE,
    POLICY_RESEARCH_WORKTREE,
    resolve_return_best_source,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _stock(tmp_path: Path) -> Path:
    root = tmp_path / "stock"
    _write(
        root / "fedot" / "api" / "api_utils" / "api_params_repository.py",
        "default_param_values_dict = dict(seed=None)\n",
    )
    _write(
        root / "fedot" / "core" / "operations" / "evaluation" /
        "operation_implementations" / "data_operations" / "sklearn_transformations.py",
        "import dask.array as da\n",
    )
    return root


def _ready(tmp_path: Path, name: str = "ready") -> Path:
    root = tmp_path / name
    _write(
        root / "fedot" / "api" / "api_utils" / "api_params_repository.py",
        "timeout_policy='legacy'\nfinal_fit_reserve_seconds=0\nmax_pipeline_evaluation_time=None\n",
    )
    _write(
        root / "fedot" / "core" / "optimisers" / "isolated_gp_optimizer.py",
        "class IsolatedEvoGraphOptimizer: pass\n",
    )
    _write(
        root / "fedot" / "core" / "operations" / "evaluation" /
        "operation_implementations" / "data_operations" / "sklearn_transformations.py",
        "import dask.array as da\n",
    )
    return root


def test_refuse_policy_fails_closed_on_stock_pin(tmp_path):
    decision = resolve_return_best_source(_stock(tmp_path), policy=POLICY_REFUSE)
    assert decision.ok is False
    assert decision.policy == POLICY_REFUSE
    assert "refuse" in decision.detail


def test_research_worktree_policy_accepts_compatible_offline_tree(tmp_path):
    stock = _stock(tmp_path)
    ready = _ready(tmp_path)
    decision = resolve_return_best_source(
        stock,
        policy=POLICY_RESEARCH_WORKTREE,
        research_worktree=ready,
    )
    assert decision.ok is True
    assert decision.source == ready.resolve()
    assert "research return_best worktree" in decision.detail


def test_research_worktree_policy_requires_env_path(tmp_path):
    decision = resolve_return_best_source(
        _stock(tmp_path),
        policy=POLICY_RESEARCH_WORKTREE,
        research_worktree=None,
    )
    assert decision.ok is False
    assert "EVOLVE_FEDOT_RETURN_BEST_ROOT" in decision.detail


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported return_best policy"):
        resolve_return_best_source(Path("."), policy="auto-merge-fedot")
