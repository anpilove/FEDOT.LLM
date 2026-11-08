"""Integration checks against the research FEDOT worktree with return_best patch.

Requires:
  /tmp/fedot-verify/FEDOT-return-best  (git worktree @ 8990f28b + research patch)

These tests do not touch FEDOT.LLM production code and do not claim a quality win.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PATCHED = Path(os.environ.get("FEDOT_RETURN_BEST_ROOT", "/tmp/fedot-verify/FEDOT-return-best"))


def _import_patched_fedot():
    if not PATCHED.is_dir():
        pytest.skip(f"patched FEDOT worktree missing: {PATCHED}")
    # Prefer the patched checkout over site-packages.
    sys.path.insert(0, str(PATCHED))
    for name in list(sys.modules):
        if name == "fedot" or name.startswith("fedot."):
            del sys.modules[name]
    from fedot.api.main import Fedot
    from fedot.api.time import ApiTime

    return Fedot, ApiTime


def test_patched_fedot_accepts_evolve_worker_timeout_kwargs():
    Fedot, _ = _import_patched_fedot()
    model = Fedot(
        problem="classification",
        timeout=0.2,
        seed=42,
        n_jobs=1,
        preset="fast_train",
        with_tuning=False,
        timeout_policy="return_best",
        final_fit_reserve_seconds=5,
        max_pipeline_evaluation_time=0.05,
        use_operations_cache=False,
        use_preprocessing_cache=False,
        use_predictions_cache=False,
    )
    assert model.params.get("timeout_policy") == "return_best"
    assert model.params.get("final_fit_reserve_seconds") == 5
    assert hasattr(model, "fit_budget_outcome")


def test_return_best_shrinks_composing_window_by_final_fit_reserve():
    _, ApiTime = _import_patched_fedot()
    legacy = ApiTime(
        time_for_automl=75,
        with_tuning=True,
        timeout_policy="legacy",
        final_fit_reserve_seconds=1500,
    )
    patched = ApiTime(
        time_for_automl=75,
        with_tuning=True,
        timeout_policy="return_best",
        final_fit_reserve_seconds=1500,
    )
    assert legacy.timeout_for_composing == pytest.approx(45.0)
    assert patched.timeout_for_composing == pytest.approx(30.0)


def test_limited_aa_smoke_on_synthetic_classification_is_measurable():
    """A/A with identical settings; only checks metric_valid + finite scores.

    Not a quality improvement claim. Skips if fit cannot finish quickly.
    """

    Fedot, _ = _import_patched_fedot()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    split = 60
    scores = []
    for side in ("a", "b"):
        model = Fedot(
            problem="classification",
            timeout=0.15,
            seed=42,
            n_jobs=1,
            preset="fast_train",
            with_tuning=False,
            timeout_policy="return_best",
            final_fit_reserve_seconds=3,
            max_pipeline_fit_time=0.05,
            cv_folds=2,
            metric="roc_auc",
            use_operations_cache=False,
            use_preprocessing_cache=False,
            use_predictions_cache=False,
            logging_level=50,
        )
        model.fit(x[:split], y[:split])
        pred = model.predict_proba(x[split:])
        # FEDOT may return class labels or probabilities depending on path.
        from sklearn.metrics import roc_auc_score

        try:
            score = float(roc_auc_score(y[split:], np.asarray(pred).ravel()))
        except Exception as exc:  # noqa: BLE001 - smoke diagnostics only
            pytest.fail(f"side={side} could not score: {exc}; pred={type(pred)}")
        assert math.isfinite(score)
        scores.append(score)
    assert len(scores) == 2
    # Same seed/settings → identical A/A scores expected for this smoke.
    assert scores[0] == pytest.approx(scores[1])
