"""Paired bootstrap uncertainty, not an IID claim about model-seed repeats."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_interval(before, after, *, metric, time_series=False, comparisons=3, repeats=2000):
    left, right = before.metric_observations, after.metric_observations
    y = np.asarray(left.get("target", []), dtype=float)
    other_y = np.asarray(right.get("target", []), dtype=float)
    a = np.asarray(left.get("prediction", []), dtype=float)
    b = np.asarray(right.get("prediction", []), dtype=float)
    if (y.ndim != 1 or len(y) < 8 or a.shape != y.shape or b.shape != y.shape
            or not np.array_equal(y, other_y) or not np.isfinite([y, a, b]).all()):
        return {"status": "insufficient_or_invalid_paired_observations", "n": len(y)}
    if metric not in {"holdout_roc_auc", "holdout_rmse"}:
        return {"status": "unsupported_metric", "n": len(y)}
    if metric == "holdout_roc_auc":
        counts = np.unique(y, return_counts=True)[1]
        if len(counts) != 2 or min(counts) < 4:
            return {"status": "insufficient_class_support", "n": len(y)}
    rng = np.random.default_rng(20260910)
    deltas = []
    block = max(2, int(np.sqrt(len(y)))) if time_series else 1
    for _ in range(repeats):
        if time_series:
            starts = rng.integers(0, len(y) - block + 1, size=int(np.ceil(len(y) / block)))
            ids = np.concatenate([np.arange(s, s + block) for s in starts])[:len(y)]
        else:
            ids = rng.integers(0, len(y), size=len(y))
        if metric == "holdout_roc_auc":
            if len(np.unique(y[ids])) != 2:
                continue
            delta = roc_auc_score(y[ids], b[ids]) - roc_auc_score(y[ids], a[ids])
        else:
            delta = np.sqrt(np.mean((y[ids] - a[ids]) ** 2)) - np.sqrt(np.mean((y[ids] - b[ids]) ** 2))
        deltas.append(float(delta))
    if len(deltas) < repeats * 0.9:
        return {"status": "insufficient_valid_resamples", "n": len(y)}
    alpha = 0.05 / comparisons
    lower, upper = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "status": "estimated", "n": len(y), "lower": float(lower), "upper": float(upper),
        "excludes_zero": bool(lower > 0), "repeats": len(deltas),
        "method": "paired_moving_block_bootstrap" if time_series else "paired_bootstrap",
        "block_length": block, "familywise_alpha": 0.05, "comparisons": comparisons,
        "limitation": "approximate interval; small samples and nonstationary time series may be unreliable",
    }
