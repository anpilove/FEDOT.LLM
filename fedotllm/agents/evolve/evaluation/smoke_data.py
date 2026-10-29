"""Small deterministic acceptance data; never evidence of a quality improvement."""

from __future__ import annotations

from fedotllm.agents.evolve.evaluation.quality_registry import QualityDataset, QualityRegistry

SMOKE_IDENTITY = "controller-smoke-v5-fit-reserve-disabled-sqlite"


def smoke_registry() -> QualityRegistry:
    datasets = tuple(
        QualityDataset(
            task_id=f"smoke-{problem}", source=SMOKE_IDENTITY,
            openml_task=0, openml_data=0, name=f"Acceptance {problem}",
            problem=problem, metric=metric, higher_is_better=problem == "classification",
            min_delta=0.01, fold=0, repeat=0,
            why_full="Controller acceptance only; not a quality benchmark",
            forecast_horizon=12 if problem == "ts_forecasting" else 0,
        )
        for problem, metric in (
            ("classification", "roc_auc"), ("regression", "rmse"), ("ts_forecasting", "mae")
        )
    )
    return QualityRegistry(
        identity=SMOKE_IDENTITY, api="fedot", preset="best_quality", with_tuning=False,
        timeout_seconds=180, fedot_search_budget_seconds=150, watchdog_seconds=600, seed=42,
        final_fit_reserve_seconds=0,
        require_search_ran=False, portfolio=False, cpu_quota=12, n_jobs_per_job=4,
        datasets=datasets, selection_study=0,
        selection_rule="Deterministic 128-row acceptance training data; no downloads",
        fedot_cache_policy="disabled-sqlite-v1",
        composer_options=(("num_of_generations", 2), ("pop_size", 4)),
    )


def load_smoke_fold(dataset: QualityDataset) -> dict:
    import numpy as np
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    if dataset not in smoke_registry().datasets:
        raise ValueError("unknown or changed smoke dataset")
    if dataset.problem == "ts_forecasting":
        steps = np.arange(140, dtype=float)
        series = 20 + 0.05 * steps + 2 * np.sin(2 * np.pi * steps / 12)
        return {"X_train":series[:128], "y_train":series[:128],
                "X_test":series[128:], "y_test":series[128:],
                "n_train":128, "n_test":12, "forecast_horizon":12,
                "loader":SMOKE_IDENTITY}
    if dataset.problem == "classification":
        features, target = make_classification(n_samples=160, n_features=6,
                                               n_informative=4, random_state=42)
    else:
        features, target = make_regression(n_samples=160, n_features=6,
                                           n_informative=4, noise=0.1, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=32, random_state=42,
        stratify=target if dataset.problem == "classification" else None,
    )
    return {"X_train":x_train, "y_train":y_train, "X_test":x_test, "y_test":y_test,
            "n_train":128, "n_test":32, "loader":SMOKE_IDENTITY}
