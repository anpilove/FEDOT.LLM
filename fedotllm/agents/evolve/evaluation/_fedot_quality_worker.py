"""Subprocess worker: one Fedot(best_quality, 60 min) side on a registered fold."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback as tb_mod
from io import StringIO
from pathlib import Path


def _history_generations(history) -> int:
    if history is None:
        return 0
    for attr in ("generations", "individuals", "archive_history"):
        value = getattr(history, attr, None)
        if value is not None:
            try:
                return int(len(value))
            except TypeError:
                continue
    empty = getattr(history, "is_empty", None)
    if callable(empty) and empty():
        return 0
    return 1


def _search_ran(model, log_text: str) -> tuple[bool, str]:
    composer = getattr(model, "api_composer", None)
    was_optimised = bool(getattr(composer, "was_optimised", False))
    generations = _history_generations(getattr(model, "history", None))
    skipped = "timeout is too small for composing" in log_text.lower()
    if skipped or not was_optimised:
        return False, "composing_did_not_start"
    if generations <= 0 and "pipeline composition started" not in log_text.lower():
        return False, "composing_did_not_start"
    return True, "composing_started"


def _positive_class_scores(prediction) -> "object":
    import numpy as np

    pred = np.asarray(getattr(prediction, "predict", prediction))
    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:, 0]
        if pred.shape[1] == 2:
            return pred[:, 1]
        return pred
    return pred.ravel()


def _score_classification(y_test, prediction, metric: str) -> float:
    """Holdout metric from probabilities. Avoids LabelEncoder.inverse_transform."""

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    y = np.asarray(y_test)
    if y.dtype.kind in "OUS":
        y = LabelEncoder().fit_transform(y)
    pred = np.asarray(getattr(prediction, "predict", prediction))
    name = str(metric or "roc_auc")
    if name == "roc_auc":
        scores = _positive_class_scores(pred)
        if getattr(scores, "ndim", 1) == 2:
            return float(roc_auc_score(y, scores, multi_class="ovr"))
        return float(roc_auc_score(y, scores))
    labels = np.argmax(pred, axis=1) if pred.ndim == 2 and pred.shape[1] > 1 else np.rint(pred.ravel())
    if name == "f1":
        return float(f1_score(y, labels, average="macro"))
    return float(accuracy_score(y, labels))


def _predict_holdout(model, dataset, fold) -> str:
    """Official-fold features. Classification uses probs, not labels inverse."""

    if dataset.problem == "ts_forecasting":
        print("worker fit done; forecast", flush=True)
        if hasattr(model, "forecast"):
            model.forecast()
        else:
            model.predict(fold["X_test"])
        return "forecast"
    if dataset.problem == "classification":
        print("worker fit done; predict_proba", flush=True)
        model.predict_proba(fold["X_test"])
        return "predict_proba"
    print("worker fit done; predict", flush=True)
    model.predict(fold["X_test"])
    return "predict"


def _holdout_metrics(model, dataset, fold) -> tuple[float, dict]:
    metric_names = [dataset.metric]
    try:
        metrics = model.get_metrics(target=fold["y_test"], metric_names=metric_names)
        return float(metrics[dataset.metric]), dict(metrics)
    except Exception:
        if dataset.problem != "classification" or getattr(model, "prediction", None) is None:
            raise
        score = _score_classification(fold["y_test"], model.prediction, dataset.metric)
        return score, {dataset.metric: score, "fallback": "sklearn_from_predict_proba"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec_json).read_text(encoding="utf-8"))
    print(
        f"worker start {spec.get('source_dataset')} "
        f"timeout_min={spec.get('timeout_minutes')} "
        f"n_jobs={spec.get('n_jobs')} preset={spec.get('preset')} "
        f"seed={spec.get('seed')}",
        flush=True,
    )
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(stdout_handler)
    logging.getLogger().setLevel(logging.INFO)
    payload: dict
    try:
        from fedot.api.main import Fedot

        from fedotllm.agents.evolve.evaluation.openml_fold import load_registered_fold
        from fedotllm.agents.evolve.evaluation.quality_registry import get_dataset

        dataset = get_dataset(spec["source_dataset"])
        fold = load_registered_fold(dataset)
        print(
            f"worker fold loaded n_train={fold['n_train']} n_test={fold['n_test']} "
            f"loader={fold.get('loader')}; calling Fedot.fit",
            flush=True,
        )
        timeout_minutes = float(spec["timeout_minutes"])
        ts = dataset.problem == "ts_forecasting"
        extra = {}
        if ts:
            from fedot.core.repository.tasks import TsForecastingParams

            extra["task_params"] = TsForecastingParams(
                forecast_length=int(dataset.forecast_horizon or fold.get("forecast_horizon") or 12)
            )
        model = Fedot(
            problem=dataset.problem,
            timeout=timeout_minutes,
            seed=int(spec["seed"]),
            n_jobs=int(spec["n_jobs"]),
            preset=str(spec["preset"]),
            with_tuning=bool(spec["with_tuning"]),
            metric=dataset.metric,
            safe_mode=False,
            logging_level=logging.INFO,
            **extra,
        )
        if ts:
            model.fit(features=fold["X_train"])
        else:
            model.fit(features=fold["X_train"], target=fold["y_train"])
        predict_path = _predict_holdout(model, dataset, fold)
        score, metrics = _holdout_metrics(model, dataset, fold)
        log_text = log_stream.getvalue()
        search_ran, search_detail = _search_ran(model, log_text)
        status = "ok" if search_ran else "invalid"
        payload = {
            "task_id": dataset.task_id,
            "status": status,
            "score": score,
            "detail": search_detail if not search_ran else "",
            "n_train": fold["n_train"],
            "n_test": fold["n_test"],
            "seed": int(spec["seed"]),
            "metric_observations": {
                "search_ran": search_ran,
                "search_detail": search_detail,
                "was_optimised": bool(
                    getattr(getattr(model, "api_composer", None), "was_optimised", False)
                ),
                "was_tuned": bool(
                    getattr(getattr(model, "api_composer", None), "was_tuned", False)
                ),
                "history_generations": _history_generations(getattr(model, "history", None)),
                "metrics": metrics,
                "loader": fold.get("loader"),
                "preset": spec["preset"],
                "timeout_seconds": spec["timeout_seconds"],
                "n_jobs": spec["n_jobs"],
                "predict_path": predict_path,
            },
            "log_tail": log_text[-4000:],
        }
        if not search_ran:
            payload["traceback"] = ""
    except Exception as exc:
        payload = {
            "task_id": spec.get("source_dataset", ""),
            "status": "crash",
            "score": float("nan"),
            "traceback": tb_mod.format_exc(),
            "detail": f"{type(exc).__name__}: {exc}",
            "n_train": 0,
            "seed": int(spec.get("seed") or 42),
            "metric_observations": {"search_ran": False},
            "log_tail": log_stream.getvalue()[-4000:],
        }
    Path(args.out).write_text(json.dumps(payload, default=str), encoding="utf-8")
    return 0 if payload.get("status") != "crash" else 2


if __name__ == "__main__":
    raise SystemExit(main())
