"""Immutable outer holdouts and training-only resamples for metric-only runs.

Model seeds never select rows. The three CV validation folds are disjoint;
SHADOW, DEV and FINAL are excluded from every development training/CV fold.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

PROTOCOL = "independent-data-v2"
SPLIT_SEED = 42
FOLDS = (0, 1, 2)


def partition_indices(target, *, problem: str, external_target=None):
    target = np.asarray(target)

    def split(ids, labels, fraction):
        stratify = labels if problem in {"classification", "text"} else None
        return train_test_split(
            ids, test_size=fraction, random_state=SPLIT_SEED,
            shuffle=True, stratify=stratify,
        )

    if external_target is None:
        train, holdout = split(np.arange(len(target)), target, 0.4)
        dev, final = split(holdout, target[holdout], 0.5)
        holdout_namespace = "train"
    else:
        train = np.arange(len(target))
        external_target = np.asarray(external_target)
        dev, final = split(np.arange(len(external_target)), external_target, 0.5)
        holdout_namespace = "test"
    core, shadow = split(train, target[train], 0.2)
    return core, {
        "dev": (holdout_namespace, dev),
        "shadow": ("train", shadow),
        "final": (holdout_namespace, final),
    }


def select_indices(target, *, problem: str, split: str, fold=None, external_target=None):
    if split not in {"dev", "shadow", "final"}:
        raise ValueError(f"unknown split: {split}")
    if fold is not None and (split != "dev" or fold not in FOLDS):
        raise ValueError("resampling is restricted to DEV folds 0, 1, 2")
    core, holdouts = partition_indices(target, problem=problem, external_target=external_target)
    if fold is None:
        namespace, validation = holdouts[split]
        return core, namespace, validation, core
    labels = np.asarray(target)[core]
    classification = problem in {"classification", "text"}
    splitter = (
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED + 1)
        if classification else KFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED + 1)
    )
    if classification and min(np.unique(labels, return_counts=True)[1]) < 3:
        raise ValueError("insufficient class support for three training-only folds")
    train, validation = list(splitter.split(core, labels))[fold]
    return core[train], "train", core[validation], core


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def evidence(*, data_hash, train, namespace, validation, core, split, fold, kind="tabular"):
    train_ids = [f"train:{int(i)}" for i in train]
    validation_ids = [f"{namespace}:{int(i)}" for i in validation]
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation rows overlap")
    return {
        "protocol": PROTOCOL, "kind": kind, "split": split, "fold": fold,
        "data_hash": data_hash, "core_hash": _digest(list(map(int, core))),
        "train_hash": _digest(train_ids), "n_train": len(train_ids),
        "validation_hash": _digest(validation_ids), "validation_ids": validation_ids,
        "n_validation": len(validation_ids), "train_validation_disjoint": True,
    }


def table_split(features, target, *, problem, split, fold=None, external=None):
    external_target = external[1] if external is not None else None
    train, namespace, validation, core = select_indices(
        target, problem=problem, split=split, fold=fold, external_target=external_target,
    )
    vx, vy = external if namespace == "test" else (features, target)
    data_hash = _digest([
        np.asarray(features).tolist(), np.asarray(target).tolist(),
        None if external is None else [np.asarray(a).tolist() for a in external],
    ])
    metadata = evidence(
        data_hash=data_hash, train=train, namespace=namespace, validation=validation,
        core=core, split=split, fold=fold,
    )
    return features[train], target[train], vx[validation], vy[validation], metadata


def ts_indices(length: int, horizon: int, *, split: str, fold=None):
    if split not in {"dev", "shadow", "final"}:
        raise ValueError(f"unknown split: {split}")
    if fold is not None and (split != "dev" or fold not in FOLDS):
        raise ValueError("rolling-origin folds are restricted to DEV")
    # Chronological held-out windows; earlier observations may train later
    # forecasts, never vice versa. CV origins precede all three outer windows.
    offset = {"final": 0, "shadow": 1, "dev": 2}[split] if fold is None else 3 + fold
    end = length - offset * horizon
    if end <= 2 * horizon:
        raise ValueError("insufficient history for frozen rolling-origin windows")
    return np.arange(end - horizon), np.arange(end - horizon, end)
