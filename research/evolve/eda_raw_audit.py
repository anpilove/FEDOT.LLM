"""Audit the raw OpenML tables used by the frozen EvolveAgent registry.

Only aggregate results are written to the repository. OpenML caches raw data
outside the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openml
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVOLVE = ROOT / "fedotllm" / "agents" / "evolve"
OUTPUT = EVOLVE / "docs" / "eda_assets" / "raw_audit.csv"


def audit_task(spec: dict) -> dict:
    task = openml.tasks.get_task(spec["openml_task"])
    dataset = task.get_dataset()
    features, target, _, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    train, test = task.get_train_test_split_indices(repeat=0, fold=0)
    fingerprints = pd.util.hash_pandas_object(
        features.astype("string").fillna("<missing>"), index=False
    )
    train_fingerprints = set(fingerprints.iloc[train])
    row = {
        "task_id": spec["task_id"],
        "openml_task": spec["openml_task"],
        "openml_data": spec["openml_data"],
        "problem": spec["problem"],
        "rows": len(features),
        "input_features": features.shape[1],
        "missing_input_cells": int(features.isna().sum().sum()),
        "rows_with_missing_input": int(features.isna().any(axis=1).sum()),
        "missing_target_rows": int(pd.isna(target).sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_rows_seen_in_train": int(fingerprints.iloc[test].isin(train_fingerprints).sum()),
        "target_unique": int(pd.Series(target).nunique(dropna=True)),
    }
    if spec["problem"] == "classification":
        counts = pd.Series(target).value_counts(dropna=True)
        row["smallest_class_share_percent"] = round(100 * counts.min() / counts.sum(), 3)
    else:
        numeric = pd.to_numeric(target, errors="coerce")
        row["target_min"] = float(numeric.min())
        row["target_median"] = float(numeric.median())
        row["target_max"] = float(numeric.max())
    return row


def main() -> None:
    registry = json.loads((EVOLVE / "evaluation" / "quality_registry.json").read_text())
    specs = [item for item in registry["datasets"] if item["source"] == "openml_task"]
    rows = []
    for spec in specs:
        row = audit_task(spec)
        rows.append(row)
        print(row["task_id"], row["rows"], row["test_rows_seen_in_train"], flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT, index=False)
    top = frame.nlargest(10, "test_rows_seen_in_train").sort_values(
        "test_rows_seen_in_train"
    )
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(top["task_id"], top["test_rows_seen_in_train"], color="#c15d4d")
    ax.set(
        title="Совпадающие признаки в train и test, fold 0",
        xlabel="Строки test с совпадением в train",
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"Audited {len(rows)} tasks; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
