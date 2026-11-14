"""Rebuild the small EDA artifacts for the frozen EvolveAgent task registry.

Usage: python research/evolve/hw2_eda.py --fedot-checkout /path/to/FEDOT
OpenML qualities are a dated metadata snapshot; raw OpenML tables are not stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVOLVE = ROOT / "fedotllm" / "agents" / "evolve"
OUTPUT = EVOLVE / "docs" / "hw2_eda"


def number(value: object) -> float:
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedot-checkout", type=Path, required=True)
    parser.add_argument("--raw-samples", action="store_true",
                        help="Download two small OpenML tables for target checks")
    args = parser.parse_args()
    registry = json.loads((EVOLVE / "evaluation" / "quality_registry.json").read_text())
    metadata = json.loads((OUTPUT / "openml_qualities.json").read_text())
    by_id = {int(row["openml_data"]): row for row in metadata["datasets"]}
    rows = []
    series = []

    for task in registry["datasets"]:
        row = {
            "task_id": task["task_id"], "name": task["name"],
            "problem": task["problem"], "source": task["source"],
            "metric": task["metric"], "rows": np.nan, "columns_reported": np.nan,
            "missing_values": np.nan, "missing_percent": np.nan,
            "minority_percent": np.nan, "forecast_horizon": np.nan,
            "checksum_ok": "", "metadata_source": "",
        }
        if task["source"] == "openml_task":
            details = by_id[int(task["openml_data"])]
            q = details["qualities"]
            for target, key in [
                ("rows", "NumberOfInstances"),
                ("columns_reported", "NumberOfFeatures"),
                ("missing_values", "NumberOfMissingValues"),
                ("missing_percent", "PercentageOfMissingValues"),
            ]:
                row[target] = number(q.get(key))
            minority = number(q.get("MinorityClassSize"))
            if pd.notna(minority) and row["rows"] > 0:
                row["minority_percent"] = 100 * minority / row["rows"]
            row["metadata_source"] = f"https://www.openml.org/d/{task['openml_data']}"
        else:
            if task["source"] == "fedot_public_ts":
                path = args.fedot_checkout / "examples/data/ts" / f"{task['ts_dataset']}.csv"
            else:
                path = (args.fedot_checkout / "examples/real_cases/data"
                        / task["data_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != task["data_sha256"]:
                raise ValueError(f"checksum mismatch for {task['task_id']}: {path}")
            frame = pd.read_csv(path)
            if task["source"] == "fedot_local_ts":
                time_column = next((column for column in frame.columns if
                    str(column).strip().lower() in {"datetime", "timestamp", "date", "time"}), None)
                if time_column is not None:
                    stamp = frame[time_column].astype(str).str.strip().str.removesuffix(".0")
                    if bool(stamp.str.fullmatch(r"\d{9,10}").all()):
                        day = pd.to_datetime(stamp.str[:8], format="%Y%m%d", errors="raise")
                        hour = pd.to_numeric(stamp.str[8:], errors="raise")
                        timestamps = day + pd.to_timedelta(hour, unit="h")
                    else:
                        timestamps = pd.to_datetime(stamp, errors="raise")
                    frame = frame.iloc[np.argsort(timestamps.to_numpy(), kind="stable")]
            target = task.get("target") or "value"
            values = pd.to_numeric(frame[target], errors="coerce")
            row.update(
                rows=len(frame), columns_reported=len(frame.columns),
                missing_values=int(values.isna().sum()),
                missing_percent=100 * values.isna().mean(),
                forecast_horizon=task["forecast_horizon"],
                checksum_ok="yes", metadata_source=str(path.relative_to(args.fedot_checkout)),
            )
            series.append((task["name"], values.dropna().to_numpy(), task["forecast_horizon"]))
        rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "task_inventory.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    table["problem"].value_counts().reindex(
        ["classification", "regression", "ts_forecasting"]
    ).plot.bar(ax=axes[0], color="#617cae")
    axes[0].set(title="Состав проверки", ylabel="Число задач", xlabel="Тип задачи")
    axes[0].tick_params(axis="x", rotation=20)
    openml = table[table.source == "openml_task"]
    for problem, group in openml.groupby("problem"):
        axes[1].scatter(group.rows, group.columns_reported, label=problem, alpha=.8)
    axes[1].set(xscale="log", yscale="log", xlabel="Строки", ylabel="Столбцы по OpenML",
                title="Объём задач OpenML")
    axes[1].legend()
    fig.savefig(OUTPUT / "overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    for ax, (name, values, horizon) in zip(axes.flat, series):
        ax.plot(values, linewidth=.7, color="#617cae")
        ax.plot(np.arange(len(values) - horizon, len(values)), values[-horizon:],
                linewidth=1.5, color="#c15d4d")
        ax.set_title(f"{name}: {len(values)} значений, горизонт {horizon}", fontsize=9)
    fig.savefig(OUTPUT / "time_series.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    missing = openml[openml.missing_percent > 0].sort_values("missing_percent")
    axes[0].barh(missing.name, missing.missing_percent, color="#c15d4d")
    axes[0].set(title="Неполные таблицы OpenML", xlabel="Пропущенные значения, %")
    classes = openml[openml.problem == "classification"].sort_values(
        "minority_percent"
    ).head(12)
    axes[1].barh(classes.name, classes.minority_percent, color="#617cae")
    axes[1].set(title="Наименьшие классы", xlabel="Доля строк, %")
    fig.savefig(OUTPUT / "data_quality.png", dpi=160)
    plt.close(fig)

    if args.raw_samples:
        import openml as openml_api

        samples = []
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for data_id, ax in ((31, axes[0]), (41021, axes[1])):
            dataset = openml_api.datasets.get_dataset(data_id, download_data=True)
            features, target, _, _ = dataset.get_data(
                target=dataset.default_target_attribute, dataset_format="dataframe"
            )
            record = {
                "openml_data": data_id, "name": dataset.name,
                "rows": len(features), "input_features": features.shape[1],
                "missing_input_cells": int(features.isna().sum().sum()),
                "missing_target_rows": int(pd.isna(target).sum()),
            }
            if data_id == 31:
                counts = pd.Series(target).value_counts()
                record["class_counts"] = {str(k): int(v) for k, v in counts.items()}
                counts.plot.bar(ax=ax, color="#617cae")
                ax.set(title="credit-g: баланс классов", xlabel="Класс", ylabel="Строки")
                ax.tick_params(axis="x", rotation=0)
            else:
                numeric = pd.to_numeric(target, errors="coerce")
                record["target_quantiles"] = {
                    str(k): float(v) for k, v in numeric.quantile([0, .25, .5, .75, 1]).items()
                }
                numeric.hist(ax=ax, bins=25, color="#c15d4d")
                ax.set(title="Moneyball: целевая переменная RS", xlabel="RS", ylabel="Строки")
            samples.append(record)
        fig.savefig(OUTPUT / "raw_samples.png", dpi=160)
        plt.close(fig)
        (OUTPUT / "raw_samples.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n"
        )

    print("tasks", len(table), "openml", len(openml), "time_series", len(series))
    print("openml rows range", int(openml.rows.min()), int(openml.rows.max()))
    print("openml columns range", int(openml.columns_reported.min()), int(openml.columns_reported.max()))
    print("openml missing datasets", int((openml.missing_values > 0).sum()))
    print("openml max missing percent", round(openml.missing_percent.max(), 2))
    print("ts lengths", int(table.loc[table.problem == "ts_forecasting", "rows"].min()),
          int(table.loc[table.problem == "ts_forecasting", "rows"].max()))
    print("ts missing", int(table.loc[table.problem == "ts_forecasting", "missing_values"].sum()))
    print("smallest minority share", openml.loc[
        openml.minority_percent.idxmin(), ["task_id", "minority_percent"]
    ].to_dict())


if __name__ == "__main__":
    main()
