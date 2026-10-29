"""Subprocess worker. Invoked with PYTHONPATH=fedot_checkout:repo_root."""

from __future__ import annotations

import argparse
import ast
import json
import time
from collections.abc import Mapping
from pathlib import Path


def _shape(value) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def _index_summary(value, *, width: int | None) -> dict | None:
    if value is None:
        return None
    try:
        import numpy as np

        items = np.asarray(value).astype(int, copy=False).ravel()
    except (TypeError, ValueError):
        return {"count": None, "min": None, "max": None, "within_width": False}
    minimum = int(items.min()) if items.size else None
    maximum = int(items.max()) if items.size else None
    valid = bool(
        width is not None
        and (items.size == 0 or (minimum is not None and minimum >= 0 and maximum < width))
    )
    return {
        "count": int(items.size),
        "min": minimum,
        "max": maximum,
        "within_width": valid,
    }


def _data_snapshot(data, *, output: bool = False) -> dict:
    features_shape = _shape(getattr(data, "features", None))
    predict_shape = _shape(getattr(data, "predict", None)) if output else None
    active_shape = predict_shape if output and predict_shape is not None else features_shape
    width = None
    if active_shape:
        width = active_shape[-1] if len(active_shape) > 1 else 1
    numerical = _index_summary(getattr(data, "numerical_idx", None), width=width)
    categorical = _index_summary(getattr(data, "categorical_idx", None), width=width)
    encoded = _index_summary(getattr(data, "encoded_idx", None), width=width)
    summaries = [item for item in (numerical, categorical, encoded) if item is not None]
    return {
        "features_shape": features_shape,
        "predict_shape": predict_shape,
        "active_width": width,
        "numerical_idx": numerical,
        "categorical_idx": categorical,
        "encoded_idx": encoded,
        "metadata_within_width": all(bool(item.get("within_width")) for item in summaries),
    }


def _public_params(value) -> dict:
    """Keep only small JSON-safe operation parameters in the runtime trace."""

    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict()
        except Exception:
            return {}
    if not isinstance(value, Mapping):
        return {}
    out: dict = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
        elif isinstance(item, (list, tuple)) and len(item) <= 12 and all(
            isinstance(part, (str, int, float, bool)) or part is None for part in item
        ):
            out[str(key)] = list(item)
    return out


def _parameter_surface(value) -> tuple[list[str], dict]:
    """Read the concrete estimator surface without changing its state."""

    candidates = [value]
    for attr in ("model", "estimator", "operation"):
        nested = getattr(value, attr, None)
        if nested is not None and nested is not value:
            candidates.append(nested)
    for candidate in candidates:
        get_params = getattr(candidate, "get_params", None)
        if callable(get_params):
            try:
                raw = get_params(deep=False)
                if isinstance(raw, dict) and raw:
                    return sorted(str(name) for name in raw)[:160], _public_params(raw)
            except Exception:
                pass
        # CatBoost exposes the complete effective fitted surface separately.
        get_all_params = getattr(candidate, "get_all_params", None)
        if callable(get_all_params):
            try:
                raw = get_all_params()
                if isinstance(raw, dict) and raw:
                    return sorted(str(name) for name in raw)[:160], _public_params(raw)
            except Exception:
                pass
    return [], {}


def _line_ranges(lines: set[int]) -> list[list[int]]:
    """Compress exact coverage lines without losing branch reachability."""

    ordered = sorted(lines)
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for line in ordered[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append([start, previous])
        start = previous = line
    ranges.append([start, previous])
    return ranges


def _qualified_symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[str, str]:
    if isinstance(node, ast.ClassDef):
        return node.name, "class"
    owner = parents.get(node)
    while owner is not None and not isinstance(owner, ast.ClassDef):
        owner = parents.get(owner)
    if isinstance(owner, ast.ClassDef):
        return f"{owner.name}.{node.name}", "method"
    return str(getattr(node, "name", "<symbol>")), "function"


def _body_was_executed(node: ast.AST, executed: set[int]) -> bool:
    """Distinguish an imported definition from a function that actually ran.

    coverage.py marks a ``def`` line when Python creates the function object during
    module import.  That is useful reachability evidence, but it is not evidence
    that the function can affect the measured metric.  A body statement hit is.
    """

    for statement in getattr(node, "body", ()):
        first = int(getattr(statement, "lineno", 0) or 0)
        last = int(getattr(statement, "end_lineno", first) or first)
        if any(first <= line <= last for line in executed):
            return True
    return False


def _install_dataflow_trace(rows: list[dict]) -> None:
    """Observe FEDOT operation boundaries without exposing feature/target values."""

    from fedot.core.operations.operation import Operation

    original = Operation._predict

    def traced(
        self,
        fitted_operation,
        data,
        params=None,
        output_mode="default",
        is_fit_stage=False,
        predictions_cache=None,
        fold_id=None,
        descriptive_id=None,
    ):
        supported_parameters, estimator_defaults = _parameter_surface(fitted_operation)
        row = {
            "operation": str(getattr(self, "operation_type", type(self).__name__)),
            "implementation": type(fitted_operation).__name__,
            "params": _public_params(
                getattr(fitted_operation, "params", None) or params
            ),
            "supported_parameters": supported_parameters,
            "estimator_defaults": estimator_defaults,
            "stage": "fit" if is_fit_stage else "predict",
            "input": _data_snapshot(data),
        }
        try:
            output = original(
                self,
                fitted_operation,
                data,
                params,
                output_mode,
                is_fit_stage,
                predictions_cache,
                fold_id,
                descriptive_id,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:240]
            rows.append(row)
            raise
        if not supported_parameters:
            # FEDOT CatBoost/LightGBM wrappers create their concrete estimator
            # during fit. Inspect it only after the original operation returns.
            supported_parameters, estimator_defaults = _parameter_surface(fitted_operation)
            row["supported_parameters"] = supported_parameters
            row["estimator_defaults"] = estimator_defaults
        row["output"] = _data_snapshot(output, output=True)
        rows.append(row)
        return output

    Operation._predict = traced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("shadow", "dev", "final"), default="dev")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--task-override-json", default="")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    checkout = args.checkout.resolve()
    cov = None
    dataflow: list[dict] = []
    if args.coverage:
        from coverage import Coverage

        cov = Coverage(source=[str(checkout / "fedot")], data_file=None, branch=False)
        cov.start()
        _install_dataflow_trace(dataflow)
    try:
        from fedotllm.agents.evolve.evaluation.scorer import score_task

        task_override = (
            json.loads(args.task_override_json) if args.task_override_json else None
        )
        payload = score_task(
            args.task,
            seed=args.seed,
            split_name=args.split,
            task_override=task_override,
        )
        payload["seed"] = args.seed
    except Exception as exc:
        import traceback

        payload = {
            "task_id": args.task,
            "status": "invalid",
            "score": float("nan"),
            "traceback": traceback.format_exc(),
            "detail": f"{type(exc).__name__}: {exc}",
            "n_train": 0,
            "seed": args.seed,
        }
    finally:
        if cov is not None:
            cov.stop()
    if cov is not None:
        rows: list[dict] = []
        data = cov.get_data()
        for filename in data.measured_files():
            path = Path(filename).resolve()
            try:
                rel = path.relative_to(checkout).as_posix()
            except (OSError, ValueError):
                continue
            if not rel.startswith("fedot/") or not path.is_file():
                continue
            executed = set(data.lines(filename) or ())
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
                symbol_lines = {
                    line for line in executed if node.lineno <= line <= end
                }
                if symbol_lines:
                    symbol, kind = _qualified_symbol(node, parents)
                    rows.append(
                        {
                            "file_path": rel,
                            "symbol": symbol,
                            "kind": kind,
                            "line": int(node.lineno),
                            "count": len(symbol_lines),
                            "line_ranges": _line_ranges(symbol_lines),
                            "body_executed": _body_was_executed(node, executed),
                        }
                    )
        payload["coverage"] = sorted(
            rows,
            key=lambda row: (-int(row["count"]), row["file_path"], int(row["line"])),
        )[:1000]
        payload["dataflow"] = dataflow[-200:]
    payload["duration_s"] = time.perf_counter() - started
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0 if payload.get("status") != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
