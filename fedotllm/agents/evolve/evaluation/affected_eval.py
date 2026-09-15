"""Controller-owned metric check for a branch proven by a public reproduction.

The verifier may prove a defect with explicit operation parameters that the broad
frozen suite never uses.  This module extracts only literal public
``PipelineBuilder.add_node(..., params=...)`` calls, applies those parameters to
an existing frozen real-data workload for the same operation, and confirms that
the suspicious source line was actually executed.  The LLM never receives the
dataset or scorer implementation, and this diagnostic cannot by itself authorize
a primary DEV/FINAL KEEP.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from fedotllm.agents.evolve.evaluation.eval import run_patched, run_stock
from fedotllm.agents.evolve.evaluation.tasks import all_tasks, load_task
from fedotllm.agents.evolve.types import PatchSite, VerificationResult


def _json_literal(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 24:
        return [_json_literal(item) for item in value]
    if isinstance(value, dict) and len(value) <= 24:
        if not all(isinstance(key, str) for key in value):
            raise ValueError("operation parameter keys must be strings")
        return {str(key): _json_literal(item) for key, item in value.items()}
    raise ValueError(f"unsupported literal parameter type: {type(value).__name__}")


def extract_operation_overlays(code: str) -> dict[str, dict]:
    """Extract literal operation parameters without executing model-authored code."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {}
    overlays: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_node":
            continue
        operation_node = node.args[0] if node.args else None
        if operation_node is None:
            operation_node = next(
                (item.value for item in node.keywords if item.arg in {"operation_type", "operation"}),
                None,
            )
        params_node = next(
            (item.value for item in node.keywords if item.arg == "params"), None
        )
        if operation_node is None or params_node is None:
            continue
        try:
            operation = ast.literal_eval(operation_node)
            params = _json_literal(ast.literal_eval(params_node))
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(operation, str) and isinstance(params, dict) and params:
            overlays[operation] = params
    return overlays


def _line_reached(result, lead: PatchSite) -> bool:
    target = lead.file_path.replace("\\", "/").lstrip("/")
    for row in result.coverage:
        file_path = str(row.get("file_path") or "").replace("\\", "/")
        if file_path != target and not file_path.endswith("/" + target):
            continue
        for start, end in row.get("line_ranges") or ():
            if int(start) <= int(lead.line) <= int(end):
                return True
    return False


def _task_override(spec, operation: str, params: dict) -> dict:
    override: dict = {"operation_params": {operation: params}}
    if spec.problem == "ts":
        # Five forecast windows leave enough post-lag rows for a downstream
        # estimator while keeping oversize/edge parameters reachable.  The
        # values still come from the frozen real series, never from the
        # reproduction.
        horizon = int(spec.forecast_horizon or 12)
        override["history_size"] = max(5 * horizon, 60)
    return override


def _normalized_delta(task_id: str, stock, patched) -> tuple[str, float | None]:
    spec = load_task(task_id)
    if stock.status == "ok" and patched.status == "crash":
        return "regressed", None
    if stock.status == "crash" and patched.status == "ok":
        return "improved", None
    if stock.status != "ok" or patched.status != "ok":
        return "invalid", None
    raw = (
        float(patched.score) - float(stock.score)
        if spec.higher_is_better
        else float(stock.score) - float(patched.score)
    )
    delta = (
        raw / max(abs(float(stock.score)), 1e-12)
        if spec.min_delta_mode == "relative"
        else raw
    )
    if delta >= spec.min_delta:
        return "improved", delta
    if delta <= -spec.min_delta:
        return "regressed", delta
    return "neutral", delta


def evaluate_affected_metric(
    source: Path,
    experiment: Path,
    verification: VerificationResult,
    lead: PatchSite,
    *,
    seed: int = 42,
    split: str = "dev",
    max_tasks: int = 2,
) -> dict:
    """Measure a verified branch on frozen data, returning a trace-safe record."""

    if verification.status != "verified_bug":
        return {"status": "not_applicable", "rows": []}
    overlays = extract_operation_overlays(verification.reproduction_code)
    if not overlays:
        return {"status": "unavailable", "reason": "no_literal_operation_params", "rows": []}

    catalog = all_tasks()
    eligible = [
        name for name in overlays if any(name in spec.nodes for spec in catalog)
    ][:4]
    if not eligible:
        return {
            "status": "unavailable",
            "reason": "reproduction_operation_has_no_frozen_workload",
            "operations": sorted(overlays),
            "rows": [],
        }
    wants_ts = "ts_forecasting" in verification.reproduction_code or "TsForecastingParams" in verification.reproduction_code
    attempted_rows: list[dict] = []
    operation = ""
    rows: list[dict] = []
    for name in eligible:
        specs = [spec for spec in catalog if name in spec.nodes]
        if wants_ts:
            matching = [spec for spec in specs if spec.problem == "ts"]
            specs = matching or specs
        operation_rows: list[dict] = []
        for spec in specs[: max(1, max_tasks)]:
            override = _task_override(spec, name, overlays[name])
            stock = run_stock(
                spec.task_id,
                checkout=source,
                split=split,
                seed=seed,
                collect_coverage=True,
                task_override=override,
            )
            reached = _line_reached(stock, lead)
            if reached:
                patched = run_patched(
                    spec.task_id,
                    checkout=experiment,
                    split=split,
                    seed=seed,
                    task_override=override,
                )
                classification, delta = _normalized_delta(
                    spec.task_id, stock, patched
                )
                patched_row = {"status": patched.status, "score": patched.score}
            else:
                classification, delta = "not_reached", None
                patched_row = {"status": "not_run", "score": None}
            operation_rows.append(
                {
                    "task_id": spec.task_id,
                    "problem": spec.problem,
                    "metric": spec.metric,
                    "override": override,
                    "lead_reached": reached,
                    "stock": {"status": stock.status, "score": stock.score},
                    "patched": patched_row,
                    "normalized_delta": delta,
                    "classification": classification,
                }
            )
        attempted_rows.extend(operation_rows)
        if any(row["lead_reached"] for row in operation_rows):
            operation = name
            rows = operation_rows
            break
    if not operation:
        rows = attempted_rows
    reached_rows = [row for row in rows if row["lead_reached"]]
    if not reached_rows:
        status = "not_reached"
    elif any(row["classification"] == "regressed" for row in reached_rows):
        status = "regressed"
    elif any(row["classification"] == "improved" for row in reached_rows):
        status = "improved"
    elif any(row["classification"] == "invalid" for row in reached_rows):
        status = "invalid"
    else:
        status = "neutral"
    return {
        "status": status,
        "operation": operation,
        "params": overlays.get(operation, {}),
        "operations_considered": eligible,
        "seed": seed,
        "split": split,
        "authority": "secondary_diagnostic_only",
        "rows": rows,
    }


def confirm_affected_metric(
    source: Path,
    experiment: Path,
    verification: VerificationResult,
    lead: PatchSite,
    *,
    seeds: tuple[int, ...] = (42, 43, 44),
    split: str = "dev",
    initial: dict | None = None,
    max_tasks: int = 2,
) -> dict:
    """Apply the ordinary no-regression confirmation rule to an affected suite."""

    runs: list[dict] = []
    for seed in seeds:
        if (
            initial
            and int(initial.get("seed", -1)) == int(seed)
            and initial.get("split", "dev") == split
        ):
            # The controller attaches this confirmation payload back onto
            # ``initial``. Keeping the same object here would make
            # initial["dev_confirmation"]["runs"][0] point to initial and
            # crash artifact persistence with "Circular reference detected".
            run = copy.deepcopy(initial)
        else:
            run = evaluate_affected_metric(
                source,
                experiment,
                verification,
                lead,
                seed=seed,
                split=split,
                max_tasks=max_tasks,
            )
        runs.append(run)
    improved_seeds = sum(run.get("status") == "improved" for run in runs)
    regressed_pairs = sum(
        row.get("classification") == "regressed"
        for run in runs
        for row in run.get("rows") or ()
        if row.get("lead_reached")
    )
    infrastructure_failures = sum(
        run.get("status") in {"invalid", "unavailable", "not_reached"}
        for run in runs
    )
    confirmed = bool(
        improved_seeds >= 2
        and regressed_pairs == 0
        and infrastructure_failures == 0
    )
    return {
        "confirmed": confirmed,
        "split": split,
        "seeds": list(seeds),
        "improved_seeds": improved_seeds,
        "regressed_task_seed_pairs": regressed_pairs,
        "infrastructure_failures": infrastructure_failures,
        "runs": runs,
    }


def affected_feedback(result: dict) -> str:
    """Return useful causal feedback without exposing frozen dataset identities."""

    compact_rows = [
        {
            "task_family": row.get("problem"),
            "metric": row.get("metric"),
            "lead_reached": row.get("lead_reached"),
            "classification": row.get("classification"),
            "normalized_delta": row.get("normalized_delta"),
        }
        for row in result.get("rows") or ()
    ]
    return (
        f"affected_metric_status={result.get('status')}; "
        f"operation={result.get('operation')}; params={result.get('params')}; "
        f"results={compact_rows}"
    )
