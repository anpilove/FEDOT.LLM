"""Controller-owned cross-dataset evidence; repetitions never create datasets.

The acceptance policy is a conservative engineering gate on this benchmark,
not a claim about every possible dataset or a statistical power guarantee.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from fedotllm.agents.evolve.evaluation.affected_eval import _line_reached
from fedotllm.agents.evolve.evaluation.eval import run_stock, run_patched
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL
from fedotllm.agents.evolve.evaluation.tasks import all_tasks, load_task
from fedotllm.agents.evolve.types import MatchSite

TRANSFER_PROTOCOL = "benchmark-transfer-v1"


def dataset_tasks(target):
    """All compatible registered sources, independent of suite/LLM filtering."""
    spec = load_task(target)
    sources = {}
    for data in sorted(all_tasks(), key=lambda item: item.task_id):
        if (data.problem, data.metric) != (spec.problem, spec.metric):
            continue
        # Columns, model variants and renamed datasets from one source do not
        # count as independent datasets. Content hashes deduplicate copies later.
        files = (data.train_file, data.test_file)
        if data.dataset == "scoring":
            files = ("scoring/scoring_train.csv", "scoring/scoring_test.csv")
        key = tuple(sorted(name for name in files if name)) or (data.dataset,)
        sources.setdefault(key, data.task_id)
    return list(sources.values())


def _override(target, params, fold=None):
    result = {"evaluation_protocol": PROTOCOL, "pipeline_task": target,
              "operation_params": params or {}}
    if fold is not None:
        result["resample_fold"] = fold
    return result


def scenario_tasks(target):
    """Predeclare registered input variants of the same pipeline, not winners."""
    primary = load_task(target)
    fields = ("kind", "nodes", "left", "right", "join", "tail")
    structure = tuple(getattr(primary, key) for key in fields)
    selected = {}
    for spec in [primary, *sorted(all_tasks(), key=lambda item: item.task_id)]:
        if (spec.problem, spec.metric) != (primary.problem, primary.metric):
            continue
        if tuple(getattr(spec, key) for key in fields) == structure:
            selected.setdefault(spec.index_offset, spec.task_id)
    return list(selected.values())


_JSON_OPERATION = re.compile(r'^\s*"([^"\n]+)"\s*:\s*\{')


def _configuration_operation_at_lead(source, lead: MatchSite) -> str | None:
    """Resolve the JSON operation owning a planned metadata line."""
    if source is None or not lead.file_path.endswith(".json"):
        return None
    path = Path(source) / lead.file_path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines[: max(0, int(lead.line))]):
        match = _JSON_OPERATION.match(line)
        if match:
            return match.group(1)
    return None


def _crash_at_lead(result, lead: MatchSite) -> bool:
    """A crash can prove reachability when coverage stops at the exception."""
    target = lead.file_path.replace("\\", "/").lstrip("/")
    trace = str(getattr(result, "traceback", "") or "").replace("\\", "/")
    return target in trace and (f":{int(lead.line)}" in trace or f"line {int(lead.line)}" in trace)


def _screen_hashes(rows, count: int) -> list[str]:
    """Pick static task-metadata quantiles, never a score-based subset."""
    sources = {}
    for row in rows:
        if not row.get("data_hash"):
            continue
        spec = load_task(row["task"])
        sources.setdefault(
            row["data_hash"],
            (spec.problem, spec.forecast_horizon, spec.history_size, spec.dataset, row["data_hash"]),
        )
    ordered = [data_hash for data_hash, _ in sorted(sources.items(), key=lambda item: item[1])]
    if len(ordered) <= count:
        return ordered
    # Endpoints plus evenly spaced interior sources preserve the range of the
    # registered benchmark without making the choice depend on patch scores.
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in dict.fromkeys(positions)]


def preregister_transfer(source, target, lead, params, *, screen_sources: int = 3):
    if screen_sources < 2:
        raise ValueError("transfer screen requires at least two sources")
    rows = []
    configuration_operation = _configuration_operation_at_lead(source, lead)
    if target is not None:
        for scenario in scenario_tasks(target):
            for task in dataset_tasks(target):
                baseline = run_stock(task, checkout=source, split="dev", seed=42,
                                     collect_coverage=True, task_override=_override(scenario, params))
                failure_at_lead = baseline.status == "crash" and _crash_at_lead(baseline, lead)
                config_reached = bool(configuration_operation) and any(
                    flow.get("operation") == configuration_operation
                    for flow in baseline.dataflow
                )
                rows.append({"task": task, "scenario": scenario, "dataset": load_task(task).dataset,
                             "data_hash": baseline.data_evidence.get("data_hash"),
                             "baseline_status": baseline.status,
                             "baseline_detail": baseline.detail,
                             "baseline_duration_seconds": baseline.duration_s,
                             # A traced exception is causal reachability: coverage cannot
                             # continue past the failing line, so requiring a normal return
                             # would reject every genuine crash-recovery patch.
                             "baseline_failure_at_lead": failure_at_lead,
                             "configuration_operation": configuration_operation,
                             "configuration_reached": config_reached,
                             "affected": _line_reached(baseline, lead) or failure_at_lead or config_reached})
    data_hashes = _screen_hashes(rows, screen_sources) if rows else []
    return {"protocol": TRANSFER_PROTOCOL, "pipeline_task": target,
            "operation_params": params or {},
            "lead": asdict(lead), "datasets": rows,
            "screen_data_hashes": data_hashes,
            "screen_selection": "static_task_metadata_quantiles",
            "policy": "screen_is_not_acceptance; full_sources_no_regressions; practical_gain_on_strict_majority_of_affected_sources_and_at_least_two"}


def validate_transfer(plan):
    if not plan or plan.get("protocol") != TRANSFER_PROTOCOL:
        return "transfer_plan_missing_or_stale"
    if not plan.get("pipeline_task"):
        return "transfer_no_covered_pipeline"
    rows = plan.get("datasets", [])
    expected = [(scenario, task) for scenario in scenario_tasks(plan["pipeline_task"])
                for task in dataset_tasks(plan["pipeline_task"])]
    if [(row.get("scenario"), row["task"]) for row in rows] != expected:
        return "transfer_dataset_inventory_changed"
    for row in rows:
        status = row.get("baseline_status")
        if not row.get("data_hash"):
            return "transfer_baseline_incomplete"
        if status == "ok":
            continue
        # A stock crash is admissible only when its traceback names the planned
        # source line. Other crashes remain infrastructure/incomplete evidence.
        if status != "crash" or not row.get("baseline_failure_at_lead"):
            return "transfer_baseline_incomplete"
    affected = {row["data_hash"] for row in rows if row["affected"]}
    if len(affected) < 2:
        return "transfer_insufficient_affected_datasets"
    screen = plan.get("screen_data_hashes")
    if not isinstance(screen, list) or len(screen) < 2 or not set(screen) <= {row["data_hash"] for row in rows}:
        return "transfer_screen_inventory_invalid"
    return None


def evaluate_transfer(source, experiment, plan, *, params=None, split, fold=None, seed=42,
                      practical=True, comparisons=3, scope="full", budget=None):
    """Measure every registered source; no cherry-picking after patched scores."""
    from fedotllm.agents.evolve.controller.metric_study import judge_pair
    from fedotllm.agents.evolve.evaluation.uncertainty import paired_interval

    error = validate_transfer(plan)
    report = {"protocol": TRANSFER_PROTOCOL, "split": split, "fold": fold,
              "seed": seed, "datasets": [], "scope": scope}
    if error:
        return False, {**report, "reason": error}
    if (params or {}) != plan.get("operation_params", {}):
        return False, {**report, "reason": "transfer_parameters_changed"}
    if scope not in {"screen", "full"}:
        return False, {**report, "reason": "transfer_unknown_scope"}
    selected_hashes = (
        set(plan["screen_data_hashes"]) if scope == "screen"
        else {row["data_hash"] for row in plan["datasets"]}
    )
    selected_rows = [row for row in plan["datasets"] if row["data_hash"] in selected_hashes]
    # FINAL pairs must draw on the reserve that MeasurementBudget keeps for
    # them; charging them as "full" would let the reserve block FINAL itself.
    budget_stage = "final" if split == "final" else scope
    lead = MatchSite(**plan["lead"])
    affected_hashes = {row["data_hash"] for row in selected_rows if row["affected"]}
    required = max(2, len(affected_hashes) // 2 + 1)
    scenario_wins = {}
    failures = []
    for row in selected_rows:
        task = row["task"]
        target = row["scenario"]
        override = _override(target, params, fold)
        started = budget.begin(budget_stage) if budget is not None else None
        if budget is not None and started is None:
            report.update(affected_dataset_count=len(affected_hashes), failures=failures,
                          measurement_budget=budget.snapshot(), reason="inconclusive_budget")
            return False, report
        try:
            before = run_stock(task, checkout=source, split=split, seed=seed,
                               collect_coverage=True, task_override=override)
            after = run_patched(task, checkout=experiment, split=split, seed=seed,
                                collect_coverage=True, task_override=override)
        finally:
            if budget is not None:
                budget.finish(started)
        recovered_crash = (
            before.status == "crash" and after.status == "ok"
            and row.get("baseline_failure_at_lead") and _crash_at_lead(before, lead)
        )
        unchanged_crash = (
            before.status == after.status == "crash" and before.detail == after.detail
            and row.get("baseline_failure_at_lead") and _crash_at_lead(before, lead)
        )
        if recovered_crash:
            passed = True
            result = {
                "reason": "recovered_crash",
                "target": task,
                "metric": load_task(task).metric,
                "delta": None,
                "recovery": {"stock_status": before.status, "patched_status": after.status},
            }
        elif unchanged_crash:
            # A source where both versions hit the same planned failure is an
            # explicit neutral guard.  It cannot vote for the patch, but it is
            # also not evidence that the patch regressed that source.
            passed = False
            result = {
                "reason": "target_no_gain",
                "target": task,
                "metric": load_task(task).metric,
                "delta": None,
                "unchanged_baseline_crash_unmeasured": True,
            }
        else:
            passed, result = judge_pair({task: before}, {task: after}, task, practical=practical)
        result.update(task=task, scenario=target, dataset=row["dataset"], affected=row["affected"],
                      before=before.score, after=after.score)
        report["datasets"].append(result)
        if before.data_evidence.get("data_hash") != row["data_hash"]:
            failures.append(f"data_changed:{task}")
        if result["reason"] not in {"passed", "target_no_gain", "recovered_crash"}:
            failures.append(f"{task}:{result['reason']}")
        configuration_reached = bool(row.get("configuration_operation")) and any(
            flow.get("operation") == row["configuration_operation"]
            for flow in before.dataflow
        )
        if row["affected"] and not (_line_reached(before, lead) or _crash_at_lead(before, lead) or configuration_reached):
            failures.append(f"branch_not_reached:{task}")
        # Parameterized reproduction must also preserve the ordinary operation
        # on every source, not only on the original target dataset.
        # A registered ordinary-index scenario already guards a shifted-index
        # scenario on this exact source. Do not run it twice. Explicit
        # reproduction parameters still require a separate ordinary run.
        explicit_default = any(
            other["task"] == task and load_task(other["scenario"]).index_offset == 0
            for other in selected_rows
        )
        if params or (load_task(target).index_offset and not explicit_default):
            normal = _override(target, {})
            normal["index_offset"] = 0
            if fold is not None:
                normal["resample_fold"] = fold
            started = budget.begin(budget_stage) if budget is not None else None
            if budget is not None and started is None:
                report.update(affected_dataset_count=len(affected_hashes), failures=failures,
                              measurement_budget=budget.snapshot(), reason="inconclusive_budget")
                return False, report
            try:
                base = run_stock(task, checkout=source, split=split, seed=seed,
                                 collect_coverage=True, task_override=normal)
                patched = run_patched(task, checkout=experiment, split=split, seed=seed,
                                      collect_coverage=True, task_override=normal)
            finally:
                if budget is not None:
                    budget.finish(started)
            _, guard = judge_pair({task: base}, {task: patched}, task, practical=False)
            result["default_guard"] = guard
            if guard["reason"] not in {"passed", "target_no_gain"}:
                failures.append(f"default:{task}:{guard['reason']}")
        if row["affected"]:
            if split == "final" and not recovered_crash:
                interval = paired_interval(before, after, metric=load_task(task).metric,
                                           time_series=load_task(task).problem == "ts",
                                           comparisons=comparisons * sum(r["affected"] for r in plan["datasets"]))
                result["uncertainty"] = interval
                passed = passed and bool(interval.get("excludes_zero"))
            # Identical content cannot accumulate votes, even under aliases.
            key = (row["data_hash"], target)
            scenario_wins[key] = scenario_wins.get(key, True) and passed
    # A source gets one vote if a preregistered affected scenario improves.
    # All scenarios must still be measured and pass regression protection.
    count = len({data_hash for (data_hash, _), won in scenario_wins.items() if won})
    report.update(affected_dataset_count=len(affected_hashes), positive_dataset_count=count,
                  required_positive_datasets=required, failures=failures,
                  measurement_budget=budget.snapshot() if budget is not None else None)
    ok = not failures and count >= required
    recovered = sum(row.get("reason") == "recovered_crash" for row in report["datasets"])
    report["evidence_kind"] = "crash_recovery" if recovered else "paired_metric"
    report["reason"] = (
        "transfer_screen_promising" if scope == "screen"
        else ("transfer_recovery_confirmed" if recovered else "transfer_confirmed")
    ) if ok else ("transfer_invalid_or_regressed" if failures else "transfer_gain_not_reproduced")
    return ok, report
