"""Metric-only research: preregister -> DEV/CV -> SHADOW -> frozen FINAL batch.

No FINAL evaluation while selecting the batch. A study has one FINAL batch;
failed/exposed holdouts cannot be reused for another adaptive search.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from fedotllm.agents.evolve.evaluation.affected_eval import _line_reached
from fedotllm.agents.evolve.evaluation.eval import run_stock, run_patched
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL, FOLDS
from fedotllm.agents.evolve.evaluation.tasks import load_task
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout, discard_experiment_checkout, source_fingerprint,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.protocol import acceptance_protocol_fingerprint
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.types import Decision, PatchCandidate, PatchEdit

BATCH_SIZE = 3
EPS = 1e-12


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def preregister(lead, stock, lift_ids, path: Path, *, operation_params=None, source=None,
                screen_sources: int = 3) -> dict:
    """Select from baseline coverage, never from patched scores or best deltas."""
    eligible = sorted(task for task in lift_ids if task in stock and _line_reached(stock[task], lead))
    operation_params = operation_params or {}
    if operation_params:
        eligible = [task for task in eligible if set(load_task(task).nodes) & set(operation_params)]
    target = eligible[0] if eligible else None
    plan = {
        "protocol": PROTOCOL, "lead": asdict(lead), "target_task": target,
        "target_selection": "lexicographically_first_baseline_covered_workload",
        "eligible_tasks": eligible, "target_metric": load_task(target).metric if target else None,
        "data_hashes": {task: result.data_evidence.get("data_hash") for task, result in sorted(stock.items())},
        "operation_params": operation_params,
        "cv_folds": list(FOLDS), "model_seeds": [42, 43, 44],
        "protect_policy": "no_practically_material_regression; identical_baseline_crash_is_unmeasured",
        "final_policy": "one_batch_of_three_candidates_frozen_before_FINAL",
    }
    from fedotllm.agents.evolve.controller.transfer import preregister_transfer
    plan["transfer"] = (
        preregister_transfer(source, target, lead, operation_params, screen_sources=screen_sources)
        if source is not None else None
    )
    plan = json.loads(json.dumps(plan))
    # Exclusive creation makes accidental retargeting on resume visible.
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("preregistered metric target cannot be changed")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
    return plan


def _pair(source, experiment, tasks, *, split, fold=None, seed=42, target=None, operation_params=None):
    override = {"evaluation_protocol": PROTOCOL}
    if fold is not None:
        override["resample_fold"] = fold
    before, after = {}, {}
    for task in tasks:
        before[task] = run_stock(task, checkout=source, split=split, seed=seed,
                                 collect_coverage=True, task_override=override)
        after[task] = run_patched(task, checkout=experiment, split=split, seed=seed,
                                  collect_coverage=True, task_override=override)
        if task == target and operation_params:
            # Preserve the original default workload as an additional guard.
            # Explicit reproduction parameters define the target scenario only.
            before[task + "::default"] = before[task]
            after[task + "::default"] = after[task]
            affected = {**override, "operation_params": operation_params}
            before[task] = run_stock(task, checkout=source, split=split, seed=seed,
                                    collect_coverage=True, task_override=affected)
            after[task] = run_patched(task, checkout=experiment, split=split, seed=seed,
                                     collect_coverage=True, task_override=affected)
    return before, after


def judge_pair(before, after, target, *, practical=True):
    """One fixed target, per-task units, separate strict protection tolerance."""
    deltas, baseline_crashes = {}, []
    for task in before:
        a, b = before[task], after[task]
        left, right = a.data_evidence, b.data_evidence
        if not left or left != right or left.get("protocol") != PROTOCOL:
            return False, {"reason": f"data_pair_mismatch:{task}", "deltas": deltas}
        if a.status == b.status == "crash" and a.detail == b.detail and task != target:
            baseline_crashes.append(task)
            deltas[task] = None
            continue
        if a.status == "ok" and b.status == "crash":
            return False, {"reason": f"regression:{task}:crash", "deltas": deltas}
        if a.status != "ok" or b.status != "ok":
            return False, {"reason": f"unmeasured:{task}", "deltas": deltas}
        spec = load_task(task.split("::")[0])
        delta = b.score - a.score if spec.higher_is_better else a.score - b.score
        if not (-float("inf") < delta < float("inf")):
            return False, {"reason": f"nonfinite:{task}", "deltas": deltas}
        deltas[task] = delta
        # The practical threshold protects against changes a user could notice.
        # A tiny counter-movement below it is retained for later aggregate
        # confirmation; treating every floating-point difference as a hard
        # regression wrongly prunes known useful default changes.
        regression_tolerance = EPS
        if practical:
            regression_tolerance = spec.min_delta
            if spec.min_delta_mode == "relative":
                regression_tolerance *= max(abs(a.score), EPS)
        if delta < -regression_tolerance:
            return False, {"reason": f"regression:{task}", "deltas": deltas}
    if target not in deltas or deltas[target] is None:
        return False, {"reason": "target_unmeasured", "deltas": deltas}
    spec = load_task(target)
    threshold = spec.min_delta
    if spec.min_delta_mode == "relative":
        threshold *= max(abs(before[target].score), EPS)
    if not practical:
        threshold = EPS
    passed = deltas[target] >= threshold
    return passed, {
        "reason": "passed" if passed else "target_no_gain",
        "target": target, "metric": spec.metric, "delta": deltas[target],
        "threshold": threshold, "deltas": deltas,
        "unchanged_baseline_crashes_unmeasured": baseline_crashes,
        "data": before[target].data_evidence,
    }


def _stage(source, experiment, tasks, target, *, split, fold=None, seed=42, practical=True,
           operation_params=None, transfer=None, transfer_scope="full", measurement_budget=None):
    before, after = _pair(source, experiment, tasks, split=split, fold=fold, seed=seed,
                          target=target, operation_params=operation_params)
    passed, result = judge_pair(before, after, target, practical=practical)
    result.update(split=split, fold=fold, seed=seed, passed=passed)
    if transfer is not None and result["reason"] in {"passed", "target_no_gain"}:
        from fedotllm.agents.evolve.controller.transfer import evaluate_transfer
        transferred, evidence = evaluate_transfer(
            source, experiment, transfer, params=operation_params, split=split,
            fold=fold, seed=seed, practical=practical, comparisons=BATCH_SIZE,
            scope=transfer_scope, budget=measurement_budget,
        )
        result["transfer"] = evidence
        passed = transferred
        result.update(passed=passed, reason="passed" if passed else evidence["reason"])
    return passed, result, before, after


def _reserve(path: Path, payload) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)
    except FileExistsError:
        return False
    return True


def evaluate_candidate(source, experiment, candidate, verification, plan, *,
                       lead, tasks, study: Path, workspace: Path, measurement_budget=None):
    """Return a research decision; no metric-neutral result satisfies the goal."""
    target = plan["target_task"]
    rows = []
    report = {"plan": plan, "stages": rows, "candidate": candidate.candidate_id}

    def done(reason, *, keep=False, final=None):
        report["reason"] = reason
        _save(workspace / "metric_study.json", report)
        delta = rows[0].get("delta") if rows else None
        final_result = report.get("current_final_result") or {}
        if final is not None:
            delta = final_result.get("delta")
        return Decision(
            keep, reason, delta, regression_deltas=final_result.get("deltas", {}),
            stage="metric_study", final_keep=final, dev_keep=bool(rows and rows[0].get("passed")),
            metric_goal_keep=keep and verification.status == "verified_bug" and final is True,
        )

    if not target:
        return done("metric_not_measured_no_covered_target")
    from fedotllm.agents.evolve.controller.transfer import validate_transfer
    transfer_error = validate_transfer(plan.get("transfer"))
    if transfer_error:
        return done(transfer_error)
    if (study / "final-batch.json").exists():
        return done("metric_study_FINAL_already_sealed")
    identity = {
        "source_hash": source_fingerprint(source),
        "acceptance_protocol": acceptance_protocol_fingerprint(),
        "data_protocol": PROTOCOL, "tasks": list(tasks), "batch_size": BATCH_SIZE,
        "data_hashes": plan["data_hashes"],
    }
    manifest = study / "study.json"
    if not _reserve(manifest, identity) and json.loads(manifest.read_text()) != identity:
        return done("metric_study_identity_mismatch")
    params = plan.get("operation_params")
    passed, result, before, _ = _stage(
        source, experiment, tasks, target, split="dev", operation_params=params,
        transfer=plan["transfer"], transfer_scope="screen", measurement_budget=measurement_budget,
    )
    rows.append(result)
    if any(not plan["data_hashes"].get(task) or before[task].data_evidence.get("data_hash") != plan["data_hashes"][task]
           for task in tasks):
        return done("metric_data_changed_since_preregistration")
    if not _line_reached(before[target], lead):
        return done("metric_not_measured_branch_not_executed")
    if not passed:
        return done("metric_DEV_rejected:" + result["reason"])

    # Seeds and folds verify that the selected template has not regressed and
    # that our split bookkeeping is sound. They are not additional datasets
    # and cannot veto a transfer signal merely because this one source is
    # neutral. The full cross-dataset SHADOW/FNAL gates decide the gain claim.
    local_seed_outcomes = []
    for seed in (43, 44):
        passed, result, measured, _ = _stage(
            source, experiment, (target,), target, split="dev", seed=seed,
            operation_params=params,
        )
        rows.append(result)
        if measured[target].data_evidence.get("data_hash") != plan["data_hashes"][target]:
            return done("metric_data_changed_since_preregistration")
        local_seed_outcomes.append(result["reason"])
        if result["reason"] not in {"passed", "target_no_gain"}:
            return done("metric_DEV_model_seed_rejected")
    heldout_ids = set(rows[0]["data"]["validation_ids"])
    seen_cv = set()
    local_fold_outcomes = []
    for fold in FOLDS:
        passed, result, measured, _ = _stage(
            source, experiment, (target,), target, split="dev", fold=fold,
            practical=False, operation_params=params,
        )
        rows.append(result)
        if measured[target].data_evidence.get("data_hash") != plan["data_hashes"][target]:
            return done("metric_data_changed_since_preregistration")
        if result["reason"] not in {"passed", "target_no_gain"}:
            return done("metric_CV_rejected:" + result["reason"])
        ids = set(result["data"]["validation_ids"])
        if not ids or ids & (heldout_ids | seen_cv):
            return done("metric_CV_validation_overlap")
        seen_cv.update(ids)
        local_fold_outcomes.append(result["reason"])
    report["local_template_guards"] = {
        "model_seed_outcomes": local_seed_outcomes,
        "fold_outcomes": local_fold_outcomes,
    }
    if verification.status != "verified_bug":
        # Quality tuning is useful research, but must not use FINAL slots meant
        # to establish three actual defect fixes.
        return done("metric_quality_signal_not_a_verified_defect")

    dataset = load_task(target).dataset
    # Candidate uniqueness and cross-dataset replication are separate checks.
    site_key = hashlib.sha256(f"{lead.file_path}:{lead.line}".encode()).hexdigest()[:16]
    frozen = {
        "candidate": asdict(candidate), "plan": plan, "dataset": dataset,
        "site_key": site_key,
        "workspace": str(workspace.resolve()), "source_hash": identity["source_hash"],
        "patch_hash": hashlib.sha256(json.dumps(asdict(candidate), sort_keys=True).encode()).hexdigest(),
    }
    # Once this site's SHADOW is exposed, do not tune another revision to it.
    if not _reserve(study / "shadow-locks" / f"{site_key}.json", frozen):
        return done("metric_SHADOW_site_already_exposed")
    passed, result, measured, _ = _stage(
        source, experiment, tasks, target, split="shadow", operation_params=params,
        transfer=plan["transfer"], measurement_budget=measurement_budget,
    )
    rows.append(result)
    if any(measured[task].data_evidence.get("data_hash") != plan["data_hashes"][task] for task in tasks):
        return done("metric_data_changed_since_preregistration")
    if not passed:
        return done("metric_heldout_rejected")  # no held-out scores for Fixer
    if set(result["data"]["validation_ids"]) & (heldout_ids | seen_cv):
        return done("metric_SHADOW_validation_overlap")
    frozen["development_evidence"] = rows
    _save(study / "ready" / f"{site_key}.json", frozen)
    ready = sorted((study / "ready").glob("*.json"))
    if len(ready) < BATCH_SIZE:
        return done("metric_ready_waiting_for_frozen_FINAL_batch")
    results = finalize_batch(source, study, measurement_budget=measurement_budget)
    report["final_batch"] = results
    current = next((r for r in results["cases"] if r["candidate"] == candidate.candidate_id), None)
    accepted = bool(current and current["passed"])
    report["current_final_result"] = current["result"] if current else {}
    return done("metric_FINAL_confirmed" if accepted else "metric_FINAL_rejected",
                keep=accepted, final=accepted)


def finalize_batch(source: Path, study: Path, *, measurement_budget=None) -> dict:
    """Serialize FINAL execution; retries resume only the already frozen batch."""
    import fcntl

    study.mkdir(parents=True, exist_ok=True)
    with (study / "final-execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _finalize_batch_locked(source, study, measurement_budget=measurement_budget)


def _finalize_batch_locked(source: Path, study: Path, *, measurement_budget=None) -> dict:
    """Seal every patch before the first FINAL query; fail closed on reuse."""
    result_path = study / "final-results.json"
    sealed = study / "final-batch.json"
    ready = (
        json.loads(sealed.read_text())["candidates"] if sealed.exists() else
        [json.loads(path.read_text()) for path in sorted((study / "ready").glob("*.json"))]
    )
    if len(ready) != BATCH_SIZE or len({r["site_key"] for r in ready}) != BATCH_SIZE:
        raise ValueError("FINAL requires three frozen candidate sites")
    manifest = json.loads((study / "study.json").read_text())
    if manifest["source_hash"] != source_fingerprint(source):
        raise ValueError("study source changed")
    if manifest["acceptance_protocol"] != acceptance_protocol_fingerprint():
        raise ValueError("study acceptance protocol changed")
    from fedotllm.agents.evolve.controller.transfer import validate_transfer
    for row in ready:
        error = validate_transfer(row["plan"].get("transfer"))
        if error:
            raise ValueError(error)
    frozen_batch = {"study": manifest, "candidates": ready}
    if not _reserve(sealed, frozen_batch) and json.loads(sealed.read_text()) != frozen_batch:
        raise ValueError("FINAL batch identity changed")
    if result_path.exists():
        return json.loads(result_path.read_text())
    checkpoint = study / "final-checkpoint.jsonl"
    cases = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()] if checkpoint.exists() else []
    completed = {row["candidate"] for row in cases}
    frozen_ids = {row["candidate"]["candidate_id"] for row in ready}
    if len(completed) != len(cases) or not completed <= frozen_ids:
        raise ValueError("FINAL checkpoint contains duplicate or foreign candidates")
    for index, row in enumerate(ready):
        if row["candidate"]["candidate_id"] in completed:
            continue
        tree = create_experiment_checkout(source, study, run_id="final", candidate_id=str(index))
        try:
            raw = row["candidate"]
            candidate = PatchCandidate(raw["candidate_id"], edits=[PatchEdit(**e) for e in raw["edits"]])
            if not apply_patch(tree, candidate):
                passed, result = False, {"reason": "frozen_patch_replay_failed"}
            else:
                passed, result, before, after = _stage(
                    source, tree, tuple(manifest["tasks"]), row["plan"]["target_task"], split="final",
                    operation_params=row["plan"].get("operation_params"),
                    transfer=row["plan"]["transfer"],
                    measurement_budget=measurement_budget,
                )
                if any(before[task].data_evidence.get("data_hash") != manifest.get("data_hashes", {}).get(task.split("::")[0])
                       for task in before):
                    passed = False
                    result["reason"] = "FINAL_data_changed_since_preregistration"
                if passed:
                    prior = {i for stage in row["development_evidence"]
                             for i in stage.get("data", {}).get("validation_ids", [])}
                    if prior & set(result["data"]["validation_ids"]):
                        passed = False
                        result["reason"] = "FINAL_validation_overlap"
                # FINAL uncertainty is evaluated for every affected source by
                # the transfer stage; the template dataset gets no extra vote.
            case = {"candidate": candidate.candidate_id, "passed": passed, "result": result}
            cases.append(case)
            append_journal(study / "final-checkpoint.jsonl", case)
        finally:
            discard_experiment_checkout(tree, workspace=study, source=source)
    result = {"cases": cases, "metric_confirmed_count": sum(r["passed"] for r in cases),
              "requires_independent_defect_and_novelty_audit": True,
              "goal_achieved": False, "FINAL_consumed": True}
    _save(result_path, result)
    return result
