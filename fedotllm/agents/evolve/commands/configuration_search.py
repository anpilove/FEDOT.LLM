"""Deterministic, evaluator-blind variants for executed operation defaults.

The LLM should identify a causal operation/mechanism.  It should not need to
guess one exact scalar value.  This module derives a small, general set of mild
alternatives from the estimator's real ``get_params(deep=False)`` surface.  It
never reads scores, datasets, benchmark cases, or FINAL.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fedotllm.agents.evolve.types import (
    Decision,
    PatchCandidate,
    PatchEdit,
    ScoreResult,
    TestResult,
)

DEFAULTS_FILE = "fedot/core/repository/data/default_operation_params.json"
BOOSTING_IMPLEMENTATIONS = (
    "fedot/core/operations/evaluation/operation_implementations/"
    "models/boostings_implementations.py"
)
_DISCARDED_EARLY_STOP = (
    "        if isinstance(esr, int):\n"
    "            lgbm_early_stopping(esr, verbose=self.params.get('verbose'))\n"
)
_APPENDED_EARLY_STOP = (
    "        if isinstance(esr, int):\n"
    "            callback.append(lgbm_early_stopping(esr, verbose=self.params.get('verbose')))\n"
)


def discarded_early_stopping_candidate(source: Path) -> PatchCandidate | None:
    """The LightGBM fit builds an early-stopping callback and then drops it."""

    path = source / BOOSTING_IMPLEMENTATIONS
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if text.count(_DISCARDED_EARLY_STOP) != 1:
        return None
    if "callback.append(lgbm_early_stopping" in text:
        return None
    return PatchCandidate(
        candidate_id="early-stopping-callback",
        file_path=BOOSTING_IMPLEMENTATIONS,
        old_code=_DISCARDED_EARLY_STOP,
        new_code=_APPENDED_EARLY_STOP,
        rationale=(
            "LightGBM builds the early-stopping callback and then discards it, "
            "so early_stopping_rounds never stops training"
        ),
        behavior_probe=(
            "from pathlib import Path\n"
            f"text = Path({BOOSTING_IMPLEMENTATIONS!r}).read_text()\n"
            "print('EVOLVE_OBSERVATION=' + ("
            "'appended' if 'callback.append(lgbm_early_stopping' in text else 'discarded'))\n"
        ),
    )
_PARAMETER_PRIORITY = (
    "min_samples_leaf",
    "min_samples_split",
    "min_data_in_leaf",
    "min_child_samples",
    "min_child_weight",
    "alpha",
    "l2_leaf_reg",
    "reg_lambda",
    "reg_alpha",
    "max_depth",
    "max_features",
    "class_weight",
    "learning_rate",
    "num_leaves",
    "colsample_bytree",
    "subsample",
    "bagging_fraction",
    "whiten",
    "algorithm",
    "svd_solver",
    "extra_trees",
    "max_iter",
    "tol",
)


@dataclass(frozen=True)
class ParameterVariant:
    operation: str
    parameter: str
    value: str | int | float | bool | None
    current_value: str | int | float | bool | None
    rationale: str


@dataclass(frozen=True)
class ParameterBundle:
    operation: str
    variants: tuple[ParameterVariant, ...]
    rationale: str


@dataclass
class ConfigurationSearchOutcome:
    candidate: PatchCandidate | None = None
    experiment: Path | None = None
    decision: Decision = field(
        default_factory=lambda: Decision(False, "no_configuration_keep", None)
    )
    tests: TestResult | None = None
    patched: dict[str, ScoreResult] = field(default_factory=dict)
    behavior_probe: dict = field(default_factory=dict)
    trials: list[dict] = field(default_factory=list)
    patch_path: Path | None = None
    dev_keeps: list[DevKeep] = field(default_factory=list)


@dataclass
class DevKeep:
    candidate: PatchCandidate
    experiment: Path
    decision: Decision
    tests: TestResult | None
    behavior_probe: dict
    patch_path: Path | None
    variant: dict


KEPT_CONFIGURATION_PATCH = "kept_configuration.patch"
CONFIGURATION_MEMORY = "configuration_memory.jsonl"


def remembered_configuration_trials(workspace_root: Path) -> list[dict]:
    """Trials this workspace already measured, including earlier run journals."""

    from fedotllm.agents.evolve.storage.journal import read_jsonl

    rows = list(read_jsonl(workspace_root / CONFIGURATION_MEMORY))
    runs = workspace_root / "runs"
    if runs.is_dir():
        for path in sorted(runs.glob("*/configuration_trials.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


HUNT_SURFACES: tuple[dict[str, tuple[str, ...] | str], ...] = (
    {
        "operation": "rf",
        "lift": ("rf", "cancer", "kc2"),
        "protect": ("rf", "cancer", "kc2"),
    },
    {
        "operation": "lgbm",
        "lift": ("lgbm", "cancer->lgbm", "kc2->lgbm"),
        "protect": (
            "lgbm",
            "cancer->lgbm",
            "kc2->lgbm",
            "rf",
            "cancer",
            "kc2",
        ),
    },
    {
        "operation": "catboost",
        "lift": ("catboost", "cancer->catboost", "kc2->catboost"),
        "protect": (
            "catboost",
            "cancer->catboost",
            "kc2->catboost",
            "lgbm",
            "cancer->lgbm",
            "kc2->lgbm",
            "rf",
            "cancer",
            "kc2",
        ),
    },
    {
        "operation": "xgboost",
        "lift": ("xgboost", "cancer->xgboost", "kc2->xgboost"),
        "protect": (
            "xgboost",
            "cancer->xgboost",
            "kc2->xgboost",
            "catboost",
            "cancer->catboost",
            "kc2->catboost",
            "lgbm",
            "cancer->lgbm",
            "kc2->lgbm",
            "rf",
            "cancer",
            "kc2",
        ),
    },
)


_MEASURED_STAGES = frozenset({"quick_dev", "full_dev", "shadow_dev", "final"})


def operation_reached_second_scale(workspace_root: Path, operation: str) -> bool:
    """True when one parameter of this operation already has two measured values.

    The queue measures every parameter's first value before a second scale, so
    a repeated parameter means that first pass is finished. Without a FINAL
    keep the hunt should move on instead of spending later runs on refinements.
    """

    runs = workspace_root / "runs"
    if not runs.is_dir():
        return False
    seen: dict[str, set[str]] = {}
    for path in runs.glob("*/configuration_trials.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("stage") not in _MEASURED_STAGES:
                continue
            variant = row.get("variant")
            if not isinstance(variant, dict) or "variants" in variant:
                continue
            if variant.get("operation") != operation:
                continue
            parameter = str(variant.get("parameter") or "")
            if not parameter or "value" not in variant:
                continue
            values = seen.setdefault(parameter, set())
            values.add(json.dumps(variant.get("value"), sort_keys=True))
            if len(values) >= 2:
                return True
    return False


def hunt_surface(workspace_root: Path, requested: tuple[str, ...] = ()) -> dict:
    """Next FEDOT operation a keyless hunt should measure.

    An explicit task list stays on that list. Otherwise a FINAL keep, or a
    finished first pass with no keep, moves the hunt to the next operation.
    The previous tasks stay protected so the new default cannot give back the
    earlier FEDOT improvement.
    """

    if requested:
        operation = requested[0].rsplit("->", 1)[-1]
        return {"operation": operation, "lift": requested, "protect": requested}
    kept: set[str] = set()
    runs = workspace_root / "runs"
    if runs.is_dir():
        for path in sorted(runs.glob("*/keyless_configuration.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if not isinstance(summary, dict) or not summary.get("final_keep"):
                continue
            change = str(summary.get("change") or "").strip()
            operation = change.split()[0] if change else ""
            if operation:
                kept.add(operation)
    for surface in HUNT_SURFACES:
        operation = str(surface["operation"])
        if operation in kept:
            continue
        if operation_reached_second_scale(workspace_root, operation):
            continue
        return {
            "operation": surface["operation"],
            "lift": surface["lift"],
            "protect": surface["protect"],
        }
    last = HUNT_SURFACES[-1]
    return {
        "operation": last["operation"],
        "lift": last["lift"],
        "protect": last["protect"],
    }


def strongest_remembered_final(workspace_root: Path) -> dict | None:
    """Earlier keyless FINAL keep with the largest target delta."""

    best: dict | None = None
    best_delta: float | None = None
    runs = workspace_root / "runs"
    if not runs.is_dir():
        return None
    for path in sorted(runs.glob("*/keyless_configuration.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if not isinstance(summary, dict) or not summary.get("final_keep"):
            continue
        delta = summary.get("final_target_delta")
        patch = Path(str(summary.get("patch") or ""))
        if not isinstance(delta, (int, float)) or not patch.is_file():
            continue
        if best_delta is None or float(delta) > best_delta:
            best = summary
            best_delta = float(delta)
    return best


_NOT_MEASURED = {"dedup", "kept_parameter"}


def parameters_already_in_defaults(source: Path, proposal: Path) -> dict[str, set[str]]:
    """Parameters a remembered patch already wrote into the defaults file."""

    def load(path: Path) -> dict:
        file = path / DEFAULTS_FILE
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    base = load(source)
    proposed = load(proposal)
    locked: dict[str, set[str]] = {}
    for operation, params in proposed.items():
        if not isinstance(params, dict):
            continue
        current = base.get(operation)
        current = current if isinstance(current, dict) else {}
        chosen = {
            str(key)
            for key, value in params.items()
            if key not in current or current.get(key) != value
        }
        if chosen:
            locked[str(operation)] = chosen
    return locked


def materialize_improved_fedot(source: Path, dest: Path, patch: Path) -> Path:
    """Copy the pinned FEDOT tree and apply the kept patch. The pin stays unchanged."""

    import shutil
    import subprocess

    source = source.resolve()
    dest = dest.resolve()
    if dest == source or dest in source.parents or source in dest.parents:
        raise ValueError("improved FEDOT and immutable source must be disjoint")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", ".pytest_cache"),
    )
    completed = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-d", str(dest), "-i", str(patch.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        detail = (completed.stderr or completed.stdout or "patch failed").strip()
        raise RuntimeError(detail)
    return dest


def proposal_checkout(source: Path, workspace: Path, patch: Path) -> Path:
    """Copy the frozen source and apply the remembered configuration patch."""

    import subprocess

    from fedotllm.agents.evolve.execution.checkout import create_experiment_checkout

    dest = create_experiment_checkout(
        source,
        workspace,
        run_id="proposal",
        candidate_id="kept-baseline",
    )
    completed = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-d", str(dest), "-i", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "patch failed").strip()
        raise RuntimeError(detail)
    return dest


def remembered_patch_hashes(trials: list[dict]) -> set[str]:
    """Patch hashes that already consumed a measurement, not a dedup note."""

    return {
        str(row["patch_hash"])
        for row in trials
        if row.get("patch_hash") and row.get("stage") not in _NOT_MEASURED
    }


def remember_configuration_trials(workspace_root: Path, run_workspace: Path) -> int:
    """Store this run's newly measured trials without copying the test logs."""

    from fedotllm.agents.evolve.storage.journal import append_journal, read_jsonl

    memory = workspace_root / CONFIGURATION_MEMORY
    known = remembered_patch_hashes(read_jsonl(memory))
    added = 0
    for row in read_jsonl(run_workspace / "configuration_trials.jsonl"):
        patch_hash = str(row.get("patch_hash") or "")
        if (
            not patch_hash
            or row.get("stage") in _NOT_MEASURED
            or patch_hash in known
        ):
            continue
        append_journal(
            memory,
            {
                "record_type": "configuration_trial",
                "patch_hash": patch_hash,
                "stage": row.get("stage"),
                "reason": row.get("reason")
                or (row.get("full_decision") or {}).get("reason")
                or (row.get("quick_decision") or {}).get("reason"),
                "variant": row.get("variant") or {},
                "candidate": row.get("candidate"),
                "run_workspace": str(run_workspace),
            },
        )
        known.add(patch_hash)
        added += 1
    return added


def choose_final_keep(finals: list[Decision]) -> int | None:
    """Index of the passing FINAL with the largest target delta."""

    chosen: int | None = None
    for index, decision in enumerate(finals):
        if not decision.keep:
            continue
        if chosen is None:
            chosen = index
            continue
        current = finals[chosen].target_delta
        candidate = decision.target_delta
        if candidate is None:
            continue
        if current is None or candidate > current:
            chosen = index
    return chosen


def measured_trial_rows(trials: list[dict]) -> list[dict]:
    """Compact record of every default the search actually scored."""

    rows: list[dict] = []
    for row in trials:
        variant = row.get("variant") or {}
        decision = row.get("full_decision") or row.get("quick_decision") or {}
        rows.append(
            {
                "parameter": variant.get("parameter"),
                "from": variant.get("current_value"),
                "to": variant.get("value"),
                "stage": row.get("stage"),
                "dev_target_delta": decision.get("target_delta"),
                "reason": row.get("reason") or decision.get("reason"),
            }
        )
    return rows


def write_kept_configuration_patch(
    workspace: Path,
    candidate: PatchCandidate,
    *,
    source: Path,
    experiment: Path,
) -> Path:
    """Save a git-applyable diff of the measured checkout."""

    chunks: list[str] = []
    edits = list(candidate.edits)
    if not any(edit.file_path == BOOSTING_IMPLEMENTATIONS for edit in edits):
        frozen = source / BOOSTING_IMPLEMENTATIONS
        changed = experiment / BOOSTING_IMPLEMENTATIONS
        if (
            frozen.is_file()
            and changed.is_file()
            and frozen.read_bytes() != changed.read_bytes()
        ):
            edits.append(PatchEdit(BOOSTING_IMPLEMENTATIONS, "callback", "callback"))
    for edit in edits:
        old_path = source / edit.file_path
        new_path = experiment / edit.file_path
        old = old_path.read_text(encoding="utf-8") if old_path.is_file() else ""
        new = new_path.read_text(encoding="utf-8") if new_path.is_file() else ""
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{edit.file_path}",
                tofile=f"b/{edit.file_path}",
            )
        )
        if diff and not diff.endswith("\n"):
            diff += "\n"
        chunks.append(diff)
    path = workspace / KEPT_CONFIGURATION_PATCH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def measure_discarded_early_stopping(
    source: Path,
    workspace: Path,
    *,
    tasks: tuple[str, ...],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    proposal_source: Path | None = None,
    skip_patch_hashes: set[str] | None = None,
    dev_seed: int = 42,
) -> dict | None:
    """Measure the discarded LightGBM callback once, through the same gates."""

    from fedotllm.agents.evolve.controller.campaign import compare_behavior_probe
    from fedotllm.agents.evolve.evaluation.judge import (
        confirm_candidate_tests,
        measure_fedot_tests,
        measure_patched,
        measure_stock,
        normalize_test_result,
        verdict,
    )
    from fedotllm.agents.evolve.execution.checkout import (
        create_experiment_checkout,
        discard_experiment_checkout,
        source_fingerprint,
    )
    from fedotllm.agents.evolve.execution.patch import apply_patch
    from fedotllm.agents.evolve.storage.hypothesis import normalized_patch_hash
    from fedotllm.agents.evolve.storage.journal import append_journal

    candidate = discarded_early_stopping_candidate(source)
    if candidate is None:
        return None
    patch_hash = normalized_patch_hash(candidate, source_fingerprint(source))
    if patch_hash in set(skip_patch_hashes or ()):
        return None
    base = proposal_source or source
    experiment = create_experiment_checkout(
        base,
        workspace,
        run_id="early-stop",
        candidate_id=candidate.candidate_id,
    )
    change = "lgbm early_stopping callback discarded -> appended"
    journal = workspace / "configuration_trials.jsonl"

    def finish(decision: Decision, stage: str) -> dict:
        append_journal(
            journal,
            {
                "event": "configuration_trial",
                "variant": {
                    "operation": "lgbm",
                    "parameter": "early_stopping_rounds",
                    "value": "callback",
                    "current_value": "discarded",
                },
                "candidate": candidate.candidate_id,
                "patch_hash": patch_hash,
                "stage": stage,
                "reason": decision.reason,
                "quick_decision": asdict(decision) if stage == "quick_dev" else None,
            },
        )
        if experiment.is_dir():
            discard_experiment_checkout(experiment, workspace=workspace, source=base)
        return {
            "ok": False,
            "decision": {
                "keep": False,
                "reason": decision.reason,
                "target_delta": decision.target_delta,
                "regression_deltas": decision.regression_deltas,
                "infrastructure_error": decision.infrastructure_error,
            },
            "summary": {
                "change": change,
                "final_keep": False,
                "measured_change": decision.reason,
            },
        }

    try:
        if not apply_patch(experiment, candidate):
            return finish(Decision(False, "early_stopping_patch_failed", None), "patch")
        lift = tuple(lift_ids)
        protect = tuple(protect_ids)
        measured_ids = tuple(dict.fromkeys((*lift, *protect)))
        stock = measure_stock(measured_ids, checkout=base, split="dev", seed=dev_seed)
        quick = verdict(
            {task_id: stock[task_id] for task_id in lift},
            measure_patched(lift, checkout=experiment, split="dev", seed=dev_seed),
            lift_ids=lift,
            protect_ids=lift,
        )
        if not quick.keep:
            return finish(quick, "quick_dev")
        full = verdict(
            stock,
            measure_patched(measured_ids, checkout=experiment, split="dev", seed=dev_seed),
            lift_ids=lift,
            protect_ids=protect,
        )
        if not full.keep:
            return finish(full, "full_dev")
        shadow_stock = measure_stock(lift, checkout=base, split="shadow", seed=dev_seed)
        shadow = verdict(
            shadow_stock,
            measure_patched(lift, checkout=experiment, split="shadow", seed=dev_seed),
            lift_ids=lift,
            protect_ids=lift,
        )
        if not shadow.keep:
            return finish(shadow, "shadow_dev")
        probe = compare_behavior_probe(source, experiment, candidate.behavior_probe)
        if probe.get("status") != "changed":
            return finish(Decision(False, f"probe {probe.get('status')}", full.target_delta), "probe")
        baseline = normalize_test_result(measure_fedot_tests(source))
        after_tests, failure, _attempts = confirm_candidate_tests(baseline, experiment)
        if failure is not None:
            return finish(failure, "tests")
        if after_tests.status != "passed":
            return finish(Decision(False, f"tests {after_tests.status}", full.target_delta), "tests")
        final_stock = measure_stock(measured_ids, checkout=base, split="final", seed=dev_seed)
        final = verdict(
            final_stock,
            measure_patched(measured_ids, checkout=experiment, split="final", seed=dev_seed),
            lift_ids=lift,
            protect_ids=protect,
        )
        kept = bool(final.keep)
        patch_path = ""
        if kept:
            published = candidate
            if proposal_source is not None:
                published = PatchCandidate(
                    candidate_id=candidate.candidate_id,
                    file_path=candidate.file_path,
                    rationale=candidate.rationale,
                    edits=[
                        *candidate.edits,
                        PatchEdit(DEFAULTS_FILE, "defaults", "defaults"),
                    ],
                )
            patch_path = str(
                write_kept_configuration_patch(
                    workspace,
                    published,
                    source=source,
                    experiment=experiment,
                )
            )
        append_journal(
            journal,
            {
                "event": "configuration_trial",
                "variant": {
                    "operation": "lgbm",
                    "parameter": "early_stopping_rounds",
                    "value": "callback",
                    "current_value": "discarded",
                },
                "candidate": candidate.candidate_id,
                "patch_hash": patch_hash,
                "stage": "final" if kept else "final_drop",
                "reason": final.reason,
            },
        )
        return {
            "ok": kept,
            "summary": {
                "change": change,
                "final_keep": kept,
                "final_target_delta": final.target_delta,
                "final_task_deltas": final.regression_deltas,
                "dev_target_delta": full.target_delta,
                "dev_task_deltas": full.regression_deltas,
                "patch": patch_path,
                "selection": "highest passing FINAL target_delta",
            },
            "checks": {"full_dev_keep": True, "final_keep": kept},
            "final": {
                "keep": kept,
                "reason": final.reason,
                "target_delta": final.target_delta,
                "regression_deltas": final.regression_deltas,
            },
        }
    finally:
        if experiment.is_dir():
            discard_experiment_checkout(experiment, workspace=workspace, source=base)


def _different(value: Any, current: Any) -> bool:
    if isinstance(value, float) and isinstance(current, (int, float)):
        return not math.isclose(value, float(current), rel_tol=0.0, abs_tol=1e-12)
    return value != current


def _numeric_scale(current: Any) -> list[float]:
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        return []
    current = float(current)
    if current <= 0 or not math.isfinite(current):
        return []
    return [current / 10.0, current * 10.0]


def generic_values(
    parameter: str,
    current: Any,
    *,
    classification: bool,
    operation: str = "",
) -> list[tuple[Any, str]]:
    """Small general alternatives ordered from conservative to broader."""

    if parameter == "min_samples_leaf" and current in (None, 1):
        return [
            (2, "mild leaf regularization reduces single-row variance"),
            (4, "stronger leaf regularization tests the same variance mechanism"),
        ]
    if parameter == "min_samples_split" and current in (None, 2):
        return [(4, "mild split regularization reduces fragile tiny partitions")]
    if parameter in {"min_data_in_leaf", "min_child_weight"} and current in (None, 0, 1, 1.0):
        return [
            (2, "require minimal support for a learned leaf"),
            (4, "test stronger support for a learned leaf"),
        ]
    if parameter == "reg_lambda" and current is None and operation == "xgboost":
        # XGBoost 3 uses lambda 1 when the parameter is unset. 0.01 removes it.
        return [
            (2.0, "increase L2 above the XGBoost default of 1"),
            (10.0, "test a stronger L2 alternative above the XGBoost default of 1"),
        ]
    if parameter in {"alpha", "l2_leaf_reg", "reg_lambda"}:
        if current in (None, 0, 0.0):
            return [
                (0.01, "introduce mild L2 regularization"),
                (0.1, "test a stronger L2-regularization alternative"),
            ]
        return [
            (value, "test a nearby regularization scale without changing the estimator")
            for value in _numeric_scale(current)
        ]
    if parameter == "reg_alpha" and current in (None, 0, 0.0):
        return [
            (0.01, "test mild sparse regularization"),
            (0.1, "test a stronger sparse-regularization alternative"),
        ]
    if parameter == "max_depth":
        if current == -1:
            return [(8, "bound model depth to reduce variance")]
        if current is None:
            # XGBoost 3 fits depth 6 when the parameter is unset. 8 is deeper.
            return [(4, "cap an unset depth below the common finite booster default")]
        if isinstance(current, int) and current > 2:
            return [(max(2, current - 2), "slightly reduce depth to test overfitting")]
    if parameter == "min_child_samples" and isinstance(current, int) and current > 1:
        return [
            (current * 2, "increase leaf support to reduce variance"),
            (max(2, current // 2), "allow finer leaves while retaining regularization"),
        ]
    if parameter == "max_features" and current == "sqrt":
        return [("log2", "test a nearby feature-subsampling regime")]
    if parameter == "learning_rate" and isinstance(current, (int, float)) and current > 0:
        return [
            (float(current) / 2.0, "slower updates may improve boosting generalization"),
            (float(current) * 2.0, "faster updates test the opposing bias/variance tradeoff"),
        ]
    if parameter == "num_leaves" and isinstance(current, int) and current > 4:
        return [
            (max(4, current // 2), "reduce tree capacity to test variance control"),
            (current * 2 + 1, "increase tree capacity to test the opposing bias mechanism"),
        ]
    if parameter in {"colsample_bytree", "subsample"} and current in (None, 1, 1.0):
        return [(0.8, "mild stochastic subsampling may improve ensemble generalization")]
    if parameter == "bagging_fraction" and isinstance(current, (int, float)):
        return [
            (0.7, "stronger row subsampling may reduce boosting variance"),
            (1.0, "disable row subsampling to test the opposing bias mechanism"),
        ]
    if parameter == "whiten" and current == "unit-variance":
        return [
            (
                "arbitrary-variance",
                "preserve independent-component scale instead of normalizing every component",
            )
        ]
    if parameter == "algorithm" and current == "parallel":
        return [("deflation", "test sequential independent-component extraction")]
    if parameter == "svd_solver" and current == "full":
        return [("randomized", "test the stable truncated randomized decomposition path")]
    if parameter == "extra_trees" and current is False:
        return [(True, "test randomized split thresholds as a variance-reduction alternative")]
    if parameter == "max_iter" and isinstance(current, int) and 0 < current < 2_000:
        return [(current * 2, "allow the iterative estimator more room to converge")]
    if parameter == "tol" and isinstance(current, (int, float)) and current > 0:
        return [(float(current) / 10.0, "test a stricter numerical convergence tolerance")]
    if parameter == "class_weight" and classification and current is None:
        return [("balanced", "use training-label frequencies to correct class imbalance")]
    return []


def parameter_surfaces(
    operation_hints: dict[str, tuple[str, ...]],
    scores: dict[str, ScoreResult],
) -> dict[str, dict[str, Any]]:
    """Merge repeated runtime observations without retaining task ids or scores."""

    surfaces: dict[str, dict[str, Any]] = {}
    for task_id, operations in operation_hints.items():
        result = scores.get(task_id)
        if result is None:
            continue
        for row in result.dataflow:
            operation = str(row.get("operation") or "")
            if operation not in operations:
                continue
            effective = dict(row.get("params") or {})
            # FEDOT wrappers such as CatBoost/LightGBM implementations do not
            # expose sklearn get_params, but every effective key in the traced
            # fit call is demonstrably accepted by that implementation.
            supported = tuple(
                str(name)
                for name in (row.get("supported_parameters") or effective.keys())
            )
            defaults = dict(row.get("estimator_defaults") or effective)
            if not supported or not defaults:
                continue
            surface = surfaces.setdefault(
                operation,
                {
                    "operation": operation,
                    "implementation": str(row.get("implementation") or ""),
                    "supported": set(),
                    "defaults": {},
                    "effective": {},
                    "uses": 0,
                },
            )
            surface["supported"].update(supported)
            surface["defaults"].update(defaults)
            surface["effective"].update(effective)
            surface["uses"] += 1
    return surfaces


def configuration_variant_budget(
    *,
    max_trials: int,
    max_operations: int,
) -> tuple[int, int]:
    """Keep a first value for every known parameter inside the queued prefix.

    Two scales of the early knobs otherwise fill a short list and a later
    parameter such as ``extra_trees`` is never proposed.
    """

    depth = len(_PARAMETER_PRIORITY) * 2
    return (
        max(depth, max_trials * 4),
        max(depth, max_trials * max(2, max_operations)),
    )


def propose_parameter_variants(
    operation_hints: dict[str, tuple[str, ...]],
    scores: dict[str, ScoreResult],
    *,
    problem_by_task: dict[str, str] | None = None,
    max_operations: int = 3,
    max_variants_per_operation: int = 2,
    max_total: int = 6,
) -> list[ParameterVariant]:
    surfaces = parameter_surfaces(operation_hints, scores)
    task_problems = problem_by_task or {}
    tasks_by_operation: dict[str, list[str]] = {}
    for task_id, operations in operation_hints.items():
        for operation in operations:
            tasks_by_operation.setdefault(operation, []).append(task_id)
    ordered = sorted(
        surfaces.values(),
        key=lambda row: (-len(tasks_by_operation.get(row["operation"], ())), row["operation"]),
    )
    variants: list[ParameterVariant] = []
    operations_with_variants = 0
    for surface in ordered:
        operation = str(surface["operation"])
        classification = any(
            task_problems.get(task_id) == "classification"
            for task_id in tasks_by_operation.get(operation, ())
        )
        operation_variants: list[ParameterVariant] = []
        ordered_parameters = sorted(
            surface["supported"],
            key=lambda name: (
                _PARAMETER_PRIORITY.index(name)
                if name in _PARAMETER_PRIORITY
                else len(_PARAMETER_PRIORITY),
                name,
            ),
        )
        for parameter in ordered_parameters:
            current = surface["effective"].get(
                parameter,
                surface["defaults"].get(parameter),
            )
            for value, rationale in generic_values(
                parameter,
                current,
                classification=classification,
                operation=operation,
            ):
                if not _different(value, current):
                    continue
                operation_variants.append(
                    ParameterVariant(
                        operation=operation,
                        parameter=parameter,
                        value=value,
                        current_value=current,
                        rationale=rationale,
                    )
                )
                if len(operation_variants) >= max(1, max_variants_per_operation):
                    break
            if len(operation_variants) >= max(1, max_variants_per_operation):
                break
        if operation_variants:
            operations_with_variants += 1
            variants.extend(operation_variants)
        if (
            len(variants) >= max(1, max_total)
            or operations_with_variants >= max(1, max_operations)
        ):
            break
    return variants[: max(1, max_total)]


def propose_parameter_bundles(
    variants: list[ParameterVariant],
    *,
    max_per_operation: int = 5,
) -> list[ParameterBundle]:
    """General coordinated alternatives for coupled boosting complexity knobs."""

    grouped: dict[str, dict[str, list[ParameterVariant]]] = {}
    for variant in variants:
        grouped.setdefault(variant.operation, {}).setdefault(variant.parameter, []).append(variant)

    pair_order = (
        ("learning_rate", "num_leaves"),
        ("min_child_samples", "num_leaves"),
        ("reg_lambda", "num_leaves"),
        ("reg_alpha", "num_leaves"),
        ("colsample_bytree", "subsample"),
    )

    def conservative(parameter: str, options: list[ParameterVariant]) -> ParameterVariant:
        if parameter in {"learning_rate", "num_leaves"}:
            lower = [row for row in options if isinstance(row.value, (int, float)) and isinstance(row.current_value, (int, float)) and row.value < row.current_value]
            if lower:
                return lower[0]
        if parameter == "min_child_samples":
            higher = [row for row in options if isinstance(row.value, (int, float)) and isinstance(row.current_value, (int, float)) and row.value > row.current_value]
            if higher:
                return higher[0]
        return options[0]

    bundles: list[ParameterBundle] = []
    for operation, by_parameter in grouped.items():
        operation_bundles = 0
        for left_name, right_name in pair_order:
            left = by_parameter.get(left_name)
            right = by_parameter.get(right_name)
            if not left or not right:
                continue
            selected = (conservative(left_name, left), conservative(right_name, right))
            bundles.append(
                ParameterBundle(
                    operation=operation,
                    variants=selected,
                    rationale=(
                        "coordinated conservative boosting update: "
                        + "; ".join(item.rationale for item in selected)
                    ),
                )
            )
            operation_bundles += 1
            if operation_bundles >= max(1, max_per_operation):
                break
    return bundles


_REFINEMENT_MARKER = "execution-guided refinement"


def spread_first_parameter_values(
    variants: list[ParameterVariant | ParameterBundle],
) -> list[ParameterVariant | ParameterBundle]:
    """Give every parameter one trial before a second scale of an earlier one.

    A broad boosting surface otherwise spends the whole budget on two scales of
    the first knob. The second scale stays queued, after the other parameters.
    A historical near-miss refinement is a second look at a parameter that
    already had a trial. It waits behind the second generic scale, which is
    the untried alternative rather than a neighborhood of a miss.
    """

    first: list[ParameterVariant | ParameterBundle] = []
    rest: list[ParameterVariant | ParameterBundle] = []
    refinements: list[ParameterVariant | ParameterBundle] = []
    seen: set[tuple[str, str]] = set()
    for variant in variants:
        if _REFINEMENT_MARKER in variant.rationale:
            refinements.append(variant)
            continue
        if isinstance(variant, ParameterBundle):
            first.append(variant)
            continue
        key = (variant.operation, variant.parameter)
        if key in seen:
            rest.append(variant)
        else:
            seen.add(key)
            first.append(variant)
    return [*first, *rest, *refinements]


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def measured_parameter_values(trials: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Values that already consumed a trial, including both sides of a bundle."""

    measured: set[tuple[str, str, str]] = set()
    for row in trials:
        if row.get("stage") in _NOT_MEASURED:
            continue
        payload = row.get("variant")
        if not isinstance(payload, dict):
            continue
        nested = payload.get("variants")
        parts = nested if isinstance(nested, list) else [payload]
        operation = str(payload.get("operation") or "")
        for part in parts:
            if not isinstance(part, dict) or "parameter" not in part:
                continue
            measured.add(
                (
                    str(part.get("operation") or operation),
                    str(part["parameter"]),
                    _value_key(part.get("value")),
                )
            )
    return measured


def defer_bundles_of_measured_values(
    variants: list[ParameterVariant | ParameterBundle],
    measured: set[tuple[str, str, str]],
) -> list[ParameterVariant | ParameterBundle]:
    """Keep a pair of already-tried values from blocking an unseen parameter."""

    if not measured:
        return variants
    leading: list[ParameterVariant | ParameterBundle] = []
    deferred: list[ParameterVariant | ParameterBundle] = []
    for variant in variants:
        parts = list(variant.variants) if isinstance(variant, ParameterBundle) else []
        if parts and all(
            (part.operation, part.parameter, _value_key(part.value)) in measured
            for part in parts
        ):
            deferred.append(variant)
        else:
            leading.append(variant)
    return [*leading, *deferred]


def _exact_zero(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == 0.0


def inert_parameter_values(trials: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Single values that left every measured task unchanged."""

    inert: set[tuple[str, str, str]] = set()
    for row in trials:
        if row.get("stage") in _NOT_MEASURED:
            continue
        payload = row.get("variant")
        decision = row.get("quick_decision")
        if (
            not isinstance(payload, dict)
            or isinstance(payload.get("variants"), list)
            or "parameter" not in payload
            or not isinstance(decision, dict)
        ):
            continue
        deltas = decision.get("regression_deltas") or {}
        if not deltas or not _exact_zero(decision.get("target_delta")):
            continue
        if not all(_exact_zero(delta) for delta in deltas.values()):
            continue
        inert.add(
            (
                str(payload.get("operation") or ""),
                str(payload["parameter"]),
                _value_key(payload.get("value")),
            )
        )
    return inert


def schedule_parameter_variants(
    singles: list[ParameterVariant],
    bundles: list[ParameterBundle],
) -> list[ParameterVariant | ParameterBundle]:
    """Place a coupled experiment as soon as all of its parts are available.

    Keeping every bundle behind the complete single-parameter queue makes the
    multi-parameter search effectively unreachable on estimators with a broad
    runtime surface.  The order below still measures each constituent first,
    but does not force unrelated later parameters to run before the coupled
    causal hypothesis.
    """

    scheduled: list[ParameterVariant | ParameterBundle] = []
    seen: list[ParameterVariant] = []
    pending = list(bundles)
    for single in singles:
        scheduled.append(single)
        seen.append(single)
        ready = [
            bundle
            for bundle in pending
            if all(part in seen for part in bundle.variants)
        ]
        scheduled.extend(ready)
        if ready:
            pending = [bundle for bundle in pending if bundle not in ready]
    scheduled.extend(pending)
    return scheduled


def _refined_numeric_value(
    current: Any,
    selected: Any,
    fraction: float,
) -> int | float | None:
    if (
        isinstance(current, bool)
        or isinstance(selected, bool)
        or not isinstance(current, (int, float))
        or not isinstance(selected, (int, float))
    ):
        return None
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in (current, selected)):
        return None
    value = float(current) + (float(selected) - float(current)) * fraction
    if isinstance(current, int) and isinstance(selected, int):
        refined: int | float = int(round(value))
    else:
        refined = round(value, 12)
    if refined <= 0 or not _different(refined, current) or not _different(refined, selected):
        return None
    return refined


def _step_farther(
    previous: dict | None,
    winner: ParameterVariant | ParameterBundle,
) -> ParameterVariant | None:
    """One step past a value that beat an earlier keep of the same parameter."""

    if previous is None or isinstance(winner, ParameterBundle):
        return None
    if _REFINEMENT_MARKER in winner.rationale:
        return None
    if (
        previous.get("operation") != winner.operation
        or previous.get("parameter") != winner.parameter
    ):
        return None
    previous_value = previous.get("value")
    current = winner.current_value
    selected = winner.value
    if (
        isinstance(previous_value, bool)
        or isinstance(current, bool)
        or isinstance(selected, bool)
        or not isinstance(previous_value, (int, float))
        or not isinstance(current, (int, float))
        or not isinstance(selected, (int, float))
    ):
        return None
    if abs(float(selected) - float(current)) <= abs(float(previous_value) - float(current)):
        return None
    farther = _refined_numeric_value(current, selected, 1.5)
    if farther is None:
        return None
    return ParameterVariant(
        operation=winner.operation,
        parameter=winner.parameter,
        value=farther,
        current_value=current,
        rationale=f"{_REFINEMENT_MARKER}: slightly farther in the promising direction",
    )


def refine_parameter_choice(
    variant: ParameterVariant | ParameterBundle,
) -> list[ParameterVariant | ParameterBundle]:
    """Create one bounded local neighborhood around a positive DEV near-miss."""

    if _REFINEMENT_MARKER in variant.rationale:
        return []
    parts = list(variant.variants) if isinstance(variant, ParameterBundle) else [variant]
    refined: list[ParameterVariant | ParameterBundle] = []
    for position, part in enumerate(parts):
        for fraction, direction in (
            (0.5, "between stock and the promising value"),
            (1.5, "slightly farther in the promising direction"),
        ):
            value = _refined_numeric_value(part.current_value, part.value, fraction)
            if value is None:
                continue
            changed = ParameterVariant(
                operation=part.operation,
                parameter=part.parameter,
                value=value,
                current_value=part.current_value,
                rationale=f"{_REFINEMENT_MARKER}: {direction}",
            )
            if isinstance(variant, ParameterBundle):
                new_parts = list(parts)
                new_parts[position] = changed
                refined.append(
                    ParameterBundle(
                        operation=variant.operation,
                        variants=tuple(new_parts),
                        rationale=(
                            f"{_REFINEMENT_MARKER} of positive multi-parameter DEV near-miss"
                        ),
                    )
                )
            else:
                refined.append(changed)
    return refined


def _variant_from_payload(payload: dict[str, Any]) -> ParameterVariant | ParameterBundle | None:
    try:
        operation = str(payload["operation"])
        nested = payload.get("variants")
        if isinstance(nested, list):
            parts = tuple(
                ParameterVariant(
                    operation=str(row.get("operation") or operation),
                    parameter=str(row["parameter"]),
                    value=row.get("value"),
                    current_value=row.get("current_value"),
                    rationale=str(row.get("rationale") or "historical configuration choice"),
                )
                for row in nested
                if isinstance(row, dict)
            )
            if not parts:
                return None
            return ParameterBundle(
                operation=operation,
                variants=parts,
                rationale=str(payload.get("rationale") or "historical parameter bundle"),
            )
        return ParameterVariant(
            operation=operation,
            parameter=str(payload["parameter"]),
            value=payload.get("value"),
            current_value=payload.get("current_value"),
            rationale=str(payload.get("rationale") or "historical configuration choice"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _positive_near_miss(row: dict[str, Any]) -> float | None:
    if row.get("stage") != "quick_dev":
        return None
    decision = row.get("quick_decision")
    delta = decision.get("target_delta") if isinstance(decision, dict) else None
    reason = str(row.get("reason") or (decision or {}).get("reason") or "")
    if delta is None:
        matched = re.search(r"target_delta\s+([+-]?[0-9]*\.?[0-9]+)", reason)
        delta = float(matched.group(1)) if matched else None
    try:
        numeric = float(delta)
    except (TypeError, ValueError):
        return None
    if numeric <= 0 or not reason.startswith("target_delta"):
        return None
    return numeric


def refinements_from_trials(trials: list[dict[str, Any]]) -> list[ParameterVariant | ParameterBundle]:
    """Rank historical DEV near-misses and turn them into fresh local choices."""

    ranked: list[tuple[float, ParameterVariant | ParameterBundle]] = []
    for row in trials:
        delta = _positive_near_miss(row)
        variant = _variant_from_payload(row.get("variant") or {})
        if delta is not None and variant is not None:
            ranked.append((delta, variant))
    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[ParameterVariant | ParameterBundle] = []
    for _delta, variant in ranked:
        out.extend(refine_parameter_choice(variant))
    return out


def _operation_block(text: str, operation: str) -> tuple[str, dict] | None:
    pattern = re.compile(rf'^  "{re.escape(operation)}"\s*:\s*', re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    try:
        value, consumed = decoder.raw_decode(text[match.end() :])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return text[match.start() : match.end() + consumed], value


def _format_operation(operation: str, params: dict) -> str:
    encoded = json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True)
    encoded = encoded.replace("\n", "\n  ")
    return f'  {json.dumps(operation)}: {encoded}'


def candidate_for_variant(
    source: Path,
    variant: ParameterVariant | ParameterBundle,
) -> PatchCandidate | None:
    path = source / DEFAULTS_FILE
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    parts = list(variant.variants) if isinstance(variant, ParameterBundle) else [variant]
    located = _operation_block(text, variant.operation)
    if located is None:
        params = {item.parameter: item.value for item in parts}
        first_operation = next(iter(payload), None) if isinstance(payload, dict) else None
        first = _operation_block(text, str(first_operation)) if first_operation else None
        if first is None:
            old_code = text
            new_code = json.dumps(
                {variant.operation: params},
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        else:
            old_code, _first_params = first
            new_code = _format_operation(variant.operation, params) + ",\n" + old_code
    else:
        old_code, params = located
        params = dict(params)
        for item in parts:
            params[item.parameter] = item.value
        new_code = _format_operation(variant.operation, params)
    token = json.dumps(
        [variant.operation, [(item.parameter, item.value) for item in parts]],
        sort_keys=True,
        default=str,
    )
    candidate_id = "config-" + hashlib.sha256(token.encode()).hexdigest()[:12]
    return PatchCandidate(
        candidate_id=candidate_id,
        edits=[PatchEdit(DEFAULTS_FILE, old_code, new_code)],
        rationale=(
            f"{variant.operation}: "
            + "; ".join(
                f"{item.parameter} {item.current_value!r} -> {item.value!r}"
                for item in parts
            )
            + f"; {variant.rationale}"
        ),
        contract=(
            f"only selected defaults of {variant.operation} change; "
            "explicit user parameters remain authoritative"
        ),
    )


def configuration_probe_code(operation: str) -> str:
    operation_literal = json.dumps(operation)
    return "\n".join(
        (
            "import json",
            "from fedot.core.repository.default_params_repository import DefaultOperationParamsRepository",
            f"params = DefaultOperationParamsRepository().get_default_params_for_operation({operation_literal})",
            'print("EVOLVE_OBSERVATION=" + json.dumps(params, sort_keys=True, default=str))',
        )
    )


def _score_rows(scores: dict[str, ScoreResult]) -> dict[str, dict]:
    return {
        task_id: {
            "status": result.status,
            "score": result.score,
            "detail": result.detail,
            "seed": result.seed,
            "env_hash": result.env_hash,
        }
        for task_id, result in scores.items()
    }


def _unmeasured_configuration_task(
    stock: dict[str, ScoreResult],
    operation_hints: dict[str, tuple[str, ...]],
) -> str | None:
    """Name one hinted task that has no measured parameter surface.

    A measured pack with no tunable keys is a real empty search. A pack whose
    stock rows never reached ``ok`` never observed a surface, so it must not
    be recorded as ``no_configuration_keep``.
    """

    if parameter_surfaces(operation_hints, stock):
        return None
    for task_id in operation_hints:
        result = stock.get(task_id)
        if result is None or result.status != "ok":
            return task_id
    return None


def _single_task_lift(deltas: dict) -> bool:
    """True when one lift task is up and every other measured task is not.

    ``kc2->xgboost`` carried ``reg_lambda`` 0.01 to 0.0711 while the other two
    lift tasks fell. That candidate then lost 0.1462 on the shadow split.
    A one-task surface stays eligible: the kept rf default was measured on
    tasks that were both up.
    """

    numeric = [
        value
        for value in deltas.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numeric) < 2:
        return False
    return sum(value > 0 for value in numeric) < 2


def search_configuration_variants(
    source: Path,
    workspace: Path,
    *,
    run_id: str,
    source_hash: str,
    stock: dict[str, ScoreResult],
    operation_hints: dict[str, tuple[str, ...]],
    problem_by_task: dict[str, str],
    lift_ids: tuple[str, ...],
    protect_ids: tuple[str, ...],
    baseline_tests: TestResult,
    skip_patch_hashes: set[str] | None = None,
    prior_trials: list[dict[str, Any]] | None = None,
    max_trials: int = 6,
    max_operations: int = 3,
    dev_seed: int = 42,
    retain_dev_keeps: bool = False,
    proposal_source: Path | None = None,
) -> ConfigurationSearchOutcome:
    """Try bounded generic defaults; return only a full-DEV/test-gated winner.

    No FINAL call occurs here.  Every candidate gets a fresh checkout.  A quick
    affected-workload pass is only a cost filter; a winner must also pass the
    complete DEV protect suite, the public behavior probe, and comparative tests.
    """

    unmeasured = _unmeasured_configuration_task(stock, operation_hints)
    if unmeasured is not None:
        return ConfigurationSearchOutcome(
            decision=Decision(
                False,
                f"configuration_stock_unmeasured {unmeasured}",
                None,
                infrastructure_error=True,
            )
        )

    from fedotllm.agents.evolve.execution.checkout import (
        create_experiment_checkout,
        discard_experiment_checkout,
    )
    from fedotllm.agents.evolve.storage.hypothesis import normalized_patch_hash
    from fedotllm.agents.evolve.storage.journal import append_journal
    from fedotllm.agents.evolve.evaluation.judge import (
        confirm_candidate_tests,
        measure_fedot_tests,
        measure_patched,
        measure_stock,
        normalize_test_result,
        verdict,
    )
    from fedotllm.agents.evolve.controller.campaign import compare_behavior_probe
    from fedotllm.agents.evolve.execution.patch import apply_patch

    # Build a deeper ordered queue than the per-campaign execution budget.
    # Durable duplicates are filtered below and must not prevent later
    # parameters of the same estimator from ever being reached.
    per_operation, total = configuration_variant_budget(
        max_trials=max_trials,
        max_operations=max_operations,
    )
    lift_operations = {
        operation
        for task_id in lift_ids
        for operation in operation_hints.get(task_id, ())
    }
    proposal_hints = {
        task_id: tuple(
            operation for operation in operations if operation in lift_operations
        )
        for task_id, operations in operation_hints.items()
        if any(operation in lift_operations for operation in operations)
    }
    single_variants = propose_parameter_variants(
        proposal_hints,
        stock,
        problem_by_task=problem_by_task,
        max_operations=max_operations,
        # Keep enough alternatives behind the already-tried prefix of one
        # estimator.  Per-operation truncation must not recreate the same
        # durable-memory starvation at a smaller level.
        max_variants_per_operation=per_operation,
        max_total=total,
    )
    variants = defer_bundles_of_measured_values(
        spread_first_parameter_values(
            [
                *refinements_from_trials(prior_trials or []),
                *schedule_parameter_variants(
                    single_variants,
                    propose_parameter_bundles(single_variants),
                ),
            ]
        ),
        measured_parameter_values(prior_trials or []),
    )
    proposal = proposal_source or source
    locked = parameters_already_in_defaults(source, proposal)
    skipped = set(skip_patch_hashes or ())
    trials: list[dict] = []
    best: ConfigurationSearchOutcome | None = None
    retained: list[DevKeep] = []
    full_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    last = Decision(False, "no_configuration_keep", None)
    evaluated_trials = 0
    inert = inert_parameter_values(prior_trials or [])
    queued_signatures = {
        json.dumps(asdict(variant), sort_keys=True, default=str)
        for variant in variants
    }
    cursor = 0
    index = 0
    while cursor < len(variants):
        variant = variants[cursor]
        cursor += 1
        index += 1
        parts = (
            list(variant.variants)
            if isinstance(variant, ParameterBundle)
            else [variant]
        )
        if any(part.parameter in locked.get(part.operation, set()) for part in parts):
            row = {
                "event": "configuration_trial",
                "index": index,
                "variant": asdict(variant),
                "stage": "kept_parameter",
                "reason": "already in the remembered default",
            }
            trials.append(row)
            append_journal(workspace / "configuration_trials.jsonl", row)
            continue
        candidate = candidate_for_variant(proposal, variant)
        if candidate is None:
            continue
        patch_hash = normalized_patch_hash(candidate, source_hash)
        if patch_hash in skipped:
            row = {
                "event": "configuration_trial",
                "index": index,
                "variant": asdict(variant),
                "candidate": candidate.candidate_id,
                "patch_hash": patch_hash,
                "stage": "dedup",
                "reason": "previously_evaluated_patch",
            }
            trials.append(row)
            append_journal(workspace / "configuration_trials.jsonl", row)
            continue
        if isinstance(variant, ParameterBundle) and any(
            (part.operation, part.parameter, _value_key(part.value)) in inert
            for part in variant.variants
        ):
            row = {
                "event": "configuration_trial",
                "index": index,
                "variant": asdict(variant),
                "candidate": candidate.candidate_id,
                "patch_hash": patch_hash,
                "stage": "inert_part",
                "reason": "paired with a parameter value that did not move any task",
            }
            trials.append(row)
            append_journal(workspace / "configuration_trials.jsonl", row)
            continue
        if evaluated_trials >= max(1, max_trials):
            break
        evaluated_trials += 1
        skipped.add(patch_hash)
        experiment = create_experiment_checkout(
            proposal,
            workspace,
            run_id=run_id,
            candidate_id=f"configuration-{index}-{candidate.candidate_id}",
        )
        keep_experiment = False
        try:
            diagnostics: list[str] = []
            if not apply_patch(experiment, candidate, diagnostics=diagnostics):
                last = Decision(False, "configuration_patch_invalid", None)
                row = {
                    "event": "configuration_trial",
                    "index": index,
                    "variant": asdict(variant),
                    "candidate": candidate.candidate_id,
                    "patch_hash": patch_hash,
                    "stage": "patch",
                    "reason": "; ".join(diagnostics),
                }
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue
            affected = tuple(
                task_id
                for task_id in full_ids
                if variant.operation in operation_hints.get(task_id, ())
            )
            if not affected:
                last = Decision(False, "configuration_no_affected_workload", None)
                row = {
                    "event": "configuration_trial",
                    "index": index,
                    "variant": asdict(variant),
                    "candidate": candidate.candidate_id,
                    "patch_hash": patch_hash,
                    "stage": "scope",
                    "reason": last.reason,
                }
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue
            quick_patched = measure_patched(
                affected,
                checkout=experiment,
                split="dev",
                seed=dev_seed,
            )
            quick_stock = {task_id: stock[task_id] for task_id in affected}
            quick = verdict(
                quick_stock,
                quick_patched,
                lift_ids=affected,
                protect_ids=affected,
            )
            row = {
                "event": "configuration_trial",
                "index": index,
                "variant": asdict(variant),
                "candidate": candidate.candidate_id,
                "patch_hash": patch_hash,
                "stage": "quick_dev",
                "quick_decision": asdict(quick),
                "quick_scores": _score_rows(quick_patched),
            }
            if quick.keep and _single_task_lift(quick.regression_deltas):
                quick = Decision(
                    False,
                    "single_task_lift",
                    quick.target_delta,
                    regression_deltas=dict(quick.regression_deltas),
                )
                row["quick_decision"] = asdict(quick)
                row["reason"] = quick.reason
            if not quick.keep:
                last = quick
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                if (
                    not isinstance(variant, ParameterBundle)
                    and quick.regression_deltas
                    and _exact_zero(quick.target_delta)
                    and all(
                        _exact_zero(delta) for delta in quick.regression_deltas.values()
                    )
                ):
                    inert.add(
                        (variant.operation, variant.parameter, _value_key(variant.value))
                    )
                if (
                    quick.target_delta is not None
                    and quick.target_delta > 0
                    and str(quick.reason).startswith("target_delta")
                ):
                    fresh: list[ParameterVariant | ParameterBundle] = []
                    for refined in refine_parameter_choice(variant):
                        signature = json.dumps(asdict(refined), sort_keys=True, default=str)
                        if signature in queued_signatures:
                            continue
                        queued_signatures.add(signature)
                        fresh.append(refined)
                    # Keep the refinement, but do not let it jump ahead of the next
                    # parameter. A refinement never refines another refinement.
                    variants.extend(fresh)
                continue

            full_patched = measure_patched(
                full_ids,
                checkout=experiment,
                split="dev",
                seed=dev_seed,
            )
            full = verdict(
                stock,
                full_patched,
                lift_ids=lift_ids,
                protect_ids=protect_ids,
            )
            row["stage"] = "full_dev"
            row["full_decision"] = asdict(full)
            row["full_scores"] = _score_rows(full_patched)
            if not full.keep:
                last = full
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue

            # A model-seed confirmation still evaluates the same frozen DEV
            # rows.  Reject a split-specific default before expensive pytest and
            # before returning control to the outer campaign.  Keeping this
            # check inside the queue means a SHADOW drop advances immediately to
            # the next parameter variant instead of requiring another run.
            shadow_stock = measure_stock(
                affected,
                checkout=proposal_source or source,
                split="shadow",
                seed=dev_seed,
            )
            shadow_patched = measure_patched(
                affected,
                checkout=experiment,
                split="shadow",
                seed=dev_seed,
            )
            shadow = verdict(
                shadow_stock,
                shadow_patched,
                lift_ids=affected,
                protect_ids=affected,
            )
            row["stage"] = "shadow_dev"
            row["shadow_decision"] = asdict(shadow)
            row["shadow_scores"] = _score_rows(shadow_patched)
            if not shadow.keep:
                last = shadow
                row["reason"] = shadow.reason
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue

            probe = compare_behavior_probe(
                source,
                experiment,
                configuration_probe_code(variant.operation),
            )
            row["behavior_probe"] = probe
            if probe.get("status") != "changed":
                last = Decision(False, f"behavior_probe_{probe.get('status')}", None)
                row["stage"] = "behavior_probe"
                row["reason"] = last.reason
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue

            # Shadow and FINAL are different splits. A shadow keep can still
            # regress on FINAL, so score that split before the pytest suite.
            final_stock = measure_stock(
                affected,
                checkout=proposal_source or source,
                split="final",
                seed=dev_seed,
            )
            final_patched = measure_patched(
                affected,
                checkout=experiment,
                split="final",
                seed=dev_seed,
            )
            final = verdict(
                final_stock,
                final_patched,
                lift_ids=affected,
                protect_ids=affected,
            )
            row["final_decision"] = asdict(final)
            row["final_scores"] = _score_rows(final_patched)
            if not final.keep:
                last = final
                row["stage"] = "final"
                row["reason"] = final.reason
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                continue

            first_tests = normalize_test_result(measure_fedot_tests(experiment))
            after_tests, blocked, test_attempts = confirm_candidate_tests(
                baseline_tests,
                experiment,
                first=first_tests,
                runner=measure_fedot_tests,
            )
            row["tests"] = asdict(after_tests)
            row["test_attempts"] = [
                {
                    "status": item.status,
                    "failed_nodes": sorted(item.failed_nodes),
                    "duration_s": item.duration_s,
                }
                for item in test_attempts
            ]
            if blocked is not None:
                last = blocked
                row["stage"] = "tests"
                row["reason"] = blocked.reason
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                if blocked.infrastructure_error:
                    if best is not None:
                        return best
                    return ConfigurationSearchOutcome(
                        decision=blocked,
                        tests=after_tests,
                        trials=trials,
                    )
                continue

            incumbent = None if best is None else best.decision.target_delta
            if incumbent is not None and (
                full.target_delta is None or full.target_delta <= incumbent
            ):
                row["stage"] = "dev_keep_not_better"
                row["reason"] = full.reason
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
                # DEV rank is not FINAL rank. Keep the checkout when the caller
                # will score every passing DEV on FINAL.
                if retain_dev_keeps:
                    full.experiment_id = experiment.name
                    candidate.behavior_probe = configuration_probe_code(variant.operation)
                    retained.append(
                        DevKeep(
                            candidate=candidate,
                            experiment=experiment,
                            decision=full,
                            tests=after_tests,
                            behavior_probe=probe,
                            patch_path=None,
                            variant=asdict(variant),
                        )
                    )
                    if best is not None:
                        best.dev_keeps = retained
                    keep_experiment = True
                continue

            if (
                not retain_dev_keeps
                and best is not None
                and best.experiment is not None
            ):
                discard_experiment_checkout(
                    best.experiment,
                    workspace=workspace,
                    source=proposal,
                )
            previous_keep = next(
                (
                    row.get("variant")
                    for row in reversed(trials)
                    if row.get("stage") == "dev_keep"
                ),
                None,
            )
            farther = _step_farther(previous_keep, variant)
            if farther is not None:
                signature = json.dumps(asdict(farther), sort_keys=True, default=str)
                if signature not in queued_signatures:
                    queued_signatures.add(signature)
                    variants.insert(cursor, farther)
            full.experiment_id = experiment.name
            candidate.behavior_probe = configuration_probe_code(variant.operation)
            patch_path = write_kept_configuration_patch(
                workspace,
                candidate,
                source=source,
                experiment=experiment,
            )
            row["stage"] = "dev_keep"
            row["reason"] = full.reason
            row["patch"] = str(patch_path)
            trials.append(row)
            append_journal(workspace / "configuration_trials.jsonl", row)
            if retain_dev_keeps:
                retained.append(
                    DevKeep(
                        candidate=candidate,
                        experiment=experiment,
                        decision=full,
                        tests=after_tests,
                        behavior_probe=probe,
                        patch_path=patch_path,
                        variant=asdict(variant),
                    )
                )
            best = ConfigurationSearchOutcome(
                candidate=candidate,
                experiment=experiment,
                decision=full,
                tests=after_tests,
                patched=full_patched,
                behavior_probe=probe,
                trials=trials,
                patch_path=patch_path,
                dev_keeps=retained,
            )
            keep_experiment = True
            last = full
        finally:
            if not keep_experiment:
                discard_experiment_checkout(
                    experiment,
                    workspace=workspace,
                    source=proposal,
                )
    if best is not None:
        return best
    return ConfigurationSearchOutcome(decision=last, trials=trials)
