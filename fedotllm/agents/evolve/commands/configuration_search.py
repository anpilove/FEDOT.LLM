"""Deterministic, evaluator-blind variants for executed operation defaults.

The LLM should identify a causal operation/mechanism.  It should not need to
guess one exact scalar value.  This module derives a small, general set of mild
alternatives from the estimator's real ``get_params(deep=False)`` surface.  It
never reads scores, datasets, benchmark cases, or FINAL.
"""

from __future__ import annotations

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
        if current in (None, -1):
            return [(8, "bound model depth to reduce variance")]
        if isinstance(current, int) and current > 2:
            return [(max(2, current - 2), "slightly reduce depth to test overfitting")]
    if parameter == "min_child_samples" and isinstance(current, int) and current > 1:
        return [
            (max(2, current // 2), "allow finer leaves while retaining regularization"),
            (current * 2, "increase leaf support to reduce variance"),
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


_REFINEMENT_MARKER = "execution-guided refinement"


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
) -> ConfigurationSearchOutcome:
    """Try bounded generic defaults; return only a full-DEV/test-gated winner.

    No FINAL call occurs here.  Every candidate gets a fresh checkout.  A quick
    affected-workload pass is only a cost filter; a winner must also pass the
    complete DEV protect suite, the public behavior probe, and comparative tests.
    """

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
    single_variants = propose_parameter_variants(
        operation_hints,
        stock,
        problem_by_task=problem_by_task,
        max_operations=max_operations,
        # Keep enough alternatives behind the already-tried prefix of one
        # estimator.  Per-operation truncation must not recreate the same
        # durable-memory starvation at a smaller level.
        max_variants_per_operation=max(16, max_trials * 4),
        max_total=max(16, max_trials * max(2, max_operations)),
    )
    variants: list[ParameterVariant | ParameterBundle] = [
        *refinements_from_trials(prior_trials or []),
        *schedule_parameter_variants(
            single_variants,
            propose_parameter_bundles(single_variants),
        ),
    ]
    skipped = set(skip_patch_hashes or ())
    trials: list[dict] = []
    full_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    last = Decision(False, "no_configuration_keep", None)
    evaluated_trials = 0
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
        candidate = candidate_for_variant(source, variant)
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
        if evaluated_trials >= max(1, max_trials):
            break
        evaluated_trials += 1
        skipped.add(patch_hash)
        experiment = create_experiment_checkout(
            source,
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
            if not quick.keep:
                last = quick
                trials.append(row)
                append_journal(workspace / "configuration_trials.jsonl", row)
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
                    # Use DEV feedback immediately. Depth is bounded because a
                    # refinement never refines another refinement.
                    variants[cursor:cursor] = fresh
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
                checkout=source,
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
                    return ConfigurationSearchOutcome(
                        decision=blocked,
                        tests=after_tests,
                        trials=trials,
                    )
                continue

            full.experiment_id = experiment.name
            candidate.behavior_probe = configuration_probe_code(variant.operation)
            row["stage"] = "dev_keep"
            row["reason"] = full.reason
            trials.append(row)
            append_journal(workspace / "configuration_trials.jsonl", row)
            keep_experiment = True
            return ConfigurationSearchOutcome(
                candidate=candidate,
                experiment=experiment,
                decision=full,
                tests=after_tests,
                patched=full_patched,
                behavior_probe=probe,
                trials=trials,
            )
        finally:
            if not keep_experiment:
                discard_experiment_checkout(
                    experiment,
                    workspace=workspace,
                    source=source,
                )
    return ConfigurationSearchOutcome(decision=last, trials=trials)
