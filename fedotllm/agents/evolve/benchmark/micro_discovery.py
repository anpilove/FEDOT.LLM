"""Private end-to-end discovery benchmark over known FEDOT defects.

Unlike ``micro_agent``, this benchmark does not reveal the responsible file.
The model first selects one of four causally plausible files, then receives the
selected file and may localize and repair it. Controller-owned probes and
oracles never enter model prompts.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fedotllm.agents.evolve.benchmark.micro import (
    MicroCase,
    validate_observation,
)
from fedotllm.agents.evolve.benchmark.micro_agent import (
    Architecture,
    _Calls,
    _localization_matches,
    _propose,
    _selected_cases,
    _usage_requests,
)
from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.discovery.navigation import architecture_cards
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.execution.smoke import import_error
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit


class FileSelection(BaseModel):
    selected_index: int = Field(ge=0)
    mechanism: str = ""
    evidence: str = ""


_POOLS: dict[str, tuple[str, ...]] = {
    "partial_poly_params": (
        "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py",
        "fedot/core/operations/evaluation/common_preprocessing.py",
        "fedot/core/pipelines/tuning/search_space.py",
        "fedot/core/operations/hyperparameters_preprocessing.py",
    ),
    "lda_effective_solver": (
        "fedot/core/operations/evaluation/classification.py",
        "fedot/core/operations/evaluation/operation_implementations/models/discriminant_analysis.py",
        "fedot/core/pipelines/tuning/search_space.py",
        "fedot/core/operations/evaluation/operation_implementations/implementation_interfaces.py",
    ),
    "lagged_reproducibility": (
        "fedot/core/operations/evaluation/time_series.py",
        "fedot/utilities/window_size_selector.py",
        "fedot/core/operations/evaluation/operation_implementations/data_operations/ts_transformations.py",
        "fedot/core/operations/hyperparameters_preprocessing.py",
    ),
    "polyfit_parameter_identity": (
        "fedot/core/pipelines/node.py",
        "fedot/core/operations/operation_parameters.py",
        "fedot/core/operations/evaluation/time_series.py",
        "fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/poly.py",
    ),
    "nonfinite_target_preprocessing": (
        "fedot/preprocessing/preprocessing.py",
        "fedot/core/data/data_preprocessing.py",
        "fedot/preprocessing/data_types.py",
        "fedot/core/data/data.py",
    ),
    "merge_parent_index_alignment": (
        "fedot/core/data/array_utilities.py",
        "fedot/core/data/merge/data_merger.py",
        "fedot/core/data/merge/supplementary_data_merger.py",
        "fedot/core/data/data.py",
    ),
    "single_column_multits_lagged": (
        "fedot/core/data/multi_modal.py",
        "fedot/core/operations/evaluation/operation_implementations/data_operations/ts_transformations.py",
        "fedot/core/operations/evaluation/time_series.py",
        "fedot/core/repository/tasks.py",
    ),
}

_FILE_PROMPT = """A FEDOT public behavior violates this contract:
{symptom}

Choose the single source file most likely to own the violated contract. The
four candidates are structurally related, so base the choice on lifecycle,
parameter ownership, and data flow visible in their architecture cards. Return
the zero-based candidate index and a concrete mechanism. Do not request source
code yet.

Candidate architecture cards:
{cards}"""

_REPAIR_PROMPT = """A FEDOT public behavior violates this contract:
{symptom}

The controller selected this source file for deeper inspection:
{file_path}

Locate the responsible mechanism in this file and make the smallest safe
correction. The private reproduction and expected output are unavailable.

FEDOT architecture card:
{card}

Complete selected source file:
--- {file_path} ---
{source}"""


def discovery_pools() -> dict[str, tuple[str, ...]]:
    """Expose immutable pool metadata for offline integrity tests."""

    return dict(_POOLS)


def _case_repaired(stages: list[dict[str, Any]]) -> bool:
    """Behavioral repair is authoritative; symbol naming is diagnostic only."""

    required = {
        "stock_probe",
        "file_localization",
        "patch",
        "import_gate",
        "behavior_probe",
    }
    return required <= {stage["stage"] for stage in stages} and all(
        stage["status"] == "passed"
        for stage in stages
        if stage["stage"] in required
    )


def _cards(source: Path, paths: tuple[str, ...]) -> str:
    return architecture_cards(source, paths, max_chars_per_file=3_000)


def _repair_context(case: MicroCase, source: Path, file_path: str) -> tuple[str, int]:
    target = source / file_path
    text = target.read_text(encoding="utf-8", errors="replace")
    card = build_fedot_context(source, file_path)
    return (
        _REPAIR_PROMPT.format(
            symptom=case.prompt.symptom,
            file_path=file_path,
            card=card.render(max_chars=6_000) if card is not None else "",
            source=text,
        ),
        len(text.splitlines()),
    )


def run_micro_discovery_benchmark(
    source: Path,
    workspace: Path,
    *,
    inference: Any,
    architecture: Architecture = "monolith",
    case_ids: Iterable[str] | None = None,
    committee_size: int = 3,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    before = source_fingerprint(source)
    requests_before = _usage_requests(inference)
    calls = _Calls(inference)
    rows: list[dict[str, Any]] = []
    run_id = f"micro-discovery-{uuid.uuid4().hex[:8]}"

    for case in _selected_cases(case_ids):
        stages: list[dict[str, Any]] = []
        stock = validate_observation(
            case, run_fedot_snippet(source, case.behavior_probe), patched=False
        )
        stages.append(
            {"stage": "stock_probe", "status": stock.status, "reason": stock.reason}
        )
        if not stock.passed:
            rows.append({"case_id": case.prompt.case_id, "ok": False, "stages": stages})
            continue

        pool = _POOLS[case.prompt.case_id]
        try:
            selection = calls.create(
                _FILE_PROMPT.format(
                    symptom=case.prompt.symptom, cards=_cards(source, pool)
                ),
                FileSelection,
                stage="file_localization",
                metadata={"benchmark": "micro_discovery", "candidate_count": len(pool)},
            )
        except Exception as exc:
            stages.append(
                {
                    "stage": "file_localization",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
            for stage in (
                "symbol_localization",
                "patch",
                "import_gate",
                "behavior_probe",
            ):
                stages.append(
                    {
                        "stage": stage,
                        "status": "not_run",
                        "reason": "file localization failed",
                    }
                )
            rows.append({"case_id": case.prompt.case_id, "ok": False, "stages": stages})
            continue
        selected = (
            pool[selection.selected_index]
            if selection.selected_index < len(pool)
            else ""
        )
        file_hit = selected == case.prompt.file_path
        stages.append(
            {
                "stage": "file_localization",
                "status": "passed" if file_hit else "failed",
                "selected_index": selection.selected_index,
                "selected_file": selected,
                "reason": ""
                if file_hit
                else "selected file did not match private target",
            }
        )
        if not file_hit:
            for stage in (
                "symbol_localization",
                "patch",
                "import_gate",
                "behavior_probe",
            ):
                stages.append(
                    {"stage": stage, "status": "not_run", "reason": "wrong file"}
                )
            rows.append({"case_id": case.prompt.case_id, "ok": False, "stages": stages})
            continue

        experiment = create_experiment_checkout(
            source, workspace, run_id=run_id, candidate_id=case.prompt.case_id
        )
        try:
            visible, file_lines = _repair_context(case, experiment, selected)
            try:
                localization, proposed = _propose(
                    architecture,
                    calls,
                    visible,
                    case_id=case.prompt.case_id,
                    committee_size=committee_size,
                )
            except Exception as exc:
                stages.append(
                    {
                        "stage": "symbol_localization",
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                )
                stages.extend(
                    {
                        "stage": name,
                        "status": "not_run",
                        "reason": "repair proposal failed",
                    }
                    for name in ("patch", "import_gate", "behavior_probe")
                )
                rows.append(
                    {"case_id": case.prompt.case_id, "ok": False, "stages": stages}
                )
                continue
            symbol_hit = bool(
                localization.symbol.strip()
                and localization.mechanism.strip()
                and 1 <= localization.line <= max(1, file_lines)
                and _localization_matches(case, localization)
            )
            stages.append(
                {
                    "stage": "symbol_localization",
                    "status": "passed" if symbol_hit else "failed",
                    "symbol": localization.symbol[:300],
                    "line": localization.line,
                    "reason": ""
                    if symbol_hit
                    else "symbol did not match private target",
                }
            )
            candidate = PatchCandidate(
                candidate_id=f"micro-discovery-{uuid.uuid4().hex[:8]}",
                edits=[PatchEdit(selected, proposed.old_code, proposed.new_code)],
                rationale=proposed.rationale,
            )
            diagnostics: list[str] = []
            applied = apply_patch(experiment, candidate, diagnostics=diagnostics)
            stages.append(
                {
                    "stage": "patch",
                    "status": "passed" if applied else "failed",
                    "reason": "; ".join(diagnostics)[:1000],
                }
            )
            if not applied:
                stages.extend(
                    {"stage": name, "status": "not_run", "reason": "patch failed"}
                    for name in ("import_gate", "behavior_probe")
                )
                rows.append(
                    {"case_id": case.prompt.case_id, "ok": False, "stages": stages}
                )
                continue
            smoke_error = import_error(experiment, selected)
            stages.append(
                {
                    "stage": "import_gate",
                    "status": "failed" if smoke_error else "passed",
                    "reason": smoke_error or "",
                }
            )
            if smoke_error:
                stages.append(
                    {
                        "stage": "behavior_probe",
                        "status": "not_run",
                        "reason": "import failed",
                    }
                )
                rows.append(
                    {"case_id": case.prompt.case_id, "ok": False, "stages": stages}
                )
                continue
            patched = validate_observation(
                case, run_fedot_snippet(experiment, case.behavior_probe), patched=True
            )
            stages.append(
                {
                    "stage": "behavior_probe",
                    "status": patched.status,
                    "reason": patched.reason,
                    "observation": patched.observation,
                }
            )
            repaired = _case_repaired(stages)
            rows.append(
                {
                    "case_id": case.prompt.case_id,
                    "ok": repaired,
                    "symbol_localization_diagnostic": symbol_hit,
                    "stages": stages,
                }
            )
        finally:
            discard_experiment_checkout(experiment, workspace=workspace, source=source)

    def count(stage: str, status: str) -> int:
        return sum(
            item["status"] == status
            for row in rows
            for item in row["stages"]
            if item["stage"] == stage
        )

    total = len(rows)
    file_hits = count("file_localization", "passed")
    symbol_hits = count("symbol_localization", "passed")
    repairs = count("behavior_probe", "passed")
    import_failures = count("import_gate", "failed")
    immutable = source_fingerprint(source) == before
    metrics = {
        "file_recall_at_1": file_hits / total if total else 0.0,
        "symbol_recall_given_file": symbol_hits / file_hits if file_hits else 0.0,
        "repair_rate_given_localized": repairs / file_hits if file_hits else 0.0,
        "end_to_end_repair_rate": repairs / total if total else 0.0,
    }
    infrastructure_markers = (
        "llmrequesttimeout",
        "apierror",
        "budget_exhausted",
        "provider_error",
        "provider_policy",
        "connectionerror",
    )
    infrastructure_failures = sum(
        stage["status"] == "failed"
        and any(
            token in str(stage.get("reason") or "").lower()
            for token in infrastructure_markers
        )
        for row in rows
        for stage in row["stages"]
    )
    execution_ok = bool(total) and not infrastructure_failures and immutable
    quality_target_met = (
        bool(total)
        and file_hits >= (3 * total + 3) // 4
        and repairs * 3 >= file_hits * 2
        and not import_failures
    )
    payload = {
        "schema_version": 1,
        "component": "micro-discovery",
        "architecture": architecture,
        "ok": execution_ok and quality_target_met,
        "execution_ok": execution_ok,
        "quality_target_met": quality_target_met,
        "immutable_source": immutable,
        "metrics": metrics,
        "counts": {
            "cases": total,
            "file_hits": file_hits,
            "symbol_hits": symbol_hits,
            "repairs": repairs,
            "import_failures": import_failures,
            "provider_requests": max(0, _usage_requests(inference) - requests_before),
            "structured_calls": sum(calls.attempted.values()),
            "infrastructure_failures": infrastructure_failures,
        },
        "cases": rows,
    }
    output = artifact_path or workspace / f"micro_discovery_{architecture}.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
