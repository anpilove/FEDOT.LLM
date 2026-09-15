"""LLM architecture benchmark over the private controller-owned micro cases.

The model sees only a case's public symptom and the complete allowed FEDOT
source file.  Private observations and probes remain controller data throughout
the run.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fedotllm.agents.evolve.benchmark.micro import (
    MicroCase,
    micro_cases,
    model_facing_context,
    validate_observation,
)
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    source_fingerprint,
)
from fedotllm.agents.evolve.execution.patch import apply_patch
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context
from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit


Architecture = Literal["monolith", "staged", "committee"]
ARCHITECTURES: tuple[Architecture, ...] = ("monolith", "staged", "committee")


class LocalizationProposal(BaseModel):
    """Source-level causal hypothesis, without an edit."""

    symbol: str = ""
    line: int = Field(default=1, ge=1)
    mechanism: str = ""
    proposed_change: str = ""


class CriticSelection(BaseModel):
    """Zero-based selection from independently produced hypotheses."""

    selected_index: int = Field(ge=0)
    reason: str = ""


class PatchProposal(BaseModel):
    """One exact SEARCH/REPLACE in the controller-selected source file."""

    old_code: str = ""
    new_code: str = ""
    rationale: str = ""


class MonolithProposal(LocalizationProposal, PatchProposal):
    """One-call localization and patch used by the monolith arm."""


_LOCALIZE = """Analyze the public correctness symptom using only the supplied source.
Return the responsible symbol and line, the concrete source mechanism, and a
minimal proposed change. Do not write a patch yet. Do not request another file."""

_PATCH = """Write one exact SEARCH/REPLACE implementing the supplied causal hypothesis.
Copy old_code exactly from the allowed source file. Keep the edit minimal and
preserve unrelated behavior. Return empty old_code/new_code when the evidence
does not support a safe correction. Do not edit or request another file."""

_MONOLITH = """Locate the responsible mechanism and write one exact SEARCH/REPLACE.
Copy old_code exactly from the allowed source file. Keep the edit minimal and
preserve unrelated behavior. Return empty old_code/new_code when the evidence
does not support a safe correction. Do not edit or request another file."""

_CRITIC = """Select the strongest causal hypothesis from the numbered proposals.
Prefer the proposal that explains the public symptom through an exact lifecycle,
parameter, state, or data contract visible in the supplied source. Return its
zero-based index. Do not invent a new hypothesis or patch."""


def _visible_case(case: MicroCase, source: Path) -> tuple[str, int]:
    """Build the complete model-visible payload without private case fields."""

    target = source / case.prompt.file_path
    if not target.is_file():
        raise FileNotFoundError(f"allowed source file is missing: {case.prompt.file_path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    card = build_fedot_context(source, case.prompt.file_path)
    library_context = card.render(max_chars=6_000) if card is not None else ""
    return (
        model_facing_context(case)
        + ("\n\nFEDOT architecture card:\n" + library_context if library_context else "")
        + "\n\nComplete allowed source file:\n"
        + f"--- {case.prompt.file_path} ---\n"
        + text,
        len(text.splitlines()),
    )


def _usage_requests(inference: Any) -> int:
    usage = getattr(inference, "usage", None)
    if not isinstance(usage, dict):
        return 0
    value = usage.get("requests", 0)
    return int(value) if isinstance(value, (int, float)) else 0


class _Calls:
    def __init__(self, inference: Any):
        self.inference = inference
        self.attempted: Counter[str] = Counter()
        self.successful: Counter[str] = Counter()

    def create(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        stage: str,
        metadata: dict[str, Any],
    ) -> BaseModel:
        self.attempted[stage] += 1
        parsed, _ = capture_structured_create(
            self.inference,
            prompt,
            schema,
            stage=f"micro_{stage}",
            metadata=metadata,
        )
        self.successful[stage] += 1
        return parsed


def _localize(
    calls: _Calls,
    visible: str,
    *,
    case_id: str,
    member: int | None = None,
) -> LocalizationProposal:
    suffix = (
        "\n\nWork independently from other committee members."
        if member is not None
        else ""
    )
    return calls.create(
        f"{_LOCALIZE}{suffix}\n\n{visible}",
        LocalizationProposal,
        stage="localization",
        metadata={"case_id": case_id, "committee_member": member},
    )


def _select_committee_hypothesis(
    calls: _Calls,
    visible: str,
    proposals: list[LocalizationProposal],
    *,
    case_id: str,
) -> LocalizationProposal:
    candidates = [proposal.model_dump() for proposal in proposals]
    selection = calls.create(
        f"{_CRITIC}\n\n{visible}\n\nNumbered proposals:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2),
        CriticSelection,
        stage="critic",
        metadata={"case_id": case_id, "candidate_count": len(candidates)},
    )
    if selection.selected_index >= len(proposals):
        raise ValueError(
            "critic selected an out-of-range hypothesis: "
            f"{selection.selected_index} >= {len(proposals)}"
        )
    return proposals[selection.selected_index]


def _patch_from_hypothesis(
    calls: _Calls,
    visible: str,
    hypothesis: LocalizationProposal,
    *,
    case_id: str,
) -> PatchProposal:
    return calls.create(
        f"{_PATCH}\n\n{visible}\n\nSelected causal hypothesis:\n"
        + hypothesis.model_dump_json(indent=2),
        PatchProposal,
        stage="patch",
        metadata={"case_id": case_id},
    )


def _propose(
    architecture: Architecture,
    calls: _Calls,
    visible: str,
    *,
    case_id: str,
    committee_size: int,
) -> tuple[LocalizationProposal, PatchProposal]:
    if architecture == "monolith":
        proposal = calls.create(
            f"{_MONOLITH}\n\n{visible}",
            MonolithProposal,
            stage="monolith",
            metadata={"case_id": case_id},
        )
        localization = LocalizationProposal.model_validate(proposal.model_dump())
        patch = PatchProposal.model_validate(proposal.model_dump())
        return localization, patch

    if architecture == "staged":
        localization = _localize(calls, visible, case_id=case_id)
        return localization, _patch_from_hypothesis(
            calls, visible, localization, case_id=case_id
        )

    proposals = [
        _localize(calls, visible, case_id=case_id, member=index)
        for index in range(committee_size)
    ]
    localization = _select_committee_hypothesis(
        calls, visible, proposals, case_id=case_id
    )
    return localization, _patch_from_hypothesis(
        calls, visible, localization, case_id=case_id
    )


def _selected_cases(case_ids: Iterable[str] | None) -> tuple[MicroCase, ...]:
    cases = micro_cases()
    if case_ids is None:
        return cases
    requested = tuple(dict.fromkeys(str(value).strip() for value in case_ids if str(value).strip()))
    by_id = {case.prompt.case_id: case for case in cases}
    unknown = [case_id for case_id in requested if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown micro case ids: {', '.join(unknown)}")
    return tuple(by_id[case_id] for case_id in requested)


def _localization_matches(case: MicroCase, proposal: LocalizationProposal) -> bool:
    """Judge a symbol privately without putting the expected name in feedback."""

    proposed = proposal.symbol.strip().lower()
    return any(expected.lower() in proposed for expected in case.oracle.symbols)


def _stage_counts(case_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in case_rows:
        for stage in row["stages"]:
            counts.setdefault(stage["stage"], Counter())[stage["status"]] += 1
    return {stage: dict(values) for stage, values in sorted(counts.items())}


def run_micro_agent_benchmark(
    source: Path,
    workspace: Path,
    *,
    inference: Any,
    architecture: Architecture,
    case_ids: Iterable[str] | None = None,
    committee_size: int = 3,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run one LLM architecture on isolated micro cases and write its artifact.

    This function performs paid calls only through the explicitly supplied
    ``inference`` object.  Importing the module and constructing schemas are
    always offline.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown micro-agent architecture: {architecture}")
    if architecture == "committee" and committee_size < 2:
        raise ValueError("committee_size must be at least 2")
    if inference is None:
        raise ValueError("micro-agent benchmark requires an explicit inference client")

    source = source.resolve()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    output = artifact_path or (workspace / f"micro_agent_{architecture}.json")
    source_hash = source_fingerprint(source)
    requests_before = _usage_requests(inference)
    calls = _Calls(inference)
    rows: list[dict[str, Any]] = []
    run_id = f"micro-agent-{architecture}-{uuid.uuid4().hex[:8]}"

    for case in _selected_cases(case_ids):
        case_id = case.prompt.case_id
        stages: list[dict[str, Any]] = []
        stock = validate_observation(
            case,
            run_fedot_snippet(source, case.behavior_probe),
            patched=False,
        )
        stages.append(
            {
                "stage": "stock_probe",
                "status": stock.status,
                "reason": stock.reason,
            }
        )
        if not stock.passed:
            rows.append({"case_id": case_id, "ok": False, "stages": stages})
            continue

        experiment = create_experiment_checkout(
            source,
            workspace,
            run_id=run_id,
            candidate_id=case_id,
        )
        try:
            visible, file_lines = _visible_case(case, experiment)
            try:
                localization, proposed = _propose(
                    architecture,
                    calls,
                    visible,
                    case_id=case_id,
                    committee_size=committee_size,
                )
            except Exception as exc:
                stages.append(
                    {
                        "stage": "llm_architecture",
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                )
                rows.append({"case_id": case_id, "ok": False, "stages": stages})
                continue

            localization_ok = bool(
                localization.symbol.strip()
                and localization.mechanism.strip()
                and 1 <= localization.line <= max(1, file_lines)
                and _localization_matches(case, localization)
            )
            stages.append(
                {
                    "stage": "localization",
                    "status": "passed" if localization_ok else "failed",
                    "reason": (
                        ""
                        if localization_ok
                        else "localization did not match the private source target"
                    ),
                    "symbol": localization.symbol[:300],
                    "line": localization.line,
                }
            )
            candidate = PatchCandidate(
                candidate_id=f"micro-{case_id}-{uuid.uuid4().hex[:8]}",
                edits=[
                    PatchEdit(
                        case.prompt.file_path,
                        proposed.old_code,
                        proposed.new_code,
                    )
                ],
                rationale=proposed.rationale,
            )
            diagnostics: list[str] = []
            try:
                applied = apply_patch(experiment, candidate, diagnostics=diagnostics)
            except (OSError, PermissionError) as exc:
                applied = False
                diagnostics.append(f"{type(exc).__name__}: {exc}")
            stages.append(
                {
                    "stage": "patch",
                    "status": "passed" if applied else "failed",
                    "reason": "; ".join(diagnostics)[:1000],
                }
            )
            if not applied:
                rows.append({"case_id": case_id, "ok": False, "stages": stages})
                continue

            patched = validate_observation(
                case,
                run_fedot_snippet(experiment, case.behavior_probe),
                patched=True,
            )
            stages.append(
                {
                    "stage": "behavior_probe",
                    "status": patched.status,
                    "reason": patched.reason,
                    "observation": patched.observation,
                    "duration_s": patched.duration_s,
                }
            )
            rows.append(
                {
                    "case_id": case_id,
                    "ok": all(stage["status"] == "passed" for stage in stages),
                    "stages": stages,
                }
            )
        finally:
            discard_experiment_checkout(
                experiment,
                workspace=workspace,
                source=source,
            )

    source_hash_after = source_fingerprint(source)
    immutable_source = source_hash_after == source_hash
    attempted = sum(calls.attempted.values())
    successful = sum(calls.successful.values())
    payload = {
        "schema_version": 1,
        "component": "micro-agent",
        "architecture": architecture,
        "ok": bool(rows) and all(row["ok"] for row in rows) and immutable_source,
        "source_hash_before": source_hash,
        "source_hash_after": source_hash_after,
        "immutable_source": immutable_source,
        "counts": {
            "cases": len(rows),
            "passed_cases": sum(bool(row["ok"]) for row in rows),
            "stage_results": _stage_counts(rows),
            "structured_calls": {
                "attempted": attempted,
                "successful": successful,
                "by_stage": {
                    stage: {
                        "attempted": calls.attempted[stage],
                        "successful": calls.successful[stage],
                    }
                    for stage in sorted(calls.attempted)
                },
            },
            "provider_requests": max(
                0, _usage_requests(inference) - requests_before
            ),
        },
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
