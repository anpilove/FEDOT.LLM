"""Fixer: one LLM patch on a scout lead. Writes only inside the FEDOT checkout."""

from __future__ import annotations

from pathlib import Path

from research.evolve.metric_agent.context import context_from_lead
from research.evolve.metric_agent.journal import write_artifact
from research.evolve.metric_agent.patch import apply_patch
from research.evolve.metric_agent.propose import propose_patch
from research.evolve.metric_agent.types import Lead, PatchCandidate


def fix_lead(
    checkout: Path,
    lead: Lead,
    *,
    inference,
    workspace: Path | None = None,
) -> PatchCandidate | None:
    ctx = context_from_lead(lead, checkout)
    if workspace is not None:
        write_artifact(workspace / "context", "llm_context.txt", ctx)
    candidate = propose_patch(inference=inference, context=ctx)
    if candidate is None:
        return None
    if workspace is not None:
        folder = workspace / "candidates" / candidate.candidate_id
        write_artifact(folder, "rationale.txt", candidate.rationale)
        write_artifact(folder, "old.py", candidate.old_code)
        write_artifact(folder, "new.py", candidate.new_code)
    try:
        applied = apply_patch(checkout, candidate)
    except PermissionError:
        return None
    if not applied:
        return None
    return candidate
