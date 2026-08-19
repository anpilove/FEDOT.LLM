from typing import Any

from fedotllm.agents.base import FedotLLMAgentState


class EvolveAgentState(FedotLLMAgentState, total=False):
    """State for FEDOT framework evolution / optional code QA."""

    repo_path: str
    repo_context: str
    evolve_mode: str  # "evolve" | "qa"
    scout_pick: str
    scout_why: str
    proposal: dict[str, Any]
    evolve_success: bool
    # Severity class of the accepted patch (1 = behavioural defect … 4 = cosmetic).
    # Reported separately from success: success alone says nothing about value.
    evolve_severity: int
    probe_resolved: list[str]
    # The run deliberately produced nothing: no evidence of a defect worth fixing.
    # Reported separately from failure — an abstention is a correct outcome.
    evolve_abstained: bool
    evolve_abstain_reason: str
    evolve_evidence: str
    audit_path: str
    audit_markdown: str
    workspace: str
