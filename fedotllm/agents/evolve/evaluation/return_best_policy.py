"""Explicit compatibility policy for FEDOT ``timeout_policy=return_best``.

Default is refuse-closed on an incompatible pin. A research-only worktree may
be selected via ``EVOLVE_FEDOT_RETURN_BEST_ROOT`` for offline verification; that
path never auto-merges into production FEDOT and must itself pass preflight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fedotllm.agents.evolve.evaluation.fedot_pin_preflight import (
    FAILURE_KIND_PIN_INCOMPATIBLE,
    FedotPinPreflight,
    check_fedot_pin_preflight,
)

POLICY_REFUSE = "refuse"
POLICY_RESEARCH_WORKTREE = "research_worktree"
SUPPORTED_POLICIES = frozenset({POLICY_REFUSE, POLICY_RESEARCH_WORKTREE})
ENV_POLICY = "EVOLVE_FEDOT_RETURN_BEST_POLICY"
ENV_WORKTREE = "EVOLVE_FEDOT_RETURN_BEST_ROOT"


@dataclass(frozen=True)
class ReturnBestPolicyDecision:
    policy: str
    source: Path
    preflight: FedotPinPreflight
    ok: bool
    detail: str


def active_return_best_policy(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get(ENV_POLICY, POLICY_REFUSE)).strip()
    if value not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported return_best policy: {value!r}")
    return value


def resolve_return_best_source(
    stock_source: Path,
    *,
    policy: str | None = None,
    research_worktree: Path | None = None,
) -> ReturnBestPolicyDecision:
    """Choose the FEDOT tree for return_best quality work under the active policy."""

    stock_source = stock_source.resolve()
    chosen_policy = active_return_best_policy(policy)
    stock_preflight = check_fedot_pin_preflight(stock_source)

    if stock_preflight.ok:
        return ReturnBestPolicyDecision(
            policy=chosen_policy,
            source=stock_source,
            preflight=stock_preflight,
            ok=True,
            detail="stock FEDOT pin already supports return_best",
        )

    if chosen_policy == POLICY_REFUSE:
        return ReturnBestPolicyDecision(
            policy=chosen_policy,
            source=stock_source,
            preflight=stock_preflight,
            ok=False,
            detail=(
                f"{stock_preflight.summary}; policy={POLICY_REFUSE} "
                f"(set {ENV_POLICY}={POLICY_RESEARCH_WORKTREE} and "
                f"{ENV_WORKTREE} only for offline research)"
            ),
        )

    worktree = research_worktree
    if worktree is None:
        env_path = os.environ.get(ENV_WORKTREE, "").strip()
        worktree = Path(env_path) if env_path else None
    if worktree is None:
        empty = FedotPinPreflight(
            ok=False,
            checks={"research_worktree": {"ok": False, "path": None}},
            failure_kind=FAILURE_KIND_PIN_INCOMPATIBLE,
            summary=f"{ENV_WORKTREE} is required for policy={POLICY_RESEARCH_WORKTREE}",
        )
        return ReturnBestPolicyDecision(
            policy=chosen_policy,
            source=stock_source,
            preflight=empty,
            ok=False,
            detail=empty.summary,
        )

    worktree = worktree.resolve()
    worktree_preflight = check_fedot_pin_preflight(worktree)
    if not worktree_preflight.ok:
        return ReturnBestPolicyDecision(
            policy=chosen_policy,
            source=worktree,
            preflight=worktree_preflight,
            ok=False,
            detail=(
                f"research worktree failed preflight: {worktree_preflight.summary}"
            ),
        )
    return ReturnBestPolicyDecision(
        policy=chosen_policy,
        source=worktree,
        preflight=worktree_preflight,
        ok=True,
        detail=f"using research return_best worktree at {worktree}",
    )
