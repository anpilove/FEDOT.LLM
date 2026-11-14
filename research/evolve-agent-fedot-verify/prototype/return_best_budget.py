"""Research prototype: return_best AutoML search budget for FEDOT.

Hypothesis
----------
EvolveAgent quality workers call ``Fedot(..., timeout_policy='return_best',
final_fit_reserve_seconds=..., max_pipeline_evaluation_time=...)``. Stock FEDOT
at pin ``8990f28b`` rejects those keys (``KeyError``) and has no shared
monotonic deadline that reserves time for a final incumbent refit. Search can
therefore spend the whole side budget on composition/tuning and leave the
worker without a scorable model.

This module is a **standalone** sketch of the missing budget arithmetic. It is
not wired into FEDOT.LLM production paths and does not claim a quality win.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_POLICIES = frozenset({"legacy", "return_best"})


@dataclass(frozen=True)
class ReturnBestBudget:
    """Per-side search budget with an explicit final-fit reserve.

    All values are seconds. ``search_deadline`` is the monotonic wall budget for
    initial incumbent + composition + tuning. ``final_fit_reserve`` is held out
    of that search window and may only be used to refit/promote an already
    selected incumbent before scoring.
    """

    side_timeout_seconds: float
    fedot_search_budget_seconds: float
    final_fit_reserve_seconds: float
    cv_folds: int
    composing_tuning_proportion: float = 0.6

    def __post_init__(self) -> None:
        if self.side_timeout_seconds <= 0:
            raise ValueError("side_timeout_seconds must be positive")
        if not 0 < self.fedot_search_budget_seconds < self.side_timeout_seconds:
            raise ValueError(
                "fedot_search_budget_seconds must be positive and below the side deadline"
            )
        if self.final_fit_reserve_seconds < 0:
            raise ValueError("final_fit_reserve_seconds must be non-negative")
        if self.final_fit_reserve_seconds >= self.fedot_search_budget_seconds:
            raise ValueError("final_fit_reserve_seconds must be below the FEDOT search budget")
        if self.cv_folds <= 0:
            raise ValueError("cv_folds must be positive")
        if not 0 < self.composing_tuning_proportion < 1:
            raise ValueError("composing_tuning_proportion must be in (0, 1)")

    @property
    def search_deadline_seconds(self) -> float:
        """Budget available to composition/tuning after reserving final fit."""

        return self.fedot_search_budget_seconds - self.final_fit_reserve_seconds

    @property
    def composing_budget_seconds(self) -> float:
        return self.search_deadline_seconds * self.composing_tuning_proportion

    @property
    def tuning_budget_seconds(self) -> float:
        return self.search_deadline_seconds - self.composing_budget_seconds

    @property
    def max_pipeline_evaluation_time_seconds(self) -> int:
        """Total per-candidate evaluation allowance under return_best."""

        return int(self.fedot_search_budget_seconds) // self.cv_folds

    @property
    def max_pipeline_fit_time_seconds(self) -> int:
        """Per-fold fit cap derived from the candidate evaluation allowance."""

        return self.max_pipeline_evaluation_time_seconds // self.cv_folds

    def remaining_search_seconds(self, elapsed_search_seconds: float) -> float:
        if elapsed_search_seconds < 0:
            raise ValueError("elapsed_search_seconds must be non-negative")
        return max(0.0, self.search_deadline_seconds - elapsed_search_seconds)

    def may_start_candidate(self, elapsed_search_seconds: float) -> bool:
        """Skip a new candidate when the remaining search window is exhausted."""

        return self.remaining_search_seconds(elapsed_search_seconds) > 0

    def may_start_final_fit(self, elapsed_search_seconds: float) -> bool:
        """Final fit uses only the reserved window, never leftover search time alone."""

        _ = elapsed_search_seconds  # search overrun must not steal the reserve contract
        return self.final_fit_reserve_seconds > 0


def validate_worker_timeout_kwargs(params: dict) -> dict:
    """Mirror the keys EvolveAgent's quality worker passes into ``Fedot``."""

    policy = params.get("timeout_policy", "legacy")
    if policy not in SUPPORTED_POLICIES:
        raise KeyError(f"unsupported timeout_policy={policy!r}")
    allowed = {
        "timeout_policy",
        "final_fit_reserve_seconds",
        "max_pipeline_evaluation_time",
    }
    unknown = set(params) - allowed
    if unknown:
        raise KeyError(f"Invalid key parameters {unknown}")
    if policy == "return_best":
        for key in ("final_fit_reserve_seconds", "max_pipeline_evaluation_time"):
            if key not in params:
                raise KeyError(f"return_best requires {key}")
    return dict(params)
