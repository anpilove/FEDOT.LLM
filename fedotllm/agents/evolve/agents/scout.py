"""Scout: walk FEDOT source. No tests, no exam, no gym."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.discovery.discover import discover_leads
from fedotllm.agents.evolve.types import PatchSite


def scout(
    checkout: Path,
    *,
    inference=None,
    max_leads: int = 5,
    max_picks: int | None = None,
    trace: dict | None = None,
    execution: list[dict] | None = None,
    trace_leads: list[PatchSite] | None = None,
    max_actions: int | None = None,
    max_runs_per_file: int = 2,
    excluded_files: set[str] | None = None,
    excluded_sites: set[tuple[str, int]] | None = None,
    prior_hypotheses: list[dict] | None = None,
    excluded_semantic_sites: set[str] | None = None,
    present_full_catalog: bool = False,
    on_pick: Callable[[list[PatchSite]], None] | None = None,
) -> list[PatchSite]:
    return discover_leads(
        checkout,
        inference=inference,
        limit=max_leads,
        max_picks=max_picks,
        trace=trace,
        execution=execution,
        trace_leads=trace_leads,
        max_actions=max_actions,
        max_runs_per_file=max_runs_per_file,
        excluded_files=excluded_files,
        excluded_sites=excluded_sites,
        prior_hypotheses=prior_hypotheses,
        excluded_semantic_sites=excluded_semantic_sites,
        present_full_catalog=present_full_catalog,
        on_pick=on_pick,
    )
