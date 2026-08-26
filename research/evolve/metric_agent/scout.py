"""Scout: walk FEDOT source. No tests, no exam, no gym."""

from __future__ import annotations

from pathlib import Path

from research.evolve.metric_agent.discover import discover_leads
from research.evolve.metric_agent.types import Lead


def scout(checkout: Path, *, inference=None, max_leads: int = 5) -> list[Lead]:
    return discover_leads(checkout, inference=inference, limit=max_leads)
