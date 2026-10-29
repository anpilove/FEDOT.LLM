"""Bound costly paired benchmark measurements within one campaign.

The controller accounts for a stock/patched task pair, rather than provider
calls.  A ceiling is intentionally fail-closed: lack of capacity is reported
as inconclusive and can never become a metric KEEP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time


@dataclass
class MeasurementBudget:
    max_pairs: int | None = None
    reserve_final_pairs: int = 0
    # Wall-clock ceiling covers the whole controller campaign.  Pair counts
    # remain useful because a fast cached run must not silently spend an
    # unbounded amount of benchmark evidence.
    max_seconds: float | None = None
    reserve_final_seconds: float = 0.0
    pairs_started: int = 0
    pairs_completed: int = 0
    elapsed_seconds: float = 0.0
    by_stage: dict[str, int] = field(default_factory=dict)
    _campaign_started_at: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if self.max_pairs is not None and self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative or None")
        if self.reserve_final_pairs < 0:
            raise ValueError("reserve_final_pairs must be non-negative")
        if self.max_seconds is not None and self.max_seconds < 0:
            raise ValueError("max_seconds must be non-negative or None")
        if self.reserve_final_seconds < 0:
            raise ValueError("reserve_final_seconds must be non-negative")
        if self.max_seconds is not None and self.reserve_final_seconds > self.max_seconds:
            raise ValueError("reserve_final_seconds exceed max_seconds")
        if self.max_pairs is not None and self.reserve_final_pairs > self.max_pairs:
            raise ValueError("reserve_final_pairs exceed max_pairs")

    def wall_elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._campaign_started_at)

    def can_start(self, stage: str) -> bool:
        if self.max_pairs is not None:
            remaining = self.max_pairs - self.pairs_started
            reserve = 0 if stage == "final" else self.reserve_final_pairs
            if remaining <= reserve:
                return False
        if self.max_seconds is not None:
            remaining_seconds = self.max_seconds - self.wall_elapsed_seconds()
            reserve_seconds = 0.0 if stage == "final" else self.reserve_final_seconds
            if remaining_seconds <= reserve_seconds:
                return False
        return True

    def begin(self, stage: str) -> float | None:
        if not self.can_start(stage):
            return None
        self.pairs_started += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        return time.monotonic()

    def finish(self, started_at: float) -> None:
        self.pairs_completed += 1
        self.elapsed_seconds += max(0.0, time.monotonic() - started_at)

    def snapshot(self) -> dict:
        # asdict intentionally excludes the monotonic origin from public evidence:
        # it is process-local and cannot be replayed on another worker.
        payload = asdict(self)
        payload.pop("_campaign_started_at", None)
        return payload | {
            "wall_elapsed_seconds": self.wall_elapsed_seconds(),
            "remaining_pairs": None if self.max_pairs is None else self.max_pairs - self.pairs_started,
            "remaining_seconds": (
                None if self.max_seconds is None
                else max(0.0, self.max_seconds - self.wall_elapsed_seconds())
            ),
        }
