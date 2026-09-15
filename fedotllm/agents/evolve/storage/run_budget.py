"""Per-run limits and durable accounting for EvolveAgent provider calls.

The budget is intentionally shared by all inference clients participating in one
campaign and has no durable ledger::

    with EvolveRunBudget(
        (scout, verifier, fixer), max_queries=40, max_cost_usd=0.25
    ) as budget:
        decision = run_once(...)
    print(budget.snapshot())

``AIInference.query`` may retry transport failures or walk a fallback model
chain.  Wrapping ``_complete`` counts the actual provider attempts rather than
the higher-level structured actions.  ``EvolveBudgetExhausted`` inherits from
``LLMRequestTimeout`` so the existing AIInference retry decorators do not issue
another provider request after a limit is reached.
"""

from __future__ import annotations

import math
import time
import uuid
import threading
from contextlib import AbstractContextManager
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from fedotllm.llm import LLMRequestTimeout
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.llm_audit import (
    _audit_path,
    current_query_context,
)


class EvolveBudgetExhausted(LLMRequestTimeout):
    """A per-run EvolveAgent limit rejected a new provider request."""


@dataclass(frozen=True)
class RunBudgetSnapshot:
    """Small JSON-compatible summary of one run's LLM consumption."""

    queries_started: int
    queries_succeeded: int
    queries_failed: int
    retries: int
    local_rejections: int
    actual_cost_usd: float
    max_queries: int | None
    max_cost_usd: float | None
    queries_by_stage: dict[str, int] = field(default_factory=dict)
    reserved_queries: dict[str, int] = field(default_factory=dict)

    @property
    def query_limit_reached(self) -> bool:
        return self.max_queries is not None and self.queries_started >= self.max_queries

    @property
    def cost_limit_reached(self) -> bool:
        return self.max_cost_usd is not None and self.actual_cost_usd >= self.max_cost_usd

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            **asdict(self),
            "query_limit_reached": self.query_limit_reached,
            "cost_limit_reached": self.cost_limit_reached,
        }


@dataclass
class _PatchedClient:
    client: Any
    had_instance_complete: bool
    previous_instance_complete: Any
    baseline_cost_usd: float


def _usage_cost(client: Any) -> float:
    usage = getattr(client, "usage", None)
    if not isinstance(usage, dict):
        return 0.0
    value = usage.get("cost_usd", 0.0)
    if not isinstance(value, (int, float)):
        return 0.0
    cost = float(value)
    return cost if math.isfinite(cost) else 0.0


class EvolveRunBudget(AbstractContextManager["EvolveRunBudget"]):
    """Share hard query and actual-cost limits across one campaign's clients.

    The cost check happens immediately before every real ``_complete`` call.
    Therefore a request which starts below the cost ceiling may finish slightly
    above it, but the next request is rejected.  Query count is reserved before
    calling the provider, so failed provider attempts count toward the hard cap.
    """

    def __init__(
        self,
        clients: Iterable[Any] | Mapping[str, Any],
        *,
        max_queries: int | None,
        max_cost_usd: float | None,
        reserved_queries: Mapping[str, int] | None = None,
    ) -> None:
        if max_queries is not None and max_queries < 0:
            raise ValueError("max_queries must be non-negative or None")
        if max_cost_usd is not None and (
            not math.isfinite(max_cost_usd) or max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be finite, non-negative, or None")
        if isinstance(clients, Mapping):
            client_rows = list(clients.items())
        else:
            client_rows = [("shared", client) for client in clients]
        unique: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for raw_stage, client in client_rows:
            if client is None or id(client) in seen:
                continue
            if not callable(getattr(client, "_complete", None)):
                raise TypeError("each budget client must provide a callable _complete")
            seen.add(id(client))
            stage = str(raw_stage).strip() or "shared"
            unique.append((stage, client))
        if not unique:
            raise ValueError("at least one inference client is required")
        reserves = {
            str(stage): int(value)
            for stage, value in (reserved_queries or {}).items()
            if int(value) > 0
        }
        if any(value < 0 for value in (reserved_queries or {}).values()):
            raise ValueError("reserved query counts must be non-negative")
        if max_queries is not None and sum(reserves.values()) > max_queries:
            raise ValueError("reserved query counts exceed max_queries")
        known_stages = {stage for stage, _ in unique}
        unknown = sorted(set(reserves) - known_stages)
        if unknown:
            raise ValueError(f"reserved query stages have no client: {', '.join(unknown)}")
        self._clients = tuple(unique)
        self.max_queries = max_queries
        self.max_cost_usd = float(max_cost_usd) if max_cost_usd is not None else None
        self.reserved_queries = reserves
        self.queries_started = 0
        self.queries_succeeded = 0
        self.queries_failed = 0
        self.retries = 0
        self.local_rejections = 0
        self.queries_by_stage: Counter[str] = Counter()
        self._logical_attempts: Counter[tuple[str, int]] = Counter()
        self._patched: list[_PatchedClient] = []
        self._lock = threading.RLock()
        self._entered = False
        self._final_cost_usd: float | None = None

    @property
    def actual_cost_usd(self) -> float:
        with self._lock:
            if self._final_cost_usd is not None:
                return self._final_cost_usd
            return sum(
                max(0.0, _usage_cost(item.client) - item.baseline_cost_usd)
                for item in self._patched
            )

    def snapshot(self) -> RunBudgetSnapshot:
        return RunBudgetSnapshot(
            queries_started=self.queries_started,
            queries_succeeded=self.queries_succeeded,
            queries_failed=self.queries_failed,
            retries=self.retries,
            local_rejections=self.local_rejections,
            actual_cost_usd=self.actual_cost_usd,
            max_queries=self.max_queries,
            max_cost_usd=self.max_cost_usd,
            queries_by_stage=dict(self.queries_by_stage),
            reserved_queries=dict(self.reserved_queries),
        )

    def _reserve_query(self, stage: str) -> None:
        with self._lock:
            current_cost = self.actual_cost_usd
            if self.max_queries is not None and self.queries_started >= self.max_queries:
                raise EvolveBudgetExhausted(
                    "evolve_run_query_budget_exhausted: "
                    f"started={self.queries_started}, limit={self.max_queries}; "
                    "no provider request sent"
                )
            if self.max_queries is not None:
                remaining = self.max_queries - self.queries_started
                protected_for_other_stages = sum(
                    max(0, minimum - self.queries_by_stage.get(other, 0))
                    for other, minimum in self.reserved_queries.items()
                    if other != stage
                )
                if remaining <= protected_for_other_stages:
                    raise EvolveBudgetExhausted(
                        "evolve_run_stage_reserve_exhausted: "
                        f"stage={stage}, remaining={remaining}, "
                        f"protected_for_other_stages={protected_for_other_stages}; "
                        "no provider request sent"
                    )
            if self.max_cost_usd is not None and current_cost >= self.max_cost_usd:
                raise EvolveBudgetExhausted(
                    "evolve_run_cost_budget_exhausted: "
                    f"actual_usd={current_cost:.8f}, limit_usd={self.max_cost_usd:.8f}; "
                    "no provider request sent"
                )
            self.queries_started += 1
            self.queries_by_stage[stage] += 1

    def __enter__(self) -> "EvolveRunBudget":
        if self._entered:
            raise RuntimeError("an EvolveRunBudget instance can be entered only once")
        self._entered = True
        try:
            for stage, client in self._clients:
                namespace = getattr(client, "__dict__", {})
                had_instance_complete = "_complete" in namespace
                previous_instance_complete = namespace.get("_complete")
                original_complete = getattr(client, "_complete")
                item = _PatchedClient(
                    client=client,
                    had_instance_complete=had_instance_complete,
                    previous_instance_complete=previous_instance_complete,
                    baseline_cost_usd=_usage_cost(client),
                )
                self._patched.append(item)

                def budgeted_complete(
                    messages,
                    _original=original_complete,
                    _stage=stage,
                    _client=client,
                ):
                    logical = current_query_context()
                    logical_key = (
                        str(logical.get("call_group_id")),
                        int(logical.get("query_number") or 0),
                    ) if logical.get("call_group_id") else (uuid.uuid4().hex, 0)
                    path = _audit_path()
                    try:
                        self._reserve_query(_stage)
                    except EvolveBudgetExhausted as exc:
                        with self._lock:
                            self.local_rejections += 1
                        if path is not None:
                            append_journal(
                                path,
                                {
                                    "event": "provider_local_rejection",
                                    "stage": _stage,
                                    **logical,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                            )
                        raise
                    with self._lock:
                        self._logical_attempts[logical_key] += 1
                        attempt_number = self._logical_attempts[logical_key]
                        if attempt_number > 1:
                            self.retries += 1
                    attempt_id = uuid.uuid4().hex
                    started = time.monotonic()
                    before_cost = _usage_cost(_client)
                    common = {
                        "attempt_id": attempt_id,
                        "stage": _stage,
                        **logical,
                        "attempt_number": attempt_number,
                        "is_retry": attempt_number > 1,
                    }
                    if path is not None:
                        append_journal(
                            path, {"event": "provider_attempt_started", **common}
                        )
                    try:
                        response = _original(messages)
                    except BaseException as exc:
                        with self._lock:
                            self.queries_failed += 1
                        if path is not None:
                            append_journal(
                                path,
                                {
                                    "event": "provider_attempt_result",
                                    **common,
                                    "status": "error",
                                    "duration_seconds": time.monotonic() - started,
                                    "cost_delta_usd": max(
                                        0.0, _usage_cost(_client) - before_cost
                                    ),
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:2_000],
                                },
                            )
                        raise
                    with self._lock:
                        self.queries_succeeded += 1
                    if path is not None:
                        append_journal(
                            path,
                            {
                                "event": "provider_attempt_result",
                                **common,
                                "status": "ok",
                                "duration_seconds": time.monotonic() - started,
                                "cost_delta_usd": max(
                                    0.0, _usage_cost(_client) - before_cost
                                ),
                            },
                        )
                    return response

                client._complete = budgeted_complete
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for item in reversed(self._patched):
            if item.had_instance_complete:
                item.client._complete = item.previous_instance_complete
            else:
                try:
                    delattr(item.client, "_complete")
                except AttributeError:
                    pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Freeze the run-local delta before clients can be reused by another
        # budget.  A later campaign must not mutate this run's summary.
        self._final_cost_usd = self.actual_cost_usd
        self._restore()
        return None
