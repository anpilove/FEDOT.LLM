from types import SimpleNamespace

import pytest

from fedotllm.agents.evolve.storage.run_budget import (
    EvolveBudgetExhausted,
    EvolveRunBudget,
)
from fedotllm.llm import AIInference


class FakeInference:
    def __init__(self, costs=()):
        self.usage = {"cost_usd": 0.0}
        self.costs = iter(costs)
        self.provider_calls = 0

    def _complete(self, messages):
        self.provider_calls += 1
        self.usage["cost_usd"] += next(self.costs, 0.0)
        return str(messages)


def test_query_cap_counts_real_provider_attempts_and_restores_client():
    client = FakeInference()
    original = client._complete

    with EvolveRunBudget((client,), max_queries=2, max_cost_usd=None) as budget:
        assert client._complete("one") == "one"
        assert client._complete("two") == "two"
        with pytest.raises(EvolveBudgetExhausted, match="query_budget_exhausted"):
            client._complete("blocked")

    assert client.provider_calls == 2
    assert budget.snapshot().queries_started == 2
    assert client._complete == original


def test_actual_cost_is_shared_and_one_started_query_may_overshoot():
    scout = FakeInference((0.04,))
    fixer = FakeInference((0.07,))

    with EvolveRunBudget(
        (scout, fixer), max_queries=10, max_cost_usd=0.10
    ) as budget:
        scout._complete("first")
        fixer._complete("overshoot")
        assert budget.actual_cost_usd == pytest.approx(0.11)
        with pytest.raises(EvolveBudgetExhausted, match="cost_budget_exhausted"):
            scout._complete("blocked")

    snapshot = budget.snapshot()
    assert snapshot.queries_started == 2
    assert snapshot.cost_limit_reached is True
    assert snapshot.to_dict()["actual_cost_usd"] == pytest.approx(0.11)


def test_budget_is_per_run_and_ignores_preexisting_usage():
    client = FakeInference((0.03, 0.03))
    client.usage["cost_usd"] = 12.0

    with EvolveRunBudget((client,), max_queries=1, max_cost_usd=0.05) as first:
        client._complete("run one")
    with EvolveRunBudget((client,), max_queries=1, max_cost_usd=0.05) as second:
        client._complete("run two")

    assert first.snapshot().actual_cost_usd == pytest.approx(0.03)
    assert second.snapshot().actual_cost_usd == pytest.approx(0.03)
    assert client.provider_calls == 2


def test_budget_exception_causes_no_aiinference_provider_retry(monkeypatch):
    monkeypatch.delenv("FEDOTLLM_LLM_FALLBACK", raising=False)
    client = AIInference.__new__(AIInference)
    client.config = SimpleNamespace(
        provider="openrouter",
        model_name="primary",
        fallback_models="fallback-a,fallback-b",
    )
    client.completion_params = {"model": "openrouter/primary"}
    client.usage = {"cost_usd": 0.0}
    provider_calls = []

    def provider(messages):
        provider_calls.append(messages)
        return "unexpected"

    client._complete = provider
    with EvolveRunBudget((client,), max_queries=0, max_cost_usd=None):
        with pytest.raises(EvolveBudgetExhausted):
            client.query("blocked")

    assert provider_calls == []


def test_failed_provider_attempt_still_consumes_query_allowance():
    client = FakeInference()

    def failing_provider(_messages):
        client.provider_calls += 1
        raise RuntimeError("provider failed")

    client._complete = failing_provider
    with EvolveRunBudget((client,), max_queries=1, max_cost_usd=None) as budget:
        with pytest.raises(RuntimeError, match="provider failed"):
            client._complete("started")
        with pytest.raises(EvolveBudgetExhausted, match="query_budget_exhausted"):
            client._complete("must not start")

    assert client.provider_calls == 1
    assert budget.snapshot().queries_started == 1
    assert budget.snapshot().queries_failed == 1


def test_same_client_is_not_double_counted_when_roles_share_it():
    shared = FakeInference((0.02,))
    with EvolveRunBudget(
        (shared, shared, shared), max_queries=2, max_cost_usd=0.05
    ) as budget:
        shared._complete("one shared provider")

    assert budget.snapshot().actual_cost_usd == pytest.approx(0.02)


def test_scout_cannot_consume_queries_reserved_for_fixer():
    scout = FakeInference()
    fixer = FakeInference()
    with EvolveRunBudget(
        {"scout": scout, "fixer": fixer},
        max_queries=5,
        max_cost_usd=None,
        reserved_queries={"fixer": 2},
    ) as budget:
        for index in range(3):
            scout._complete(f"scout-{index}")
        with pytest.raises(EvolveBudgetExhausted, match="stage_reserve_exhausted"):
            scout._complete("would steal repair capacity")
        fixer._complete("repair")
        fixer._complete("probe check")

    snapshot = budget.snapshot()
    assert snapshot.queries_started == 5
    assert snapshot.queries_by_stage == {"scout": 3, "fixer": 2}
    assert snapshot.reserved_queries == {"fixer": 2}


def test_unused_stage_reserve_is_available_to_the_reserved_stage():
    scout = FakeInference()
    fixer = FakeInference()
    with EvolveRunBudget(
        {"scout": scout, "fixer": fixer},
        max_queries=4,
        max_cost_usd=None,
        reserved_queries={"fixer": 2},
    ):
        fixer._complete("one")
        fixer._complete("two")
        fixer._complete("three")
        fixer._complete("four")


def test_budget_distinguishes_sent_success_failure_retry_and_local_rejection(
    tmp_path, monkeypatch
):
    import json

    audit = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("EVOLVE_AGENT_LLM_AUDIT", str(audit))
    client = FakeInference()
    original = client._complete
    calls = 0

    def flaky(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("temporary")
        return original(messages)

    client._complete = flaky
    with EvolveRunBudget((client,), max_queries=2, max_cost_usd=None) as budget:
        with pytest.raises(ConnectionError):
            client._complete("first")
        assert client._complete("retry") == "retry"
        with pytest.raises(EvolveBudgetExhausted):
            client._complete("blocked")

    snapshot = budget.snapshot()
    assert snapshot.queries_started == 2
    assert snapshot.queries_succeeded == 1
    assert snapshot.queries_failed == 1
    assert snapshot.local_rejections == 1
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert sum(row["event"] == "provider_attempt_started" for row in rows) == 2
    assert sum(row["event"] == "provider_attempt_result" for row in rows) == 2
    assert sum(row["event"] == "provider_local_rejection" for row in rows) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_queries": -1, "max_cost_usd": None},
        {"max_queries": None, "max_cost_usd": -0.01},
        {"max_queries": None, "max_cost_usd": float("inf")},
        {
            "max_queries": 1,
            "max_cost_usd": None,
            "reserved_queries": {"fixer": 2},
        },
    ],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EvolveRunBudget((FakeInference(),), **kwargs)


def test_campaign_reserves_correctness_and_repair_calls():
    from fedotllm.agents.evolve.__main__ import _campaign_reserves

    assert _campaign_reserves(
        40, 12, has_verifier=True, has_fixer=True
    ) == {"verifier": 3, "fixer": 12}
    assert _campaign_reserves(
        4, 12, has_verifier=True, has_fixer=True
    ) == {"verifier": 3}
