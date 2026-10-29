"""Campaign journals, findings, scoreboards, replay state, and run budgets."""

from importlib import import_module

_RUN_BUDGET_EXPORTS = {
    "EvolveBudgetExhausted",
    "EvolveRunBudget",
    "RunBudgetSnapshot",
}


def __getattr__(name: str):
    """Load provider-backed budget code only when a caller requests it.

    Quality-queue workers use the storage journal but do not call the LLM.
    Keeping these exports lazy lets the FEDOT-only runtime import journal and
    result storage without requiring the agent's LiteLLM dependency stack.
    """
    if name in _RUN_BUDGET_EXPORTS:
        module = import_module("fedotllm.agents.evolve.storage.run_budget")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EvolveBudgetExhausted",
    "EvolveRunBudget",
    "RunBudgetSnapshot",
]
