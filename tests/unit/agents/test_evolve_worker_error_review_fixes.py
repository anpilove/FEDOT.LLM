"""Classifying a worker error must not import a new ML runtime or change logging."""

import builtins
import sys
from types import ModuleType

from fedotllm.agents.evolve.evaluation import _fedot_quality_worker as worker


def test_error_classification_does_not_import_fedot(monkeypatch):
    monkeypatch.delitem(sys.modules, "fedot.api.api_utils.timeout", raising=False)
    original_import = builtins.__import__

    def forbid_fedot_import(name, *args, **kwargs):
        if name == "fedot" or name.startswith("fedot."):
            raise AssertionError("classifying an unrelated error must not initialize FEDOT")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_fedot_import)
    assert worker._is_fedot_budget_timeout(ValueError("invalid request")) is False


def test_error_classification_recognizes_only_the_loaded_runtime_type(monkeypatch):
    class FedotBudgetExceeded(TimeoutError):
        pass

    module = ModuleType("fedot.api.api_utils.timeout")
    module.FedotBudgetExceeded = FedotBudgetExceeded
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert worker._is_fedot_budget_timeout(FedotBudgetExceeded("budget exhausted")) is True
    assert worker._is_fedot_budget_timeout(TimeoutError("unrelated timeout")) is False
