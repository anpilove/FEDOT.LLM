"""FEDOT.LLM Evolve Controller; runtime dependencies load only when used."""

from importlib import import_module

__all__ = ["eval_contract", "run_once", "drain_quality_queue", "measure_fedot_quality"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "campaign" if name in {"eval_contract", "run_once"} else "quality_executor"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
