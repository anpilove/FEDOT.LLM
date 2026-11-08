"""Compatibility of a loaded ``Fedot`` with the worker's return_best kwargs.

``return_best`` passes budget knobs the stock 0.7.5 constructor does not
name. A ``**kwargs`` catch-all is not compatibility: those keys become
composer parameters and FEDOT raises ``KeyError`` after the fit has started.
The worker rejects that pin before ``Fedot()``. ``legacy`` keeps today's
call shape and does not pretend the pin implements return_best.
"""

from __future__ import annotations

import inspect
import re

from fedotllm.agents.evolve.commands.preflight import WORKER_PARAMETERS

# GOLEM evaluate_single returns None once its timer is reached. A later
# holdout number is not a fitness. These phrases are the FEDOT budget report,
# not the separate "timeout is too small for composing" skip.
_BUDGET_TIMEOUT_TEXT = re.compile(
    r"fedotbudgetexceeded|budget exhausted|budget exceeded",
    re.IGNORECASE,
)

_EXPLICIT = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


class ReturnBestIncompatible(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        self.report = {
            "mode": "legacy_reject",
            "compatible": False,
            "missing": self.missing,
            "action": "invalid_before_fit",
        }
        names = ", ".join(self.missing)
        super().__init__(f"Fedot.__init__ missing return_best parameters: {names}")


def budget_timeout_reported(outcome, log_text: str = "") -> bool:
    """True when FEDOT reported that its fit budget already stopped the search."""

    if isinstance(outcome, str) and _BUDGET_TIMEOUT_TEXT.search(outcome):
        return True
    return bool(_BUDGET_TIMEOUT_TEXT.search(log_text or ""))


def signature_compatibility(fedot_cls: type) -> dict:
    params = inspect.signature(fedot_cls.__init__).parameters
    explicit = {name for name, param in params.items() if param.kind in _EXPLICIT}
    missing = [name for name in WORKER_PARAMETERS if name not in explicit]
    if missing:
        return {
            "mode": "legacy_reject",
            "compatible": False,
            "missing": missing,
            "action": "invalid_before_fit",
        }
    return {
        "mode": "return_best",
        "compatible": True,
        "missing": [],
        "action": "pass_worker_kwargs",
    }


def worker_action(policy: str, fedot_cls: type) -> dict:
    if policy not in {"legacy", "return_best"}:
        raise ValueError(f"unsupported FEDOT timeout policy: {policy}")
    report = signature_compatibility(fedot_cls)
    if policy == "legacy":
        return {**report, "mode": "legacy", "action": "legacy_kwargs_unchanged"}
    if not report["compatible"]:
        return report
    return report
