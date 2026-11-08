"""Static preflight for a FEDOT checkout used by EvolveAgent quality workers.

Fails closed when the pin cannot accept the worker contract
(``timeout_policy=return_best``, optional isolated optimizer, dask import site).
Does not invent metrics and does not mutate the checkout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAILURE_KIND_PIN_INCOMPATIBLE = "fedot_pin_incompatible"


class FedotPinIncompatibleError(RuntimeError):
    """Raised when the active FEDOT pin cannot satisfy the worker contract."""

    failure_kind = FAILURE_KIND_PIN_INCOMPATIBLE

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail or message


def is_fedot_pin_parameter_keyerror(exc: BaseException) -> bool:
    """True when stock FEDOT rejected EvolveAgent timeout-policy kwargs."""

    if not isinstance(exc, KeyError):
        return False
    text = str(exc)
    return any(
        token in text
        for token in (
            "timeout_policy",
            "final_fit_reserve_seconds",
            "max_pipeline_evaluation_time",
            "Invalid key parameters",
        )
    )

_PARAMS_REL = Path("fedot/api/api_utils/api_params_repository.py")
_ISOLATED_REL = Path("fedot/core/optimisers/isolated_gp_optimizer.py")
_DASK_SITE_REL = Path(
    "fedot/core/operations/evaluation/operation_implementations/"
    "data_operations/sklearn_transformations.py"
)


@dataclass(frozen=True)
class FedotPinPreflight:
    ok: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_kind: str | None = None
    summary: str = ""


def _file_text(root: Path, rel: Path) -> str | None:
    path = root / rel
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def check_fedot_pin_preflight(
    source: Path,
    *,
    require_return_best: bool = True,
    require_isolated_optimizer: bool = True,
    require_dask_import_site: bool = True,
) -> FedotPinPreflight:
    """Inspect a FEDOT source tree for EvolveAgent quality-worker requirements."""

    source = source.resolve()
    checks: dict[str, dict[str, Any]] = {
        "source": {"ok": source.is_dir(), "path": str(source)},
    }
    if not source.is_dir():
        return FedotPinPreflight(
            ok=False,
            checks=checks,
            failure_kind=FAILURE_KIND_PIN_INCOMPATIBLE,
            summary=f"FEDOT source is not a directory: {source}",
        )

    params_text = _file_text(source, _PARAMS_REL)
    has_timeout_policy = bool(params_text and "timeout_policy" in params_text)
    has_final_reserve = bool(params_text and "final_fit_reserve_seconds" in params_text)
    has_max_eval = bool(params_text and "max_pipeline_evaluation_time" in params_text)
    return_best_ok = has_timeout_policy and has_final_reserve and has_max_eval
    checks["timeout_policy_return_best"] = {
        "ok": (not require_return_best) or return_best_ok,
        "required": require_return_best,
        "has_timeout_policy": has_timeout_policy,
        "has_final_fit_reserve_seconds": has_final_reserve,
        "has_max_pipeline_evaluation_time": has_max_eval,
        "path": str(_PARAMS_REL),
    }

    isolated_path = source / _ISOLATED_REL
    isolated_ok = isolated_path.is_file()
    checks["isolated_gp_optimizer"] = {
        "ok": (not require_isolated_optimizer) or isolated_ok,
        "required": require_isolated_optimizer,
        "present": isolated_ok,
        "path": str(_ISOLATED_REL),
    }

    dask_text = _file_text(source, _DASK_SITE_REL)
    dask_site = bool(dask_text and "import dask" in dask_text)
    checks["dask_import_site"] = {
        "ok": (not require_dask_import_site) or dask_site,
        "required": require_dask_import_site,
        "present": dask_site,
        "path": str(_DASK_SITE_REL),
        "note": "pin imports dask; runtime must provide the dask package",
    }

    failed = [name for name, item in checks.items() if name != "source" and not item.get("ok")]
    if not checks["source"]["ok"]:
        failed = ["source", *failed]
    ok = not failed
    if ok:
        return FedotPinPreflight(ok=True, checks=checks, failure_kind=None, summary="fedot pin compatible")

    parts = []
    if "timeout_policy_return_best" in failed:
        parts.append("timeout_policy/final_fit_reserve/max_pipeline_evaluation_time")
    if "isolated_gp_optimizer" in failed:
        parts.append("isolated_gp_optimizer")
    if "dask_import_site" in failed:
        parts.append("dask import site")
    if "source" in failed:
        parts.append("source directory")
    summary = "FEDOT pin incompatible with EvolveAgent quality worker: " + ", ".join(parts)
    return FedotPinPreflight(
        ok=False,
        checks=checks,
        failure_kind=FAILURE_KIND_PIN_INCOMPATIBLE,
        summary=summary,
    )
