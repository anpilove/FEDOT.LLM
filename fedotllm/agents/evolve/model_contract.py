"""Model identity contract for reproducible EvolveAgent campaigns.

The contract is intentionally independent from the campaign controller.  It can
be evaluated before any model request and its JSON-serializable result can be
written directly to a campaign artifact.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


REQUIRED_MODEL_NAME = "z-ai/glm-5.3-flash"
KNOWN_MODEL_STAGES = ("scout", "verifier", "fixer")
MODEL_CONTRACT_SCHEMA_VERSION = 1


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _fallbacks(value: Any) -> list[str]:
    """Return normalized fallback model names without assuming config types."""

    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw = value
    else:
        raw = ()
    return list(dict.fromkeys(item for value in raw if (item := _text(value))))


def build_model_contract(
    clients: Mapping[str, Any],
    *,
    required_model_name: str = REQUIRED_MODEL_NAME,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect Evolve inference clients without issuing an inference request.

    Objects without a ``config`` attribute are test doubles and do not
    participate in validation.  A supplied client with a config is considered a
    real inference client: every known stage must use the required model and
    must have neither configured nor environment-provided fallbacks.
    """

    environment = os.environ if environ is None else environ
    required = _text(required_model_name)
    env_fallbacks = _fallbacks(environment.get("FEDOTLLM_LLM_FALLBACK", ""))
    diagnostics: list[str] = []
    stages: dict[str, dict[str, Any]] = {}

    for stage in KNOWN_MODEL_STAGES:
        client = clients.get(stage)
        if client is None:
            stages[stage] = {"status": "not_supplied"}
            continue
        if not hasattr(client, "config"):
            stages[stage] = {"status": "ignored_test_double"}
            continue

        config = getattr(client, "config")
        model_name = _text(getattr(config, "model_name", ""))
        provider = _text(getattr(config, "provider", ""))
        configured_fallbacks = _fallbacks(getattr(config, "fallback_models", ""))
        effective_fallbacks = list(
            dict.fromkeys(env_fallbacks or configured_fallbacks)
        )
        row = {
            "status": "checked",
            "provider": provider,
            "model_name": model_name,
            "qualified_model": "/".join(
                part for part in (provider, model_name) if part
            ),
            "configured_fallback_models": configured_fallbacks,
            "environment_fallback_models": env_fallbacks,
            "effective_fallback_models": effective_fallbacks,
        }
        stages[stage] = row

        if model_name != required:
            diagnostics.append(
                f"{stage}: model_name must be {required!r}, got {model_name!r}"
            )
        if configured_fallbacks:
            diagnostics.append(
                f"{stage}: fallback_models must be empty, got "
                + ", ".join(configured_fallbacks)
            )
        if env_fallbacks:
            diagnostics.append(
                f"{stage}: FEDOTLLM_LLM_FALLBACK must be empty, got "
                + ", ".join(env_fallbacks)
            )

    checked_stages = [
        stage for stage, row in stages.items() if row["status"] == "checked"
    ]
    ignored_stages = [
        stage
        for stage, row in stages.items()
        if row["status"] in {"not_supplied", "ignored_test_double"}
    ]
    return {
        "schema_version": MODEL_CONTRACT_SCHEMA_VERSION,
        "required_model_name": required,
        "known_stages": list(KNOWN_MODEL_STAGES),
        "ok": not diagnostics,
        "diagnostics": diagnostics,
        "checked_stages": checked_stages,
        "ignored_stages": ignored_stages,
        "stages": stages,
    }


