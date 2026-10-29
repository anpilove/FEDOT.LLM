"""Lossless local audit trail for EvolveAgent model calls.

The regular controller journal is intentionally compact.  This module records
the exact message passed to ``AIInference.query``, the provider text returned to
the agent, and the structured action produced from it.  It is observability,
not a security boundary: credentials are never intentionally put in prompts and
known API-key values are redacted defensively before writing the local file.
"""

from __future__ import annotations

import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fedotllm.agents.evolve.storage.journal import append_journal


_CURRENT_QUERY: ContextVar[dict[str, Any] | None] = ContextVar(
    "evolve_current_llm_query", default=None
)


def current_query_context() -> dict[str, Any]:
    """Identify one logical query while transport-level attempts are running."""

    return dict(_CURRENT_QUERY.get() or {})


def _audit_path() -> Path | None:
    raw = os.environ.get("EVOLVE_AGENT_LLM_AUDIT", "").strip()
    return Path(raw) if raw else None


def _redact(value: Any) -> Any:
    """Remove credentials without truncating prompts or model output."""

    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for name in (
        "FEDOTLLM_LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        secret = os.environ.get(name, "")
        if secret:
            text = text.replace(secret, f"<{name}:redacted>")
    return text


def _usage(inference: Any) -> dict[str, int | float]:
    raw = getattr(inference, "usage", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _usage_delta(before: dict, after: dict) -> dict[str, int | float]:
    keys = set(before) | set(after)
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(keys)
        if isinstance(after.get(key, 0), (int, float))
        and isinstance(before.get(key, 0), (int, float))
    }


def _model_name(inference: Any) -> str:
    config = getattr(inference, "config", None)
    provider = getattr(config, "provider", "")
    model = getattr(config, "model_name", "")
    return "/".join(part for part in (provider, model) if part) or type(inference).__name__


def capture_structured_create(
    inference: Any,
    prompt: str,
    response_model: type[BaseModel],
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[BaseModel, str | None]:
    """Call ``inference.create`` and losslessly audit every underlying query.

    ``AIInference.create`` may issue a second query to repair malformed JSON;
    monkeypatching the instance's ``query`` for the duration of this synchronous
    call ensures both messages and both raw responses are recorded.
    """

    path = _audit_path()
    call_group_id = uuid.uuid4().hex
    raw_chunks: list[str] = []
    original = getattr(inference, "query", None)
    query_number = 0

    if path is not None:
        append_journal(
            path,
            {
                "event": "llm_structured_started",
                "call_group_id": call_group_id,
                "stage": stage,
                "model": _model_name(inference),
                "response_schema": response_model.__name__,
                "metadata": metadata or {},
            },
        )

    if callable(original):

        def wrapped(messages, *args, **kwargs):
            nonlocal query_number
            query_number += 1
            started = time.monotonic()
            before = _usage(inference)
            base = {
                "event": "llm_query",
                "call_group_id": call_group_id,
                "query_number": query_number,
                "stage": stage,
                "model": _model_name(inference),
                "response_schema": response_model.__name__,
                "metadata": metadata or {},
                "messages": _redact(messages),
            }
            if path is not None:
                append_journal(
                    path,
                    {
                        **base,
                        "event": "llm_query_started",
                    },
                )
            token = _CURRENT_QUERY.set(
                {
                    "call_group_id": call_group_id,
                    "query_number": query_number,
                    "stage": stage,
                    "model": _model_name(inference),
                    "response_schema": response_model.__name__,
                }
            )
            try:
                response = original(messages, *args, **kwargs)
            except BaseException as exc:
                if path is not None:
                    append_journal(
                        path,
                        {
                            **base,
                            "status": "error",
                            "duration_seconds": time.monotonic() - started,
                            "usage_delta": _usage_delta(before, _usage(inference)),
                            "error_type": type(exc).__name__,
                            "error": _redact(str(exc)),
                        },
                    )
                raise
            finally:
                _CURRENT_QUERY.reset(token)
            raw = response if isinstance(response, str) else str(response or "")
            if raw:
                raw_chunks.append(raw)
            if path is not None:
                append_journal(
                    path,
                    {
                        **base,
                        "status": "ok",
                        "duration_seconds": time.monotonic() - started,
                        "usage_delta": _usage_delta(before, _usage(inference)),
                        "response": _redact(raw),
                    },
                )
            return response

        inference.query = wrapped

    try:
        parsed = inference.create(prompt, response_model)
    except BaseException as exc:
        if path is not None:
            append_journal(
                path,
                {
                    "event": "llm_structured_result",
                    "call_group_id": call_group_id,
                    "stage": stage,
                    "model": _model_name(inference),
                    "response_schema": response_model.__name__,
                    "metadata": metadata or {},
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": _redact(str(exc)),
                },
            )
        raise
    finally:
        if callable(original):
            inference.query = original

    if path is not None:
        payload = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
        append_journal(
            path,
            {
                "event": "llm_structured_result",
                "call_group_id": call_group_id,
                "stage": stage,
                "model": _model_name(inference),
                "response_schema": response_model.__name__,
                "metadata": metadata or {},
                "status": "ok",
                "parsed": _redact(payload),
            },
        )
    return parsed, "\n---\n".join(raw_chunks) if raw_chunks else None
