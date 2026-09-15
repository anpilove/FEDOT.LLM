"""Typed failures at the boundary between EvolveAgent and an LLM provider."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from fedotllm.agents.evolve.storage.run_budget import EvolveBudgetExhausted
from fedotllm.llm import EmptyLLMResponse, LLMRequestTimeout


_POLICY_MARKERS = (
    "access denied by security policy",
    "content policy",
    "policy violation",
    "request was blocked",
    "moderation",
)


@dataclass
class AgentModelFailure(RuntimeError):
    """A model call failed before EvolveAgent obtained a usable typed action."""

    category: str
    detail: str
    infrastructure: bool
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.category}: {self.detail}"


def classify_model_failure(exc: Exception) -> AgentModelFailure:
    """Map provider, budget and response failures to stable campaign outcomes."""

    detail = f"{type(exc).__name__}: {exc}"[:1_000]
    lowered = str(exc).lower()
    if isinstance(exc, EvolveBudgetExhausted):
        return AgentModelFailure("budget_exhausted", detail, True, False)
    if isinstance(exc, LLMRequestTimeout):
        return AgentModelFailure("timeout", detail, True, True)
    if any(marker in lowered for marker in _POLICY_MARKERS):
        return AgentModelFailure("provider_policy", detail, True, False)
    if isinstance(exc, (ValidationError, json.JSONDecodeError, EmptyLLMResponse)):
        return AgentModelFailure("invalid_response", detail, False, False)
    module = type(exc).__module__.lower()
    if module.startswith(("litellm", "openai", "httpx", "httpcore")):
        return AgentModelFailure("provider_error", detail, True, True)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return AgentModelFailure("provider_error", detail, True, True)
    return AgentModelFailure("unexpected_model_error", detail, True, False)
