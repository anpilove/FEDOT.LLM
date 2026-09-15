import json
import math
import os
import queue
import threading
from numbers import Real
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
import tiktoken
from litellm.caching.caching import Cache, LiteLLMCacheType
from pydantic import BaseModel, ValidationError
from openai import OpenAIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fedotllm import prompts
from fedotllm.configs.schema import EmbeddingsConfig, LLMConfig
from fedotllm.log import logger
from fedotllm.utils.parsers import parse_json

T = TypeVar("T", bound=BaseModel)
LLM_TIMEOUT_SECONDS = float(os.getenv("FEDOTLLM_LLM_TIMEOUT", "120"))
# One initial provider request plus at most one transient retry. Clamp the
# legacy environment override so higher layers cannot multiply requests.
LLM_RETRY_ATTEMPTS = min(
    2, max(1, int(os.getenv("FEDOTLLM_LLM_RETRY_ATTEMPTS", "2")))
)

litellm._logging._disable_debugging()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class LLMRequestTimeout(TimeoutError):
    """A provider exceeded the total wall-clock budget for one request."""


class EmptyLLMResponse(RuntimeError):
    """A provider completed a request without emitting user-visible content."""


_NON_RETRYABLE_PROVIDER_MARKERS = (
    "access denied by security policy",
    "content policy",
    "policy violation",
    "request was blocked",
    "moderation",
)


def _retryable_provider_error(exc: BaseException) -> bool:
    """Retry one transient provider failure, never policy or run-budget stops."""

    if type(exc).__name__ == "EvolveBudgetExhausted":
        return False
    if any(marker in str(exc).lower() for marker in _NON_RETRYABLE_PROVIDER_MARKERS):
        return False
    return isinstance(exc, (LLMRequestTimeout, OpenAIError, ConnectionError))


def _provider_retry_wait(retry_state) -> float:
    """Respect numeric Retry-After, including OpenRouter's wrapped error body."""
    fallback = wait_exponential(multiplier=1, min=4, max=10)(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return fallback
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    delay = headers.get("retry-after") or headers.get("Retry-After")
    if delay is None:
        # LiteLLM APIError drops the response/body but retains the JSON message.
        message = str(getattr(exc, "message", ""))
        start = message.find("{")
        try:
            body, _ = json.JSONDecoder().raw_decode(message[start:]) if start >= 0 else ({}, 0)
            metadata = body.get("error", {}).get("metadata", {})
            nested_headers = metadata.get("headers", {})
            delay = nested_headers.get("Retry-After") or nested_headers.get("retry-after")
        except (ValueError, AttributeError, TypeError):
            delay = None
    try:
        seconds = float(delay)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(seconds) or seconds < 0:
        return fallback
    # Keep the existing retry count bounded; do not resend before the provider's
    # requested delay. Logging exposes the pause without printing the error body.
    wait = max(fallback, seconds)
    logger.info("LLM provider requested retry after %gs", wait)
    return wait


def completion_with_timeout(messages: list[dict[str, Any]], params: dict[str, Any]):
    """Run LiteLLM in a daemon thread so streaming heartbeats cannot hang the agent."""
    timeout = float(params.get("timeout", LLM_TIMEOUT_SECONDS))
    result: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def call() -> None:
        try:
            result.put(("response", litellm.completion(messages=messages, **params)))
        except BaseException as exc:
            result.put(("error", exc))

    worker = threading.Thread(target=call, name="fedotllm-completion", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise LLMRequestTimeout(f"LLM request exceeded {timeout:g}s wall-clock timeout")
    kind, value = result.get_nowait()
    if kind == "error":
        raise value
    return value


class AIInference:
    def __init__(self, config: LLMConfig, session_id: Optional[str] = None):
        self.config = config
        self.usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.0,
        }

        if not self.config.api_key:
            raise ValueError(
                "API key not provided and FEDOTLLM_LLM_API_KEY environment variable not set"
            )

        self.completion_params = {
            "model": f"{config.provider}/{config.model_name}",
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
            "extra_headers": self.config.extra_headers,
            "metadata": {"session_id": session_id},
            "timeout": LLM_TIMEOUT_SECONDS,
            **self.config.completion_params,
        }
        if "FEDOTLLM_LLM_TIMEOUT" in os.environ:
            self.completion_params["timeout"] = LLM_TIMEOUT_SECONDS

        if config.caching.enabled:
            litellm.cache = Cache(
                type=LiteLLMCacheType.DISK, disk_cache_dir=config.caching.dir_path
            )

    def _primary_model(self) -> str:
        return f"{self.config.provider}/{self.config.model_name}"

    def _model_chain(self) -> list[str]:
        primary = self._primary_model()
        raw = os.environ.get("FEDOTLLM_LLM_FALLBACK") or self.config.fallback_models or ""
        prefix = f"{self.config.provider}/"
        chain = [primary]
        for item in raw.split(","):
            name = item.strip()
            if not name:
                continue
            model = name if "/" in name and name.startswith(prefix) else f"{prefix}{name}"
            if model not in chain:
                chain.append(model)
        return chain

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=2),
        retry=retry_if_exception(lambda exc: isinstance(exc, EmptyLLMResponse)),
        reraise=True,
    )
    def create(self, messages: str, response_model: Type[T]) -> T:
        """Create one typed response with at most one JSON-correction query.

        Transport retries belong exclusively to :meth:`query`.  Retrying this
        whole method used to multiply provider calls after transport errors.
        An empty completed response gets one fresh request; schema correction
        remains bounded to one query inside the same structured call.
        """
        messages = f"{messages}\n{prompts.utils.structured_response(response_model)}"
        response = self.query(messages)
        if not response or not response.strip():
            raise EmptyLLMResponse("LLM returned no structured response")
        json_obj = parse_json(response)
        try:
            return response_model.model_validate(json_obj)
        except ValidationError as exc_info:
            messages = f"{prompts.utils.fix_structured_response(json_obj, str(exc_info), response_model)}"
            response = self.query(messages)
            json_obj = parse_json(response) if response else None
            return response_model.model_validate(json_obj)

    @retry(
        stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
        wait=_provider_retry_wait,
        retry=retry_if_exception(_retryable_provider_error),
        reraise=True,
    )
    def query(self, messages: str | List[Dict[str, Any]]) -> str | None:
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        last: LLMRequestTimeout | None = None
        try:
            for model in self._model_chain():
                self.completion_params["model"] = model
                logger.debug(
                    "LLM request %s: %s messages, %s chars",
                    model,
                    len(messages),
                    sum(len(str(item.get("content", ""))) for item in messages),
                )
                try:
                    return self._complete(messages)
                except LLMRequestTimeout as exc:
                    last = exc
                    logger.warning("LLM timeout on %s, trying fallback", model)
                    continue
        finally:
            self.completion_params["model"] = self._primary_model()
        assert last is not None
        raise last

    def _complete(self, messages: list[dict[str, Any]]) -> str:
        response = completion_with_timeout(messages, self.completion_params)
        usage = getattr(response, "usage", None)

        def number(source: Any, key: str) -> float:
            value = (
                source.get(key, 0)
                if isinstance(source, dict)
                else getattr(source, key, 0)
                if source is not None
                else 0
            )
            return float(value) if isinstance(value, Real) else 0.0

        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
            if usage is not None
            else None
        )
        hidden = getattr(response, "_hidden_params", None)
        cost = number(usage, "cost") or number(hidden, "response_cost")
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += int(number(usage, "prompt_tokens"))
        self.usage["completion_tokens"] += int(number(usage, "completion_tokens"))
        self.usage["cached_tokens"] += int(number(details, "cached_tokens"))
        self.usage["cost_usd"] += cost
        content = response.choices[0].message.content or ""
        logger.debug("LLM response %s: %s chars", self.completion_params["model"], len(content))
        return content


class LiteLLMEmbeddings:
    MAX_INPUT = 8191

    def __init__(self, config: EmbeddingsConfig):
        if not config.api_key:
            raise Exception(
                "OpenAI API env variable FEDOTLLM_EMBEDDINGS_API_KEY not set"
            )

        self.embedding_params = {
            "model": f"{config.provider}/{config.model_name}",
            "api_key": config.api_key,
            "base_url": config.base_url,
            **config.embedding_params,
        }

    def encode(self, input: str):
        try:
            response = litellm.embedding(
                input=input, encoding_format="float", **self.embedding_params
            )
        except Exception:
            len_embeddings = num_tokens_from_string(input)
            if len_embeddings > self.MAX_INPUT:
                raise Exception(f"Input exceeds the limit of <{self.model}>!")
            else:
                raise Exception("Embeddings generation failed!")
        return response.data


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


if __name__ == "__main__":
    from fedotllm.configs.loader import load_config

    config = load_config()
    inference = AIInference(config.llm)
    print(inference.query("Say hello world!"))
