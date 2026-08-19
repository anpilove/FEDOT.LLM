import os
import queue
import threading
from numbers import Real
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
import tiktoken
from litellm.caching.caching import Cache, LiteLLMCacheType
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from fedotllm import prompts
from fedotllm.configs.schema import EmbeddingsConfig, LLMConfig
from fedotllm.log import logger
from fedotllm.utils.parsers import parse_json

T = TypeVar("T", bound=BaseModel)
LLM_TIMEOUT_SECONDS = float(os.getenv("FEDOTLLM_LLM_TIMEOUT", "120"))
LLM_RETRY_ATTEMPTS = int(os.getenv("FEDOTLLM_LLM_RETRY_ATTEMPTS", "2"))

litellm._logging._disable_debugging()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class LLMRequestTimeout(TimeoutError):
    """A provider exceeded the total wall-clock budget for one request."""


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

    @retry(
        stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type(LLMRequestTimeout),
        reraise=True,
    )
    def create(self, messages: str, response_model: Type[T]) -> T:
        messages = f"{messages}\n{prompts.utils.structured_response(response_model)}"
        response = self.query(messages)
        json_obj = parse_json(response) if response else None
        try:
            return response_model.model_validate(json_obj)
        except ValidationError as exc_info:
            messages = f"{prompts.utils.fix_structured_response(json_obj, str(exc_info), response_model)}"
            response = self.query(messages)
            json_obj = parse_json(response) if response else None
            return response_model.model_validate(json_obj)

    @retry(
        stop=stop_after_attempt(LLM_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type(LLMRequestTimeout),
        reraise=True,
    )
    def query(self, messages: str | List[Dict[str, Any]]) -> str | None:
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        logger.debug("Sending messages to LLM: %s", messages)
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
        logger.debug(
            "Received response from LLM: %s", response.choices[0].message.content
        )
        return response.choices[0].message.content


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
