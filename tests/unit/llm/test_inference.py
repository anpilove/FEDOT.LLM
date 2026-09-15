import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import litellm
from pydantic import BaseModel, Field, ValidationError
from tenacity import wait_none

from fedotllm.configs.schema import LLMConfig
from fedotllm.llm import AIInference, EmptyLLMResponse, LLMRequestTimeout, _provider_retry_wait


def _inflight_error():
    return litellm.APIError(
        status_code=402,
        message='APIError: OpenrouterException - ' + json.dumps({
            'error': {'metadata': {'reason': 'in_flight_budget_exhausted',
                                  'headers': {'Retry-After': '120'}}}
        }),
        llm_provider='openrouter', model='test-model',
    )


def test_query_honors_inflight_retry_delay(llm_config, monkeypatch):
    inference = AIInference(llm_config)
    inference._complete = MagicMock(side_effect=[_inflight_error(), 'ok'])
    pauses = []
    monkeypatch.setattr(inference.query.retry, 'sleep', pauses.append)
    monkeypatch.setattr(inference.query.retry, 'wait', _provider_retry_wait)
    assert inference.query('same request') == 'ok'
    assert pauses == [120.0]
    assert inference._complete.call_count == 2
    assert inference._complete.call_args_list[0] == inference._complete.call_args_list[1]


def test_structured_call_does_not_multiply_transport_retries(llm_config, monkeypatch):
    inference = AIInference(llm_config)
    inference._complete = MagicMock(side_effect=_inflight_error())
    pauses = []
    monkeypatch.setattr(inference.query.retry, 'sleep', pauses.append)
    monkeypatch.setattr(inference.query.retry, 'wait', _provider_retry_wait)
    monkeypatch.setattr(inference.create.retry, 'sleep', lambda _: None)
    with pytest.raises(litellm.APIError):
        inference.create('request', response_model=UserModel)
    assert inference._complete.call_count == 2
    assert pauses == [120.0]


@pytest.mark.parametrize('delay,expected', [('27', 27), ('nan', 4), ('-1', 4), ('invalid', 4)])
def test_provider_retry_http_headers(delay, expected):
    exc = SimpleNamespace(response=SimpleNamespace(headers={'retry-after': delay}))
    state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc), attempt_number=1)
    assert _provider_retry_wait(state) == expected


class UserModel(BaseModel):
    """Test model for structured response testing"""

    name: str = Field(..., description="User name")
    age: int = Field(..., description="User age", ge=0, le=120)
    email: str = Field(..., description="User email")
    active: bool = Field(default=True, description="User active status")


# Fixtures
@pytest.fixture
def llm_config():
    config = LLMConfig(
        provider="test-provider",
        model_name="test-model",
        base_url="https://test.api.com",
        api_key="sk-12345",
        extra_headers={"X-Title": "FEDOT.LLM-Test"},
        completion_params={"temperature": 1.0},
    )
    return config


@patch("fedotllm.llm.litellm")
def test_query(mock_litellm, llm_config):
    """Test querying with AIInference"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello, world!"))],
    )
    inference = AIInference(llm_config)
    response = inference.query("Say hello")
    assert response == "Hello, world!"
    assert inference.completion_params["timeout"] > 0


@patch("fedotllm.llm.litellm")
def test_query_accumulates_provider_usage(mock_litellm, llm_config):
    mock_litellm.completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 80},
            "cost": 0.0042,
        },
        _hidden_params={},
    )
    inference = AIInference(llm_config)

    inference.query("first")
    inference.query("second")

    assert inference.usage == {
        "requests": 2,
        "prompt_tokens": 240,
        "completion_tokens": 60,
        "cached_tokens": 160,
        "cost_usd": pytest.approx(0.0084),
    }


@patch("fedotllm.llm.litellm")
def test_query_has_a_wall_clock_timeout(mock_litellm, llm_config, monkeypatch):
    llm_config.completion_params["timeout"] = 0.01

    def hangs(**_kwargs):
        time.sleep(0.1)

    mock_litellm.completion.side_effect = hangs
    inference = AIInference(llm_config)
    monkeypatch.setattr(inference.query.retry, "wait", wait_none())

    with pytest.raises(LLMRequestTimeout, match="wall-clock"):
        inference.query("never finishes")
    assert mock_litellm.completion.call_count == 2


@patch("fedotllm.llm.litellm")
def test_query_falls_back_after_timeout(mock_litellm, llm_config):
    llm_config.completion_params["timeout"] = 0.01
    llm_config.fallback_models = "small-model"

    def hang_then_ok(**kwargs):
        if kwargs["model"].endswith("small-model"):
            return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])
        time.sleep(0.1)
        raise AssertionError("primary should have been timed out")

    mock_litellm.completion.side_effect = hang_then_ok
    inference = AIInference(llm_config)
    assert inference.query("hello") == "ok"
    models = [call.kwargs["model"] for call in mock_litellm.completion.call_args_list]
    assert models[0].endswith("test-model")
    assert models[-1].endswith("small-model")
    assert inference.completion_params["model"].endswith("test-model")


@patch("fedotllm.llm.litellm")
def test_query_does_not_stick_on_fallback(mock_litellm, llm_config):
    llm_config.completion_params["timeout"] = 0.01
    llm_config.fallback_models = "small-model"
    n = {"i": 0}

    def hang_once_then_primary(**kwargs):
        n["i"] += 1
        if n["i"] == 1:
            time.sleep(0.1)
            raise AssertionError("primary should have been timed out")
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    mock_litellm.completion.side_effect = hang_once_then_primary
    inference = AIInference(llm_config)
    inference.query("first")
    assert inference.query("second") == "ok"
    models = [call.kwargs["model"] for call in mock_litellm.completion.call_args_list]
    assert models[-1].endswith("test-model")


def test_create_structured_object(llm_config):
    """Test creating a structured object with AIInference"""
    inference = AIInference(llm_config)
    inference.query = (
        lambda *args,
        **kwargs: '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}'
    )
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(messages="", response_model=UserModel)
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True


def test_create_structured_object_invalid(llm_config):
    """Test creating a structured object that fails validation"""

    inference = AIInference(llm_config)
    inference.query = (
        lambda *args,
        **kwargs: '{"name": "John Doe", "age": 150, "email": "john@example.com", "active": true}'
    )
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*less than or equal to 120.*"):
        inference.create(messages="", response_model=UserModel)


def test_create_structured_object_missing_field(llm_config):
    """Test creating a structured object with missing fields"""
    inference = AIInference(llm_config)
    inference.query = lambda *args, **kwargs: '{"name": "John Doe", "age": 30}'
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=r".*[fF]ield required.*"):
        inference.create(messages="", response_model=UserModel)


@pytest.mark.parametrize(
    "response,expected_error",
    [
        (
            'name: "John Doe", "age": 30, "email": "john@example.com", "active": true',
            r".*valid dictionary.*",
        ),
    ],
)
def test_create_structured_object_invalid_format(response, expected_error, llm_config):
    """Test creating a structured object with invalid response"""

    inference = AIInference(llm_config)
    inference.query = lambda *args, **kwargs: response
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    with pytest.raises(ValidationError, match=expected_error):
        inference.create(messages="", response_model=UserModel)


@pytest.mark.parametrize("response", ["", None])
def test_create_retries_empty_provider_response(response, llm_config):
    inference = AIInference(llm_config)
    inference.query = MagicMock(return_value=response)
    inference.create.retry.wait = wait_none()

    with pytest.raises(EmptyLLMResponse, match="no structured response"):
        inference.create(messages="", response_model=UserModel)

    assert inference.query.call_count == 2


@pytest.mark.parametrize(
    "response",
    [
        '{"name": "John Doe", "age": 30, "email": "john@example.com", "active": true}',
    ],
)
@patch("fedotllm.llm.litellm")
def test_create_structured_use_query_valid(mock_litellm, response, llm_config):
    """Test creating a structured object using query method"""
    mock_litellm.completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))],
    )
    inference = AIInference(llm_config)
    inference.create.retry.wait = wait_none()  # Disable retry for this test
    user = inference.create(messages="Create a user object", response_model=UserModel)
    assert user.name == "John Doe"
    assert user.age == 30
    assert user.email == "john@example.com"
    assert user.active is True
