from __future__ import annotations

import json
from types import SimpleNamespace

from fedotllm.agents.evolve.controller.campaign import run_once

from fedotllm.agents.evolve.model_contract import (
    KNOWN_MODEL_STAGES,
    REQUIRED_MODEL_NAME,
    build_model_contract,
    model_contract_diagnostics,
)


def _client(
    model_name: str = REQUIRED_MODEL_NAME,
    *,
    fallback_models: str = "",
):
    return SimpleNamespace(
        config=SimpleNamespace(
            provider="openrouter",
            model_name=model_name,
            fallback_models=fallback_models,
        )
    )


def test_model_contract_accepts_glm_for_every_known_stage():
    payload = build_model_contract(
        {stage: _client() for stage in KNOWN_MODEL_STAGES},
        environ={},
    )

    assert payload["ok"] is True
    assert payload["diagnostics"] == []
    assert payload["checked_stages"] == list(KNOWN_MODEL_STAGES)
    assert payload["ignored_stages"] == []
    assert all(
        row["model_name"] == REQUIRED_MODEL_NAME
        and row["effective_fallback_models"] == []
        for row in payload["stages"].values()
    )
    json.dumps(payload)


def test_model_contract_ignores_test_doubles_without_config():
    payload = build_model_contract(
        {"scout": object(), "verifier": None, "fixer": object()},
        environ={},
    )

    assert payload["ok"] is True
    assert payload["checked_stages"] == []
    assert payload["ignored_stages"] == list(KNOWN_MODEL_STAGES)
    assert payload["stages"]["scout"] == {"status": "ignored_test_double"}
    assert payload["stages"]["verifier"] == {"status": "not_supplied"}


def test_model_contract_rejects_wrong_model_with_stage_diagnostic():
    payload = build_model_contract(
        {
            "scout": _client("openai/gpt-5-mini"),
            "verifier": _client(),
            "fixer": _client(),
        },
        environ={},
    )

    assert payload["ok"] is False
    assert payload["diagnostics"] == [
        "scout: model_name must be 'z-ai/glm-5.3-flash', got "
        "'openai/gpt-5-mini'"
    ]
    assert payload["stages"]["scout"]["qualified_model"] == (
        "openrouter/openai/gpt-5-mini"
    )


def test_model_contract_rejects_configured_fallbacks_for_each_stage():
    payload = build_model_contract(
        {
            "scout": _client(fallback_models="openai/gpt-5-mini, anthropic/claude"),
            "verifier": _client(),
            "fixer": _client(),
        },
        environ={},
    )

    assert payload["ok"] is False
    assert payload["stages"]["scout"]["effective_fallback_models"] == [
        "openai/gpt-5-mini",
        "anthropic/claude",
    ]
    assert payload["diagnostics"] == [
        "scout: fallback_models must be empty, got "
        "openai/gpt-5-mini, anthropic/claude"
    ]


def test_model_contract_rejects_environment_fallback_for_real_clients_only():
    clients = {
        "scout": _client(),
        "verifier": object(),
        "fixer": _client(),
    }
    payload = build_model_contract(
        clients,
        environ={"FEDOTLLM_LLM_FALLBACK": "openai/gpt-5-mini"},
    )

    assert payload["ok"] is False
    assert payload["stages"]["verifier"] == {"status": "ignored_test_double"}
    assert payload["diagnostics"] == [
        "scout: FEDOTLLM_LLM_FALLBACK must be empty, got openai/gpt-5-mini",
        "fixer: FEDOTLLM_LLM_FALLBACK must be empty, got openai/gpt-5-mini",
    ]
    assert model_contract_diagnostics(clients, environ={}) == []


def test_model_contract_treats_config_without_model_name_as_real_and_invalid():
    payload = build_model_contract(
        {"scout": SimpleNamespace(config=SimpleNamespace())},
        environ={},
    )

    assert payload["ok"] is False
    assert payload["diagnostics"] == [
        "scout: model_name must be 'z-ai/glm-5.3-flash', got ''"
    ]


def test_campaign_stops_before_any_request_when_model_contract_is_invalid(tmp_path):
    source = tmp_path / "source"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/__init__.py").write_text("", encoding="utf-8")

    class Client:
        def __init__(self):
            self.config = SimpleNamespace(
                provider="openrouter",
                model_name="openai/gpt-5-mini",
                fallback_models="",
            )
            self.calls = 0

        def create(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("model request must not start")

    client = Client()
    workspace = tmp_path / "work"
    decision = run_once(
        checkout=source,
        scout_inference=client,
        verifier_inference=client,
        fixer_inference=client,
        workspace=workspace,
    )

    assert decision.infrastructure_error is True
    assert decision.reason.startswith("model_contract_invalid:")
    assert client.calls == 0
    artifact = json.loads((workspace / "model_contract.json").read_text())
    assert artifact["ok"] is False
