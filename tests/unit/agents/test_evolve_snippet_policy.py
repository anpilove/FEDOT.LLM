from __future__ import annotations

from pathlib import Path

import pytest

from fedotllm.agents.evolve.execution.snippet_policy import (
    ast_import_blocked,
    redact_snippet_output,
    snippet_runtime_guards,
    source_token_blocked,
)


def test_source_token_blocked():
    assert source_token_blocked("open('cases.json')") == "cases.json"
    assert source_token_blocked("print(1)") is None


def test_ast_import_blocked_importlib():
    code = "import importlib\nimportlib.import_module('fedotllm.agents.evolve.evaluation.scorer')"
    assert ast_import_blocked(code) == "fedotllm.agents.evolve.evaluation.scorer"


def test_redact_snippet_output_masks_common_secrets():
    text = "api_key=abc123 token=deadbeef sk-abcdefghijklmnopqrstuvwxyz123456"
    redacted = redact_snippet_output(text)
    assert "abc123" not in redacted
    assert "REDACTED" in redacted


def test_snippet_runtime_guards_block_outside_checkout(tmp_path: Path):
    checkout = tmp_path / "exp"
    checkout.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("leak", encoding="utf-8")
    with snippet_runtime_guards(checkout):
        with pytest.raises(PermissionError):
            open(outside, encoding="utf-8").read()
