from __future__ import annotations

from pathlib import Path

import pytest

from fedotllm.agents.evolve.execution.snippet_policy import (
    ast_import_blocked,
    redact_snippet_output,
    snippet_runtime_guards,
    source_token_blocked,
)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_snippet_rejects_unbounded_or_nonpositive_deadline_before_selecting_python(
    tmp_path, monkeypatch, timeout
):
    from fedotllm.agents.evolve.execution import run_code

    def must_not_start(_checkout):
        raise AssertionError(
            "invalid timeout must be rejected before interpreter selection"
        )

    monkeypatch.setattr(run_code, "fedot_python", must_not_start)
    with pytest.raises(ValueError, match="timeout_s must be finite and positive"):
        run_code.run_fedot_snippet(tmp_path, "print(1)", timeout_s=timeout)


def test_real_snippet_preserves_redacted_streams_and_runtime_error(
    tmp_path, monkeypatch
):
    import sys
    from fedotllm.agents.evolve.execution import run_code

    package = tmp_path / "fedot"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(run_code, "fedot_python", lambda _: sys.executable)
    result = run_code.run_fedot_snippet(
        tmp_path,
        "import sys\nprint('token=fixturesecret', flush=True)\n"
        "print('password=fixturesecret', file=sys.stderr, flush=True)\nraise RuntimeError('fixture crash')",
        timeout_s=5,
    )
    assert result.status == "runtime_error"
    assert result.exit_code != 0
    assert "REDACTED" in result.stdout and "fixturesecret" not in result.stdout
    assert "REDACTED" in result.stderr and "fixturesecret" not in result.stderr
    assert "fixture crash" in result.stderr


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


def test_snippet_runtime_guards_allow_package_resource_reads_from_that_package(
    tmp_path: Path,
):
    import importlib.util

    checkout = tmp_path / "exp"
    checkout.mkdir()
    site_packages = tmp_path / "site-packages"
    package = site_packages / "xgboost"
    package.mkdir(parents=True)
    resource = package / "VERSION"
    resource.write_text("3.0.0", encoding="utf-8")
    module_path = package / "reader.py"
    module_path.write_text(
        "from pathlib import Path\n"
        "def read_version():\n"
        "    with open(Path(__file__).with_name('VERSION')) as handle:\n"
        "        return handle.read()\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("xgboost.reader", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with snippet_runtime_guards(checkout, runtime_package_roots=(site_packages,)):
        assert module.read_version() == "3.0.0"
