"""The controller is an independently usable module and deployment."""

import json
import subprocess
import sys

from fedotllm.agents.evolve.execution.guard import repo_root


def test_controller_help_works_without_llm_or_ml_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "fedotllm.agents.evolve.controller", "--help"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "serve" in result.stdout and "status" in result.stdout


def test_controller_info_identifies_the_actual_module():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "fedotllm.agents.evolve.controller", "info"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "FEDOT.LLM Evolve Controller"
    assert payload["module"] == "fedotllm.agents.evolve.controller"
    assert payload["controller_sha256"]


def test_status_does_not_create_or_modify_a_run(tmp_path):
    workspace = tmp_path / "run"
    workspace.mkdir()
    original = '{"state":"running","heartbeat_at":"old"}\n'
    (workspace / "logs").mkdir()
    (workspace / "logs/service_status.json").write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fedotllm.agents.evolve.controller",
            "status",
            "--workspace",
            str(workspace),
        ],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["state"] == "running"
    assert (workspace / "logs/service_status.json").read_text() == original
    assert not (workspace / "quality_queue").exists()


def test_status_reads_a_flat_saved_snapshot_without_modifying_it(tmp_path):
    workspace = tmp_path / "flat-snapshot"
    workspace.mkdir()
    original = '{"state":"running","heartbeat_at":"snapshot"}\n'
    status_path = workspace / "service_status.json"
    status_path.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fedotllm.agents.evolve.controller",
            "status",
            "--workspace",
            str(workspace),
        ],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"state": "running", "heartbeat_at": "snapshot"}
    assert status_path.read_text() == original


def test_direct_drain_scopes_worker_artifacts_to_its_workspace(
    tmp_path, monkeypatch, capsys
):
    import os
    from pathlib import Path

    from fedotllm.agents.evolve.controller import __main__ as controller_cli
    from fedotllm.agents.evolve.controller import quality_executor

    observed = []

    def drain(_source, workspace, **_kwargs):
        observed.append(Path(os.environ["EVOLVE_AGENT_TMP"]))
        return {"jobs": []}

    with monkeypatch.context() as scoped:
        # Record values before main() changes them directly so this in-process
        # CLI fixture cannot leak its environment into later tests.
        scoped.setenv("EVOLVE_AGENT_TMP", "")
        scoped.setenv("FEDOTLLM_REPO_PATH", "")
        scoped.setattr(quality_executor, "drain_quality_queue", drain)
        for name in ("run-a", "run-b"):
            workspace = tmp_path / name
            assert (
                controller_cli.main(
                    [
                        "drain",
                        "--fedot",
                        str(repo_root()),
                        "--workspace",
                        str(workspace),
                    ]
                )
                == 0
            )
            capsys.readouterr()

    assert observed == [
        tmp_path / "run-a" / "worker_artifacts",
        tmp_path / "run-b" / "worker_artifacts",
    ]
