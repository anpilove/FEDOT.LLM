"""The server launcher validates a fixed release and never reuses an old run."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from fedotllm.agents.evolve.controller.supervisor import _controller_version
from fedotllm.agents.evolve.execution.guard import repo_root


def command(tmp_path):
    source = tmp_path / "FEDOT"
    (source / "fedot").mkdir(parents=True)
    (source / "fedot/version.py").write_text("version = 'fixture'\n")
    return [
        sys.executable,
        str(repo_root() / "research/evolve/scripts/launch_quality_service.py"),
        "--controller-root",
        str(repo_root()),
        "--fedot-source",
        str(source),
        "--python",
        sys.executable,
        "--workspace",
        str(tmp_path / "new-run"),
        "--expected-controller-sha256",
        _controller_version(),
    ]


def test_default_launcher_only_checks_and_does_not_create_a_run(tmp_path):
    result = subprocess.run(
        command(tmp_path), capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "check"
    assert payload["command"][1:4] == [
        "-m",
        "fedotllm.agents.evolve.controller",
        "serve",
    ]
    assert "--hunt" not in payload["command"]
    assert not (tmp_path / "new-run").exists()


def _proposal_seed(tmp_path):
    source = tmp_path / "FEDOT"
    (source / "fedot/version.py").write_text("version = 'fixture'\n")
    seed = tmp_path / "old-quality-queue"
    seed.mkdir()
    patch = "--- a/fedot/version.py\n+++ b/fedot/version.py\n@@ -1 +1 @@\n-version = 'fixture'\n+version = 'changed'\n"
    (seed / "improvement.patch").write_text(patch)
    (seed / "improvement.json").write_text(
        json.dumps(
            {
                "candidate_id": "improvement",
                "file_path": "fedot/version.py",
                "patch": "improvement.patch",
                "status": "running",
                "hypothesis": {"claim": "Original proposed improvement"},
            }
        )
    )
    return seed


def test_launcher_checks_proposals_and_seeds_a_frozen_queue_before_start(tmp_path):
    args = command(tmp_path)
    seed = _proposal_seed(tmp_path)
    original = (seed / "improvement.json").read_bytes()
    args += ["--proposals-from", str(seed)]
    check = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["initial_proposal_count"] == 1
    assert not (tmp_path / "new-run").exists()
    launched = subprocess.run(
        [*args, "--mode", "launch", "--duration-seconds", "0.1"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert launched.returncode == 0, launched.stderr
    audit = json.loads(launched.stdout)
    try:
        proposal = json.loads(
            (tmp_path / "new-run/quality_queue/improvement.json").read_text()
        )
        assert proposal["source_sha256"]
        assert proposal["check_identity"]["registry_sha256"]
        assert len(proposal["required_task_ids"]) == 40
        assert (seed / "improvement.json").read_bytes() == original
        assert not (tmp_path / "quality_queue").exists()
    finally:
        try:
            os.killpg(audit["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_launcher_refuses_a_seed_with_a_changed_patch_before_creating_workspace(
    tmp_path,
):
    args = command(tmp_path)
    seed = _proposal_seed(tmp_path)
    path = seed / "improvement.json"
    record = json.loads(path.read_text())
    record["patch_sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    result = subprocess.run(
        [*args, "--proposals-from", str(seed), "--mode", "launch"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "patch hash mismatch" in result.stderr
    assert not (tmp_path / "new-run").exists()


def test_launcher_refuses_to_overwrite_an_existing_run(tmp_path):
    args = command(tmp_path)
    workspace = tmp_path / "new-run"
    workspace.mkdir()
    journal = workspace / "quality_jobs.jsonl"
    journal.write_text("preserve this old result\n")
    result = subprocess.run(
        [*args, "--mode", "launch"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert journal.read_text() == "preserve this old result\n"


def test_launcher_requires_explicit_stale_run_ack_and_preserves_old_status(tmp_path):
    args = command(tmp_path)
    previous = tmp_path / "previous-run"
    (previous / "logs").mkdir(parents=True)
    (previous / ".quality_service.lock").touch()
    (previous / ".quality_queue.lock").touch()
    status_path = previous / "logs/service_status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "child_running": True,
                "heartbeat_at": "2026-10-04T23:22:48+00:00",
                "active_child": {"kind": "quality-drain", "log": "/old/controller.log"},
            }
        )
    )
    (previous / "logs/service-launch.log").write_text("")
    original = status_path.read_bytes()
    args += ["--previous-run", str(previous)]

    refused = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert refused.returncode != 0
    assert "no terminal lifecycle evidence" in refused.stderr
    assert not (tmp_path / "new-run").exists()

    checked = subprocess.run(
        [*args, "--allow-stale-previous-run"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert checked.returncode == 0, checked.stderr
    audit = json.loads(checked.stdout)
    assert audit["previous_run"]["state"] == "stale_unconfirmed"
    assert "no active owner" in audit["previous_run"]["locks"]
    assert audit["previous_run"]["terminal_outcome"].startswith("unknown")
    assert status_path.read_bytes() == original
    assert not (tmp_path / "new-run").exists()


def test_launcher_rejects_a_different_release_hash(tmp_path):
    args = command(tmp_path)
    args[-1] = "0" * 64
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "controller hash mismatch" in result.stderr
    assert not (tmp_path / "new-run").exists()


def test_launcher_hashes_the_requested_release_instead_of_imported_checkout(tmp_path):
    args = command(tmp_path)
    other_release = tmp_path / "different-release"
    package = other_release / "fedotllm/agents/evolve/controller"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Different controller source\n")
    args[args.index("--controller-root") + 1] = str(other_release)
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "controller hash mismatch" in result.stderr
    assert not (tmp_path / "new-run").exists()


def test_library_launcher_pins_the_physical_release_instead_of_current_alias(tmp_path):
    args = command(tmp_path)
    library = tmp_path / "FEDOT.LLM"
    release = library / "releases/version-one/library"
    shutil.copytree(
        repo_root() / "fedotllm",
        release / "fedotllm",
        ignore=shutil.ignore_patterns("__pycache__", "cache"),
    )
    (library / "current").symlink_to(
        "releases/version-one/library", target_is_directory=True
    )
    (library / "fedotllm").symlink_to("current/fedotllm", target_is_directory=True)
    args[args.index("--controller-root") + 1] = str(library)
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit["controller_root"] == str(release.resolve())
    assert not (tmp_path / "new-run").exists()


def test_launch_keeps_selected_library_when_current_changes_before_child_import(
    tmp_path,
):
    args = command(tmp_path)
    library = tmp_path / "FEDOT.LLM"
    release = library / "releases/version-one/library"
    shutil.copytree(
        repo_root() / "fedotllm",
        release / "fedotllm",
        ignore=shutil.ignore_patterns("__pycache__", "cache"),
    )
    new_release = library / "releases/version-two/library"
    (new_release / "fedotllm").mkdir(parents=True)
    (new_release / "fedotllm/__init__.py").write_text("# different release\n")
    current = library / "current"
    current.symlink_to("releases/version-one/library", target_is_directory=True)
    (library / "fedotllm").symlink_to("current/fedotllm", target_is_directory=True)
    python = tmp_path / "delayed_python_fixture"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "root = Path(__file__).parent\n(root / 'child-ready').touch()\n"
        "deadline = time.monotonic() + 10\n"
        "while not (root / 'allow-import').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "sys.path[0] = os.environ['PYTHONPATH'].split(os.pathsep)[1]\n"
        "import fedotllm\n"
        "(root / 'selected-library.json').write_text(json.dumps({\n"
        "    'module': str(Path(fedotllm.__file__).resolve()), 'cwd': os.getcwd(),\n"
        "    'pythonpath': os.environ['PYTHONPATH']}))\n"
        "time.sleep(10)\n"
    )
    python.chmod(0o700)
    args[args.index("--controller-root") + 1] = str(library)
    args[args.index("--python") + 1] = str(python)
    result = subprocess.run(
        [*args, "--mode", "launch"], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "child-ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "child-ready").exists()
        replacement = library / "next-current"
        replacement.symlink_to("releases/version-two/library", target_is_directory=True)
        replacement.replace(current)
        (tmp_path / "allow-import").touch()
        output = tmp_path / "selected-library.json"
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert output.is_file()
        selected = json.loads(output.read_text())
        assert selected["module"] == str(release / "fedotllm/__init__.py")
        assert selected["cwd"] == str(tmp_path / "new-run")
        assert str(release) in selected["pythonpath"].split(os.pathsep)
        assert str(library) not in selected["pythonpath"].split(os.pathsep)
    finally:
        # Only the process this test launched; no process discovery.
        try:
            os.killpg(audit["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_launcher_starts_the_real_idle_cli_and_records_its_outcome(tmp_path):
    args = command(tmp_path)
    source_file = tmp_path / "FEDOT/fedot/version.py"
    original = source_file.read_bytes()
    result = subprocess.run(
        [*args, "--mode", "launch", "--duration-seconds", "0.1"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit["pid"] > 0
    status_path = tmp_path / "new-run/logs/service_status.json"
    deadline = time.monotonic() + 12
    status = {}
    while time.monotonic() < deadline:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") == "duration_exhausted":
                break
        time.sleep(0.05)
    assert status.get("state") == "duration_exhausted"
    assert status["controller_launches"] == 0
    assert status["hunt_campaigns"] == 0
    assert source_file.read_bytes() == original
    assert (
        json.loads((tmp_path / "new-run/launch_audit.json").read_text())["pid"]
        == audit["pid"]
    )


@pytest.mark.parametrize(
    "damage", ["header", "missing-target", "missing-search", "syntax"]
)
def test_launcher_refuses_an_inapplicable_seed_without_writing_source(tmp_path, damage):
    args = command(tmp_path)
    seed = _proposal_seed(tmp_path)
    path = seed / "improvement.patch"
    patch = path.read_text()
    if damage == "header":
        patch = patch.replace("@@ -1 +1 @@", "@@ invalid header")
    elif damage == "missing-target":
        patch = patch.replace("fedot/version.py", "fedot/absent.py")
    elif damage == "missing-search":
        patch = patch.replace("-version = 'fixture'", "-version = 'absent'")
    else:
        patch = patch.replace("+version = 'changed'", "+version = (")
    path.write_text(patch)
    original = (tmp_path / "FEDOT/fedot/version.py").read_bytes()
    result = subprocess.run(
        [*args, "--proposals-from", str(seed), "--mode", "launch"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "invalid FEDOT proposal patch" in result.stderr
    assert not (tmp_path / "new-run").exists()
    assert (tmp_path / "FEDOT/fedot/version.py").read_bytes() == original
