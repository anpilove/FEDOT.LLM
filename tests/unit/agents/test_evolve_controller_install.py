"""Installation publishes code separately from runs without launching ML."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile

from fedotllm.agents.evolve.execution.guard import repo_root


def fixture_release(tmp_path, *, unsafe=False):
    content = b'"""controller fixture"""\n'
    name = "fedotllm/agents/evolve/controller/__init__.py"
    digest = hashlib.sha256()
    digest.update(b"agents/evolve/controller/__init__.py")
    digest.update(content)
    version = digest.hexdigest()
    manifest = {
        "name": "FEDOT.LLM Evolve Controller",
        "controller_sha256": version,
        "files": {name: hashlib.sha256(content).hexdigest()},
    }
    archive = tmp_path / "controller.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path, data in (
            (name, content),
            ("release.json", json.dumps(manifest).encode()),
        ):
            item = tarfile.TarInfo(path)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
        if unsafe:
            item = tarfile.TarInfo("../escaped")
            item.size = 1
            tar.addfile(item, io.BytesIO(b"x"))
    return archive, version


def install_command(tmp_path, archive, version):
    return [
        sys.executable,
        str(repo_root() / "fedotllm/agents/evolve/controller/install.py"),
        "--root",
        str(tmp_path / "FEDOT.LLM"),
        "--archive",
        str(archive),
        "--archive-sha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "--controller-sha256",
        version,
    ]


def test_default_install_checks_release_without_creating_directories(tmp_path):
    archive, version = fixture_release(tmp_path)
    result = subprocess.run(
        install_command(tmp_path, archive, version),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"] == "checked"
    assert not (tmp_path / "FEDOT.LLM").exists()


def test_install_separates_controller_releases_and_runs_and_preserves_old_files(
    tmp_path,
):
    archive, version = fixture_release(tmp_path)
    old = tmp_path / "old-quality-run"
    old.mkdir()
    (old / "quality_jobs.jsonl").write_text("old results\n")
    result = subprocess.run(
        [*install_command(tmp_path, archive, version), "--mode", "install"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    evolve = tmp_path / "FEDOT.LLM/evolve"
    assert (
        evolve / "controller/fedotllm/agents/evolve/controller/__init__.py"
    ).is_file()
    assert (evolve / "runs").is_dir() and not list((evolve / "runs").iterdir())
    assert (evolve / "releases" / version / "installation.json").is_file()
    assert (old / "quality_jobs.jsonl").read_text() == "old results\n"


def test_invalid_archive_never_creates_a_deployment(tmp_path):
    archive, version = fixture_release(tmp_path, unsafe=True)
    result = subprocess.run(
        [*install_command(tmp_path, archive, version), "--mode", "install"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "unsafe archive" in result.stderr
    assert not (tmp_path / "FEDOT.LLM").exists()
    assert not (tmp_path / "escaped").exists()
