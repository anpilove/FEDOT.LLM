from __future__ import annotations

import os
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import repo_root

MARKER = ".evolve-agent-checkout.json"


def resolve_fedot_src() -> Path:
    local = os.environ.get("FEDOTLLM_REPO_PATH")
    if local and Path(local).is_dir():
        return Path(local).resolve()
    cache = Path(os.environ.get("FEDOTLLM_REPO_CACHE", str(repo_root() / ".repo_cache" / "FEDOT")))
    if cache.is_dir():
        return cache.resolve()
    raise FileNotFoundError(
        "Set FEDOTLLM_REPO_PATH to a FEDOT checkout (stock 0.7.5)."
    )


def source_fingerprint(source: Path) -> str:
    """Hash runtime source and reproducibility metadata, not path identity."""

    source = source.resolve()
    digest = hashlib.sha256()
    roots = [source / "fedot"]
    extras = [source / name for name in ("pyproject.toml", "setup.py", "requirements.txt")]
    files = sorted(
        path
        for root in roots
        if root.is_dir()
        for pattern in ("*.py", "*.json")
        for path in root.rglob(pattern)
    )
    files.extend(path for path in extras if path.is_file())
    if not files:
        digest.update(str(source).encode())
    for path in files:
        rel = path.relative_to(source).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_commit(source: Path) -> str:
    if not (source / ".git").exists():
        return ""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def create_experiment_checkout(
    source: Path,
    workspace: Path,
    *,
    run_id: str,
    candidate_id: str,
) -> Path:
    """Create one owned checkout. The source is never modified or removed."""

    source = source.resolve()
    experiments = (workspace.resolve() / "experiments" / run_id).resolve()
    dest = (experiments / candidate_id).resolve()
    if dest == source or dest in source.parents or source in dest.parents:
        raise ValueError("experiment and immutable source must be disjoint")
    if experiments not in dest.parents:
        raise ValueError("experiment escaped workspace")
    if dest.exists():
        discard_experiment_checkout(dest, workspace=workspace, source=source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", ".pytest_cache"),
    )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "source": str(source),
        "source_commit": source_commit(source),
        "source_hash": source_fingerprint(source),
    }
    (dest / MARKER).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def discard_experiment_checkout(
    checkout: Path,
    *,
    workspace: Path,
    source: Path | None = None,
) -> None:
    checkout = checkout.resolve()
    workspace = workspace.resolve()
    experiments = (workspace / "experiments").resolve()
    if checkout == experiments or experiments not in checkout.parents:
        raise PermissionError(f"refuse to remove non-experiment path: {checkout}")
    if source is not None:
        source = source.resolve()
        if checkout == source or checkout in source.parents or source in checkout.parents:
            raise PermissionError("refuse to remove immutable source or its parent")
    marker = checkout / MARKER
    if not checkout.exists():
        return
    if not marker.is_file():
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError(f"invalid EvolveAgent ownership marker: {checkout}") from exc
    if payload.get("source") == str(checkout):
        raise PermissionError("ownership marker points at checkout as source")
    if payload.get("candidate_id") != checkout.name or payload.get("run_id") != checkout.parent.name:
        raise PermissionError("ownership marker does not match experiment path")
    if source is not None and payload.get("source") != str(source):
        raise PermissionError("ownership marker does not match immutable source")
    if not payload.get("source_hash"):
        raise PermissionError("ownership marker has no source fingerprint")
    shutil.rmtree(checkout)


def snapshot_diff(checkout: Path, rel: str, *, source: Path | None = None) -> str:
    import difflib

    src = (source.resolve() if source is not None else resolve_fedot_src()) / rel
    dst = checkout / rel
    if not dst.is_file():
        return ""
    old = src.read_text(encoding="utf-8", errors="replace").splitlines() if src.is_file() else []
    new = dst.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, fromfile=f"stock/{rel}", tofile=f"patched/{rel}", lineterm="")
    )
