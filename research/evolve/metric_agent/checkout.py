from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from research.evolve.metric_agent.guard import repo_root


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


def make_disposable_checkout(dest: Path | None = None) -> Path:
    src = resolve_fedot_src()
    dest = dest or Path(os.environ.get("METRIC_AGENT_CHECKOUT", "/tmp/metric-agent-fedot"))
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
    return dest.resolve()


def revert_checkout(checkout: Path) -> None:
    git_dir = checkout / ".git"
    if git_dir.exists():
        subprocess.run(["git", "checkout", "--", "."], cwd=checkout, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=checkout, check=True)
        return
    fresh = make_disposable_checkout(checkout)
    if fresh != checkout.resolve():
        raise RuntimeError("checkout path changed on revert")


def snapshot_diff(checkout: Path, rel: str) -> str:
    import difflib

    src = resolve_fedot_src() / rel
    dst = checkout / rel
    if not dst.is_file():
        return ""
    old = src.read_text(encoding="utf-8", errors="replace").splitlines() if src.is_file() else []
    new = dst.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(
        difflib.unified_diff(old, new, fromfile=f"stock/{rel}", tofile=f"patched/{rel}", lineterm="")
    )
