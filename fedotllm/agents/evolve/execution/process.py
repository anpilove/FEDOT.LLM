"""Shared subprocess policy for every EvolveAgent execution boundary.

The generated candidate is trusted to run local FEDOT code, but it must not
inherit API tokens or accidentally use a different Python than ``doctor``.
This module is deliberately small so evaluator, pytest, import and snippet
workers all share exactly the same interpreter and environment policy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


_SAFE_ENV_NAMES = {
    "HOME",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "MPLBACKEND",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "SYSTEMROOT",
    "WINDIR",
}

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def thread_environment() -> dict[str, str]:
    """Bound small evaluator workloads; preserve explicit user overrides."""
    return {name: os.environ.get(name, "1") for name in _THREAD_ENV_NAMES}


def fedot_python(checkout: Path) -> str:
    """Return the one interpreter used by every FEDOT subprocess."""

    configured = os.environ.get("FEDOTLLM_REPO_PYTHON")
    if configured:
        path = Path(os.path.abspath(Path(configured).expanduser()))
        if not path.is_file():
            raise FileNotFoundError(f"FEDOTLLM_REPO_PYTHON does not exist: {path}")
        return str(path)
    for candidate in (
        checkout / ".venv" / "bin" / "python",
        checkout.parent / ".venv-fedot" / "bin" / "python",
    ):
        if candidate.is_file():
            # Resolving a venv symlink loses pyvenv.cfg and installed packages.
            return str(Path(os.path.abspath(candidate)))
    if Path(sys.executable).is_file():
        return os.path.abspath(sys.executable)
    discovered = shutil.which("python3") or shutil.which("python")
    if discovered:
        return os.path.abspath(discovered)
    raise FileNotFoundError("no Python interpreter available for FEDOT subprocesses")


@lru_cache(maxsize=8)
def _interpreter_identity(executable: str) -> str:
    """Fingerprint the selected interpreter, not merely the controller Python."""

    command = (
        "import json, platform, sys; "
        "print(json.dumps({'executable': sys.executable, "
        "'implementation': platform.python_implementation(), "
        "'version': platform.python_version()}))"
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES or name.startswith("LC_")
    }
    env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [executable, "-c", command],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        path = Path(executable)
        try:
            stat = path.stat()
            return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return executable


def interpreter_identity(checkout: Path) -> str:
    return _interpreter_identity(fedot_python(checkout))


def clean_subprocess_env(
    checkout: Path,
    *,
    repo_root: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal environment without credentials.

    ``repo_root`` is appended only when a worker is launched with ``python -m``;
    the checkout remains first so ``import fedot`` always resolves to the
    experiment rather than an installed package.
    """

    env = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES or name.startswith("LC_")
    }
    python_paths = [str(checkout.resolve())]
    if repo_root is not None:
        python_paths.append(str(repo_root.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(thread_environment())
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env
