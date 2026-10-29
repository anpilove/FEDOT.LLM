"""Controller release identity; no ML or provider imports."""

import hashlib
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import repo_root


def controller_version(controller_root: Path | None = None):
    digest = hashlib.sha256()
    root = (controller_root if controller_root is not None else repo_root()) / "fedotllm"
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
