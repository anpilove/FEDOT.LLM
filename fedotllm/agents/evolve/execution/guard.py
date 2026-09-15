from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

DENIED_PREFIXES = (
    "research/evolve/evolve_agent/",
    "research/evolve/",
    "data/",
    "fedotllm/",
    "_local_fedot_patches/",
)


def repo_root() -> Path:
    return _REPO_ROOT


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def deny_write(path: Path, *, checkout: Path | None = None) -> str | None:
    """Return a reason if this path must not be written; None if allowed.

    A disposable FEDOT checkout is the only writable tree when ``checkout`` is
    set. Product-prefix denials apply to the FEDOT.LLM repo itself, not to a
    copied library that happens to live under ``research/`` or contains
    ``evaluation`` in a FEDOT source path.
    """

    resolved = _resolve(Path(path))
    if checkout is not None:
        check = checkout.resolve()
        if resolved == check or check in resolved.parents:
            return None
        return f"deny path outside FEDOT checkout: {resolved}"
    root = _REPO_ROOT.resolve()
    try:
        rel = resolved.relative_to(root).as_posix()
    except ValueError:
        rel = None
    if rel is not None:
        for prefix in DENIED_PREFIXES:
            if rel == prefix.rstrip("/") or rel.startswith(prefix):
                return f"deny evaluator/product path: {rel}"
    return "deny path outside FEDOT checkout"


def guard_path(path: str | Path, *, checkout: Path | None = None) -> str:
    reason = deny_write(Path(path), checkout=checkout)
    return "deny" if reason else "allow"
