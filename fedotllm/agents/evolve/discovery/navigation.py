"""Shared bounded source navigation for production and blind benchmarks."""

from __future__ import annotations

from pathlib import Path

from fedotllm.agents.evolve.discovery.fedot_context import build_fedot_context


def architecture_contexts(
    checkout: Path,
    paths: tuple[str, ...],
    *,
    max_chars_per_file: int = 1_500,
) -> str:
    rows: list[str] = []
    for index, file_path in enumerate(paths):
        source_context = build_fedot_context(checkout, file_path)
        rendered = (
            source_context.render(max_chars=max_chars_per_file)
            if source_context is not None
            else f"file={file_path}"
        )
        rows.append(f"[{index}]\n{rendered}")
    return "\n\n".join(rows)


def inspect_shortlist(
    checkout: Path,
    paths: tuple[str, ...],
    indices: list[int],
    *,
    max_chars_per_file: int = 14_000,
) -> str:
    """Read at most three selected source files, preserving catalog indices."""

    rows: list[str] = []
    for index in list(dict.fromkeys(indices))[:3]:
        if not 0 <= index < len(paths):
            continue
        target = checkout / paths[index]
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        rows.append(f"[{index}] --- {paths[index]} ---\n{text[:max_chars_per_file]}")
    return "\n\n".join(rows)
