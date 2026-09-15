from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import deny_write
from fedotllm.agents.evolve.types import PatchCandidate, PatchEdit

_GUTTER = re.compile(r"^\s*\d+\|")

def strip_gutter(text: str) -> str:
    """Drop `  123|` prefixes if the model copied numbered context."""

    lines = (text or "").splitlines()
    nonempty = [line for line in lines if line.strip()]
    if nonempty and all(_GUTTER.match(line) for line in nonempty):
        return "\n".join(_GUTTER.sub("", line) for line in lines)
    return text or ""


def same_runtime(old: str, new: str) -> bool:
    """True when SEARCH/REPLACE would not change runtime behavior."""

    return _norm_code(old) == _norm_code(new)


def _norm_code(text: str) -> str:
    lines: list[str] = []
    for line in strip_gutter(text).splitlines():
        code = line.split("#", 1)[0].rstrip()
        if code.strip():
            lines.append(" ".join(code.split()))
    return "\n".join(lines)


def hunk_list(candidate: PatchCandidate) -> list[tuple[str, str]]:
    if candidate.hunks:
        return [(strip_gutter(old), strip_gutter(new)) for old, new in candidate.hunks]
    if candidate.old_code:
        return [(strip_gutter(candidate.old_code), strip_gutter(candidate.new_code))]
    return []


def edit_list(candidate: PatchCandidate) -> list[PatchEdit]:
    if candidate.edits:
        return [
            PatchEdit(edit.file_path, strip_gutter(edit.old_code), strip_gutter(edit.new_code))
            for edit in candidate.edits
        ]
    return [
        PatchEdit(candidate.file_path, old, new)
        for old, new in hunk_list(candidate)
    ]


def apply_patch(
    checkout: Path,
    candidate: PatchCandidate,
    *,
    diagnostics: list[str] | None = None,
) -> bool:
    """Validate every edit in memory, then write all touched files atomically."""

    checkout = checkout.resolve()

    def reject(reason: str) -> bool:
        if diagnostics is not None:
            diagnostics.append(reason)
        return False

    edits = edit_list(candidate)
    if not edits:
        return reject("candidate has no edits")
    grouped: dict[Path, list[PatchEdit]] = {}
    seen_searches: dict[tuple[Path, str], int] = {}
    for index, edit in enumerate(edits, start=1):
        target = (checkout / edit.file_path).resolve()
        reason = deny_write(target, checkout=checkout)
        if reason:
            raise PermissionError(reason)
        try:
            relative = target.relative_to(checkout).as_posix()
        except ValueError as exc:
            raise PermissionError(f"patch target escaped checkout: {target}") from exc
        if not relative.startswith("fedot/"):
            raise PermissionError(
                f"source edits are restricted to fedot/**; proposed test changes "
                f"must remain review-only artifacts: {relative}"
            )
        if not target.is_file():
            return reject(f"edit {index}: target is not a file: {edit.file_path}")
        if not edit.old_code or same_runtime(edit.old_code, edit.new_code):
            return reject(f"edit {index}: SEARCH is empty or replacement is a no-op")
        if any(line.strip() == "..." for line in edit.new_code.splitlines()):
            return reject(f"edit {index}: replacement contains an ellipsis placeholder")
        duplicate = seen_searches.get((target, edit.old_code))
        if duplicate is not None:
            return reject(
                f"edits {duplicate} and {index}: duplicate SEARCH for {edit.file_path}; remove or combine it"
            )
        seen_searches[(target, edit.old_code)] = index
        grouped.setdefault(target, []).append(edit)

    prepared: dict[Path, str] = {}
    for target, file_edits in grouped.items():
        patched = target.read_text(encoding="utf-8")
        for index, edit in enumerate(file_edits, start=1):
            old, new = edit.old_code, edit.new_code
            if len(old.strip()) < 4:
                if len(file_edits) != 1:
                    return reject(
                        f"{target.relative_to(checkout)} edit {index}: SEARCH is too short for a multi-edit file"
                    )
                replaced = _replace_symbol_text(patched, old, new)
                if replaced is None:
                    return reject(
                        f"{target.relative_to(checkout)} edit {index}: symbol replacement is missing or ambiguous"
                    )
                patched = replaced
                continue
            hits = patched.count(old)
            if hits == 0:
                located = _match_ignoring_indent(patched, old)
                if located is None:
                    return reject(
                        f"{target.relative_to(checkout)} edit {index}: SEARCH was not found uniquely"
                    )
                start, end, file_indent = located
                model_indent = old.strip("\n").splitlines()[0]
                model_indent = model_indent[: len(model_indent) - len(model_indent.lstrip())]
                body = _reindent(new.strip("\n"), model_indent, file_indent)
                lines = patched.splitlines()
                patched = "\n".join(lines[:start] + body.splitlines() + lines[end:])
                if patched and not patched.endswith("\n"):
                    patched += "\n"
                continue
            if hits != 1:
                return reject(
                    f"{target.relative_to(checkout)} edit {index}: SEARCH occurs {hits} times; include neighboring lines"
                )
            patched = patched.replace(old, new, 1)
        if target.suffix == ".py":
            try:
                ast.parse(patched)
            except SyntaxError as exc:
                return reject(
                    f"{target.relative_to(checkout)}: replacement is invalid Python at line {exc.lineno}: {exc.msg}"
                )
        elif target.suffix == ".json":
            try:
                json.loads(patched)
            except json.JSONDecodeError as exc:
                return reject(
                    f"{target.relative_to(checkout)}: replacement is invalid JSON at line {exc.lineno}: {exc.msg}"
                )
        else:
            return reject(
                f"{target.relative_to(checkout)}: only FEDOT .py and .json files are patchable"
            )
        prepared[target] = patched
    originals = {target: target.read_text(encoding="utf-8") for target in prepared}
    written: list[Path] = []
    try:
        for target, patched in prepared.items():
            target.write_text(patched, encoding="utf-8")
            written.append(target)
    except OSError:
        for target in written:
            target.write_text(originals[target], encoding="utf-8")
        raise
    return True


def _replace_symbol_text(text: str, old_code: str, new_code: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    name = _def_name(new_code) or _def_name(old_code)
    if not name:
        return None
    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        and getattr(node, "end_lineno", None)
    ]
    if len(targets) != 1:
        return None
    node = targets[0]
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    body = textwrap.indent(textwrap.dedent(new_code.strip("\n")), indent).rstrip("\n") + "\n"
    patched = "".join(lines[:start]) + body + "".join(lines[node.end_lineno :])
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return patched


def _def_name(code: str) -> str | None:
    match = re.search(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", code, re.MULTILINE)
    return match.group(1) if match else None


def _match_ignoring_indent(text: str, old: str) -> tuple[int, int, str] | None:
    wanted = [ln.strip() for ln in old.strip("\n").splitlines() if ln.strip()]
    if not wanted:
        return None
    lines = text.splitlines()
    hits: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        if line.strip() != wanted[0]:
            continue
        k, j = 1, i + 1
        while k < len(wanted) and j < len(lines):
            if not lines[j].strip():
                j += 1
                continue
            if lines[j].strip() != wanted[k]:
                break
            k += 1
            j += 1
        if k != len(wanted):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        hits.append((i, j, indent))
    return hits[0] if len(hits) == 1 else None


def _reindent(block: str, old_indent: str, new_indent: str) -> str:
    if old_indent == new_indent:
        return block
    out = []
    for ln in block.splitlines():
        if ln.startswith(old_indent):
            out.append(new_indent + ln[len(old_indent) :])
        else:
            out.append(ln)
    return "\n".join(out)
