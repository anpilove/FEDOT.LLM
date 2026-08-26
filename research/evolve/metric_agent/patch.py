from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from research.evolve.metric_agent.guard import deny_write
from research.evolve.metric_agent.types import PatchCandidate


def apply_patch(checkout: Path, candidate: PatchCandidate) -> bool:
    target = (checkout / candidate.file_path).resolve()
    reason = deny_write(target, checkout=checkout)
    if reason:
        raise PermissionError(reason)
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    hunks = candidate.hunks or ([(candidate.old_code, candidate.new_code)] if candidate.old_code else [])
    if not hunks:
        return False
    for _, new in hunks:
        if any(ln.strip() == "..." for ln in new.splitlines()):
            return False
    patched = text
    for old, new in hunks:
        if not old or len(old.strip()) < 4:
            return _replace_symbol(target, candidate) if len(hunks) == 1 else False
        hits = patched.count(old)
        if hits == 0:
            located = _match_ignoring_indent(patched, old)
            if located is None:
                return False
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
            return False
        patched = patched.replace(old, new, 1)
    try:
        ast.parse(patched)
    except SyntaxError:
        return False
    target.write_text(patched, encoding="utf-8")
    return True


def _replace_symbol(path: Path, candidate: PatchCandidate) -> bool:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    name = _def_name(candidate.new_code) or _def_name(candidate.old_code)
    if not name:
        return False
    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        and getattr(node, "end_lineno", None)
    ]
    if len(targets) != 1:
        return False
    node = targets[0]
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    body = textwrap.indent(textwrap.dedent(candidate.new_code.strip("\n")), indent).rstrip("\n") + "\n"
    patched = "".join(lines[:start]) + body + "".join(lines[node.end_lineno :])
    try:
        ast.parse(patched)
    except SyntaxError:
        return False
    path.write_text(patched, encoding="utf-8")
    return True


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
