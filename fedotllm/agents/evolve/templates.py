"""Deterministic failing tests for defects whose shape is known from the rule.

Measured across three configurations: the dominant failure is not a bad patch but
a bad proof. On the small model the test did not fail before the patch 138 times;
on the large one it failed for the wrong reason (a fixture file that never
existed). Either way the agent spends its budget inventing a demonstration it
cannot get right.

For a defect located by a lint rule the demonstration is not open-ended — the rule
*is* the specification. These templates build it from the source, so the test is
correct by construction: it fails on the unpatched tree, passes once the defect is
gone, and cannot fail for an unrelated reason because it never calls the function
or touches the filesystem.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from fedotllm.log import logger

# Rules whose demonstration this module can build.
SUPPORTED_RULES = ("B008", "B006", "RUF012")
TEMPLATE_CACHE_VERSION = 2
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

_IMMUTABLE_GUARD = "(type(None), str, int, float, bool, bytes, tuple, frozenset)"


@dataclass
class GeneratedTest:
    rule: str
    test_name: str
    test_code: str
    target: str  # what the test pins down, for the audit report


def module_path(file_rel: str) -> str:
    """`fedot/core/data/data.py` -> `fedot.core.data.data`."""
    return file_rel[:-3].replace("/", ".") if file_rel.endswith(".py") else file_rel


def _enclosing(tree: ast.AST, line: int) -> tuple[ast.ClassDef | None, ast.AST | None]:
    """Class and function that contain `line`, either of which may be absent."""
    cls = fn = None
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not getattr(node, "end_lineno", None):
            continue
        if not (node.lineno <= line <= node.end_lineno):
            continue
        if isinstance(node, ast.ClassDef):
            cls = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Innermost wins for nested definitions.
            if fn is None or node.lineno > fn.lineno:
                fn = node
    return cls, fn


def _contains_position(node: ast.AST, line: int, column: int = 0) -> bool:
    if not hasattr(node, "lineno") or not getattr(node, "end_lineno", None):
        return False
    if not (node.lineno <= line <= node.end_lineno):
        return False
    if not column:
        return True
    col = column - 1
    if line == node.lineno and col < node.col_offset:
        return False
    if line == node.end_lineno and col > getattr(node, "end_col_offset", col):
        return False
    return True


def _constructed_default_at(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    line: int,
    column: int = 0,
) -> str | None:
    """Parameter whose constructed default contains the ruff location."""
    args = fn.args
    positional = args.posonlyargs + args.args
    pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
    pairs += [
        (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
    ]
    constructed = [
        (a.arg, d)
        for a, d in pairs
        if isinstance(d, (ast.Call, ast.List, ast.Dict, ast.Set, ast.DictComp, ast.ListComp))
    ]
    located = [name for name, default in constructed if _contains_position(default, line, column)]
    return located[0] if located else None


def _mutable_class_attr_at(cls: ast.ClassDef, line: int, column: int = 0) -> str | None:
    """Mutable class attribute declared at the ruff location."""
    for stmt in cls.body:
        value = getattr(stmt, "value", None)
        if value is None or not isinstance(value, (ast.List, ast.Dict, ast.Set, ast.Call)):
            continue
        if not _contains_position(stmt, line, column):
            continue
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        elif isinstance(stmt, ast.Assign):
            return next((t.id for t in stmt.targets if isinstance(t, ast.Name)), None)
    return None


def _default_test(module: str, accessor: str, param: str, test_name: str) -> str:
    return f'''import inspect

from {module} import {accessor.split(".")[0]}


def {test_name}():
    """A default built by a call is created once at import and shared by every caller."""
    default = inspect.signature({accessor}).parameters["{param}"].default
    assert isinstance(default, {_IMMUTABLE_GUARD}), (
        f"parameter {param!r} of {accessor} has a constructed default "
        f"{{default!r}} ({{type(default).__name__}}); it is built once at import "
        f"and shared by every call"
    )
'''


def _class_attr_test(module: str, cls: str, attr: str, test_name: str) -> str:
    """Two assertions: the defect is gone, and the public API survived.

    The first one alone is satisfiable by wrapping the attribute in a
    ``classmethod`` — an agent tried exactly that. The value stops being a
    container, so the check passes, while every reader of the attribute breaks.
    Instantiating the class instead would be stronger still, but not every class
    takes zero arguments, and a test that can never pass is worse than a weak one.
    """
    return f'''from {module} import {cls}


def {test_name}():
    """A mutable class attribute is one object shared by every instance."""
    declared = vars({cls}).get("{attr}")
    assert not isinstance(declared, (list, dict, set)), (
        f"{cls}.{attr} is a mutable {{type(declared).__name__}} declared on the class; "
        f"every instance shares it, so a change through one is seen by all"
    )
    assert not isinstance(declared, (classmethod, staticmethod, property)), (
        f"{cls}.{attr} was turned into a {{type(declared).__name__}}; that removes the "
        f"shared object but also changes the public API every reader depends on"
    )
'''


def build_test(
    repo: Path,
    rule: str,
    file_rel: str,
    line: int,
    column: int = 0,
) -> GeneratedTest | None:
    """Build the failing test for one lint finding, or None when unsupported."""
    if rule not in SUPPORTED_RULES:
        return None
    path = repo / file_rel
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None

    module = module_path(file_rel)
    cls, fn = _enclosing(tree, line)

    if rule in ("B008", "B006") and fn is not None:
        param = _constructed_default_at(fn, line, column)
        if param is None:
            return None
        accessor = f"{cls.name}.{fn.name}" if cls else fn.name
        name = f"test_{fn.name}_default_{param}_is_not_constructed"
        return GeneratedTest(rule, name, _default_test(module, accessor, param, name),
                             f"{accessor}({param}=…)")

    if rule == "RUF012" and cls is not None:
        attr = _mutable_class_attr_at(cls, line, column)
        if attr is None:
            return None
        name = f"test_{cls.name}_{attr}_is_not_shared_mutable"
        return GeneratedTest(rule, name, _class_attr_test(module, cls.name, attr, name),
                             f"{cls.name}.{attr}")

    return None


def symbol_source(repo: Path, file_rel: str, accessor: str) -> str:
    """Exact source of the symbol a template targets, with line numbers.

    Without it the agent gets a 12 000-character file holding five near-identical
    methods and one sentence saying which is defective — and patches the wrong
    ones. Measured: four hunks, none of them in the function under test.
    """
    path = repo / file_rel
    if not path.is_file():
        return ""
    wanted = accessor.split("(")[0].split(".")[-1]
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except (SyntaxError, OSError):
        return ""
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == wanted
            and getattr(node, "end_lineno", None)
        ):
            # Raw lines, no line-number prefix: the agent has to reproduce this
            # indentation verbatim, and a `397 | ` gutter made it miscount the
            # leading spaces and quote a block that exists nowhere in the file.
            return "\n".join(text.splitlines()[node.lineno - 1 : node.end_lineno])
    return ""


def verify_fails_on_pristine(repo: Path, python: str, gen: GeneratedTest) -> bool:
    """A template is only worth offering if its test really fails right now.

    Five of forty templates target modules that need optional dependencies
    (H2O, TPOT, TabPFN); their tests error on import rather than fail, and an
    import error proves nothing.
    """
    tmp = repo / "test" / "unit" / "test_fedotllm_probe_tmpl.py"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(gen.test_code, encoding="utf-8")
        proc = subprocess.run(
            [python, "-m", "pytest", str(tmp.relative_to(repo)), "-q"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("template verification failed for %s: %s", gen.test_name, exc)
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return " failed" in proc.stdout


def collect_findings(repo: Path) -> list[str]:
    """Concise ruff lines for the rules this module can turn into tests."""
    args = [
        "check", "fedot/", f"--select={','.join(SUPPORTED_RULES)}",
        "--output-format=concise", "--no-fix", "--isolated",
    ]
    for cmd in (["ruff", *args], ["uvx", "ruff", *args]):
        try:
            proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)
        except (OSError, FileNotFoundError):
            continue
        if proc.stdout.strip():
            return [
                _ANSI_ESCAPE.sub("", line)
                for line in proc.stdout.splitlines()
                if line.strip()
            ]
    return []


def proven_defects(repo: Path, python: str, limit: int = 40) -> list[GeneratedTest]:
    """Defects that already have a failing test — no invention required.

    This inverts the usual order. The agent used to find a defect, invent a
    demonstration and write a patch, and it broke on the middle step every time.
    Here the demonstration exists first, so the only open task is the patch.
    """
    proven: list[GeneratedTest] = []
    seen: set[str] = set()
    for line in collect_findings(repo):
        if len(proven) >= limit:
            break
        gen = build_from_finding(repo, line)
        if gen is None or gen.test_name in seen:
            continue
        seen.add(gen.test_name)
        if verify_fails_on_pristine(repo, python, gen):
            proven.append(gen)
    logger.info("templates: %s proven defects (test already fails)", len(proven))
    return proven


def proven_defects_cached(repo: Path, python: str) -> list[GeneratedTest]:
    """`proven_defects` memoised on the working tree, like the runtime probe.

    Verification costs one pytest run per candidate (~2 minutes for forty); a
    benchmark resets the repo before every run, so the answer never changes.
    """
    from fedotllm.agents.evolve.probe import _repo_state  # same identity rule

    cache = Path(os.environ.get("FEDOTLLM_TEMPLATE_CACHE", "/tmp/fedotllm_templates.json"))
    state = _repo_state(repo)
    if cache.is_file():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
            if (
                blob.get("version") == TEMPLATE_CACHE_VERSION
                and blob.get("state") == state
            ):
                logger.info("templates: reusing %s cached proven defects", len(blob["items"]))
                return [GeneratedTest(**it) for it in blob["items"]]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            pass
    items = proven_defects(repo, python)
    try:
        cache.write_text(
            json.dumps({
                "version": TEMPLATE_CACHE_VERSION,
                "state": state,
                "items": [asdict(i) for i in items],
            }),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not write template cache: %s", exc)
    return items


def build_from_finding(repo: Path, finding: str) -> GeneratedTest | None:
    """Parse a concise ruff line (`path:line:col: CODE message`) and build its test."""
    try:
        location, _ = finding.split(": ", 1)
        file_rel, line_s, column_s = location.split(":")[:3]
        rule = finding.split(": ", 1)[1].split(" ", 1)[0]
    except (ValueError, IndexError):
        return None
    try:
        built = build_test(repo, rule, file_rel, int(line_s), int(column_s))
    except Exception as exc:  # a template must never break the round
        logger.warning("test template failed for %r: %s", finding[:60], exc)
        return None
    return built
