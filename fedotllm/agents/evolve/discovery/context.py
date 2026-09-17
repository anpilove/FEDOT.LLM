from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

from fedotllm.agents.evolve.execution.guard import deny_write
from fedotllm.agents.evolve.types import MatchSite

_FILE = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?$', re.MULTILINE)
_FALLBACK_RADIUS = 12
# A FEDOT operation method often keeps its fit/transform contract, parameter
# normalization and metadata updates in one 100-200 line body.  Clipping such a
# method to a local window made the research model reason from half a lifecycle.
# Keep the complete executable symbol when it is still comfortably bounded;
# larger generated/dispatcher bodies retain the local-window fallback.
_MAX_FUNC_LINES = 240
_MAX_FRAMES = 3
_MAX_FILE_CHARS = 80_000
_WHOLE_FILE_LINES = 400


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= max(0, limit):
        return text
    return text[: max(0, limit)].rsplit("\n", 1)[0] + "\n...[truncated]"


def _compact_lead_evidence(lead: MatchSite, *, max_chars: int = 3_000) -> str:
    """Preserve causal evidence without letting repeated workloads hide source.

    Coverage contributes one runtime/operation row per workload.  Dumping all
    effective parameter dictionaries into a small Verifier/Fixer prompt used to
    consume the budget before the marked method was shown.  The full rows remain
    in trace.jsonl and are retrievable through tools; the initial context needs
    the executed lines, affected workload families, and concrete implementations.
    """

    direct: list[str] = []
    workloads: list[str] = []
    implementations: list[str] = []
    for raw in lead.evidence:
        item = str(raw or "").strip()
        if not item:
            continue
        if item.startswith("runtime operation instances:"):
            for instance in item.partition(":")[2].split("|"):
                head = instance.strip().split(" ", 1)[0]
                if head and head not in implementations:
                    implementations.append(head)
            continue
        if item.startswith("workload family="):
            if item not in workloads:
                workloads.append(item)
            continue
        direct.append(item)

    rows = list(dict.fromkeys(direct))
    if implementations:
        rows.append(
            "runtime implementations observed: "
            + ", ".join(implementations[:16])
        )
    if workloads:
        rows.append("affected frozen workloads:")
        rows.extend(f"  {item}" for item in workloads[:12])
    return _clip("\n".join(f"- {row}" for row in rows), max_chars)


def show_source(
    path: str | Path,
    *,
    checkout: Path,
    around: int = 1,
    radius: int = _FALLBACK_RADIUS,
) -> str:
    """Enclosing function/class if AST allows, else a small window around the line.

    Not ±40 of the file: that pulled in unrelated docstrings (Agentless is
    file → function → lines).
    """

    checkout = checkout.resolve()
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (checkout / raw).resolve()
    if deny_write(target, checkout=checkout):
        return ""
    if not target.is_file():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = _window(lines, around, radius=radius)
    numbered = [f"{i + 1:5d}|{lines[i]}" for i in range(start, end)]
    rel = target.relative_to(checkout).as_posix()
    return f"# {rel}\n" + "\n".join(numbered)


def scout_source_context(
    lead: MatchSite,
    *,
    checkout: Path,
    max_chars: int = 12_000,
) -> str:
    """Semantic source bundle for discovery, independent of benchmark scores.

    The marked method is not a self-contained unit: its state is usually
    created in ``__init__``/``fit`` and consumed in ``transform``/``predict``.
    Include bounded same-class and inherited behavior automatically instead of
    relying on a cheap model to discover that lifecycle through extra turns.
    """

    parts = [
        show_source(
            lead.file_path,
            checkout=checkout,
            around=max(1, lead.line),
            radius=20,
        )
    ]
    siblings = _runtime_siblings(checkout, lead, limit=4)
    if siblings:
        parts.append("Related methods from the same class:\n" + siblings)
    bases = _base_class_sources(checkout, lead, limit=4)
    if bases:
        parts.append("Inherited runtime contract:\n" + bases)
    return _clip("\n\n".join(filter(None, parts)), max_chars)


def open_runtime(checkout: Path, rel: str, *, line: int = 1) -> str:
    """Whole file, or the enclosing function if line > 1. Empty outside fedot/ runtime."""

    from fedotllm.agents.evolve.discovery.repo_map import in_metric_scan

    norm = (rel or "").lstrip("/")
    if "fedot/" in norm and not norm.startswith("fedot/"):
        norm = norm[norm.index("fedot/") :]
    if not in_metric_scan(norm):
        return ""
    if int(line or 1) > 1:
        return show_source(norm, checkout=checkout, around=int(line))
    return show_file(norm, checkout=checkout)


def show_file(
    path: str | Path,
    *,
    checkout: Path,
    max_chars: int = _MAX_FILE_CHARS,
) -> str:
    """Whole library file with line numbers. Empty if outside the FEDOT checkout."""

    checkout = checkout.resolve()
    raw = Path(path)
    target = raw.resolve() if raw.is_absolute() else (checkout / raw).resolve()
    if deny_write(target, checkout=checkout) or not target.is_file():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    numbered = [f"{i + 1:5d}|{lines[i]}" for i in range(len(lines))]
    rel = target.relative_to(checkout).as_posix()
    text = f"# {rel}\n" + "\n".join(numbered)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return text


def inspect_trace(
    traceback: str,
    *,
    checkout: Path,
    radius: int = _FALLBACK_RADIUS,
) -> list[dict]:
    """Frames inside the FEDOT checkout only. Site-packages and scorer paths dropped."""

    checkout = checkout.resolve()
    frames: list[dict] = []
    for match in _FILE.finditer(traceback or ""):
        raw, line, func = match.group(1), int(match.group(2)), match.group(3) or ""
        try:
            raw_path = Path(raw)
            resolved = raw_path.resolve() if raw_path.is_absolute() else (checkout / raw_path).resolve()
            rel = resolved.relative_to(checkout).as_posix()
        except (OSError, ValueError):
            continue
        if deny_write(resolved, checkout=checkout):
            continue
        frames.append(
            {
                "file": rel,
                "line": line,
                "func": func,
                "source": show_source(resolved, checkout=checkout, around=line, radius=radius),
            }
        )
    return frames


def context_from_lead(lead: MatchSite, checkout: Path, *, max_chars: int = _MAX_FILE_CHARS) -> str:
    target = checkout / lead.file_path
    n_lines = 0
    if target.is_file():
        n_lines = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
    whole = n_lines <= _WHOLE_FILE_LINES
    source = show_file(lead.file_path, checkout=checkout) if whole else show_source(
        lead.file_path, checkout=checkout, around=lead.line
    )
    # On bounded prompts, an otherwise "small" 300-line file can still consume
    # the entire budget before the marked line.  Prefer the enclosing executable
    # symbol when the whole numbered file would crowd out the lead and contract.
    focused_for_budget = bool(
        whole
        and lead.line > 1
        and len(source) > max(4_000, int(max_chars * 0.55))
    )
    if focused_for_budget:
        source = show_source(lead.file_path, checkout=checkout, around=lead.line)
    parts = [
        f"Site: {lead.file_path}:{lead.line}",
        (
            "Read this whole file. Patch anywhere that can change runtime quality — not only the marked line."
            if whole and not focused_for_budget
            else "File is long; this is the enclosing method. Patch here or the real cause next to it."
        ),
    ]
    if lead.why:
        parts.append(f"Research hypothesis (not evidence): {_clip(lead.why, 2_500)}")
    if lead.mechanism or lead.proposed_change or lead.expected_metric_effect:
        parts.append(
            "Scout causal contract (proposal, not evidence):\n"
            f"Mechanism: {_clip(lead.mechanism, 2_000)}\n"
            f"Proposed source change: {_clip(lead.proposed_change, 2_000)}\n"
            f"Expected metric effect: {_clip(lead.expected_metric_effect, 2_000)}"
        )
    if lead.evidence:
        parts.append("Measured evidence:\n" + _compact_lead_evidence(lead))
    contract_fields = _fields_in_evidence(lead)
    if contract_fields:
        parts.append(
            "Data-contract fields named by runtime evidence: "
            + ", ".join(contract_fields)
            + ". Trace the class that owns each field and the producer/constructor "
            "that copies it before patching; do not assume a direct Data field lives "
            "inside supplementary_data."
        )
    if source:
        parts.append("Exact source at the measured site:\n" + source)
    if not whole or focused_for_budget:
        siblings = _runtime_siblings(checkout, lead)
        if siblings:
            parts.append("Same class:\n" + siblings)
        bases = _base_class_sources(checkout, lead)
        if bases:
            parts.append("Inherited runtime behavior:\n" + bases)
        related = _related_implementation_sources(checkout, lead)
        if related:
            parts.append(
                "Related implementations sharing a base class (protect their behavior):\n"
                + related
            )
        if "_convert_to_output(" in f"{source}\n{bases}":
            conversion = _named_runtime_sources(
                checkout,
                ("_convert_to_output", "_convert_to_output_function"),
                limit=4,
            )
            if conversion:
                parts.append("Output conversion contract:\n" + conversion)
    callees = _callee_sources(checkout, lead, skip_file=lead.file_path if whole else None)
    if callees:
        parts.append("Called from this function:\n" + callees)
    from fedotllm.agents.evolve.discovery.repo_map import (
        format_map,
        search_callers,
        search_field_usage,
    )

    func = lead.why.rsplit(" in ", 1)[-1].strip() if " in " in lead.why else ""
    if "." in func:
        func = func.rsplit(".", 1)[-1]
    if not func:
        token = (lead.why or "").strip().split()[-1] if (lead.why or "").strip() else ""
        func = token.rsplit(".", 1)[-1] if token else ""
    if func:
        callers = search_callers(checkout, func, limit=6)
        if callers:
            parts.append("Callers:\n" + format_map(callers))
    fields = list(dict.fromkeys([*_attrs_in_source(source), *contract_fields]))[:8]
    for field in fields:
        usages = search_field_usage(checkout, field, limit=8)
        if usages:
            parts.append(f"Field {field}:\n" + format_map(usages))
    body = "\n\n".join(parts)
    # Source alone says how code works, not what FEDOT contract it implements.
    # Reserve part of the budget for public docs/docstrings/operation metadata.
    from fedotllm.agents.evolve.discovery.knowledge import knowledge_for_lead

    knowledge = knowledge_for_lead(
        checkout,
        lead,
        max_chars=min(4_000, max(0, max_chars // 3)),
    )
    if not knowledge:
        return body[:max_chars]
    suffix = f"\n\n{knowledge}"
    return body[: max(0, max_chars - len(suffix))] + suffix[:max_chars]


def _named_runtime_sources(
    checkout: Path,
    names: tuple[str, ...],
    *,
    limit: int = 4,
) -> str:
    """Show concrete helper bodies referenced by inherited runtime methods."""

    from fedotllm.agents.evolve.discovery.repo_map import iter_symbols, skip_metric_noise

    wanted = set(names)
    by_name: dict[str, list] = {name: [] for name in names}
    for symbol in iter_symbols(checkout):
        if (
            symbol.name not in wanted
            or symbol.kind not in {"method", "function"}
            or skip_metric_noise(symbol.file_path)
        ):
            continue
        by_name[symbol.name].append(symbol)
    # Retrieve at least one body for every requested dependency name.  The old
    # global limit could be exhausted by four same-named wrapper methods before
    # reaching the concrete helper they delegated to.
    ordered = []
    for name in names:
        candidates = sorted(
            by_name.get(name, ()),
            key=lambda item: (
                0 if item.kind == "function" else 1,
                0 if "implementation_interfaces.py" in item.file_path else 1,
                item.file_path,
                item.line,
            ),
        )
        if candidates:
            ordered.append(candidates[0])
    for name in names:
        for symbol in by_name.get(name, ()):
            if symbol not in ordered:
                ordered.append(symbol)
    parts: list[str] = []
    seen: set[tuple[str, int]] = set()
    for symbol in ordered:
        key = (symbol.file_path, symbol.line)
        if key in seen:
            continue
        seen.add(key)
        snippet = show_source(symbol.file_path, checkout=checkout, around=symbol.line)
        if snippet:
            parts.append(snippet)
        if len(parts) >= max(len(names), max(1, limit)):
            break
    return "\n\n".join(parts)


_SKIP_ATTRS = frozenset(
    {
        "shape",
        "dtype",
        "size",
        "ndim",
        "T",
        "name",
        "value",
        "real",
        "imag",
        "log",
        "logger",
        "cache",
        "nodes",
        "content",
        "parameters",
        "metadata",
        "tags",
        "parent",
        "copy",
        "update",
        "append",
    }
)


def _attrs_in_source(source: str, *, limit: int = 4) -> list[str]:
    body = "\n".join(line.split("|", 1)[-1] for line in (source or "").splitlines() if "|" in line)
    body = textwrap.dedent(body)
    if not body.strip():
        return []
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    seen: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = node.attr
        if not name or name.startswith("_") or name in _SKIP_ATTRS or name in seen:
            continue
        seen.append(name)
        if len(seen) >= limit:
            break
    return seen


def _fields_in_evidence(lead: MatchSite, *, limit: int = 6) -> list[str]:
    """Extract concrete data-contract field names from measured runtime rows."""

    joined = "\n".join(str(item or "") for item in lead.evidence)
    candidates = re.findall(
        r"\b(?:[A-Za-z][A-Za-z0-9_]*_(?:idx|indices|names)|categorical_features)\b",
        joined,
    )
    return list(dict.fromkeys(candidates))[: max(1, limit)]


def _runtime_siblings(checkout: Path, lead: MatchSite, *, limit: int = 3) -> str:
    """Other methods on the same class — not only fit/transform/predict."""

    from fedotllm.agents.evolve.discovery.repo_map import _NOISE_IN_NAME

    target = checkout / lead.file_path
    if deny_write(target, checkout=checkout) or not target.is_file():
        return ""
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    owner: ast.ClassDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= lead.line <= end:
            owner = node
            break
    if owner is None:
        return ""
    parts: list[str] = []
    for item in owner.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(bit in item.name.lower() for bit in _NOISE_IN_NAME):
            continue
        end = getattr(item, "end_lineno", item.lineno) or item.lineno
        if item.lineno <= lead.line <= end:
            continue
        snippet = show_source(target, checkout=checkout, around=item.lineno)
        if snippet:
            parts.append(snippet)
        if len(parts) >= limit:
            break
    return "\n\n".join(parts)


def _base_class_sources(checkout: Path, lead: MatchSite, *, limit: int = 6) -> str:
    """Runtime methods inherited by the class containing the lead.

    A long FEDOT file is normally sliced to one enclosing method.  For thin
    operation subclasses (for example PCAImplementation) the quality-changing
    fit/transform data flow lives in a base class and was previously invisible
    to the fixer.
    """

    from fedotllm.agents.evolve.discovery.repo_map import iter_symbols

    target = checkout / lead.file_path
    if deny_write(target, checkout=checkout) or not target.is_file():
        return ""
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    owner: ast.ClassDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        if node.lineno <= lead.line <= end:
            owner = node
            break
    if owner is None:
        return ""

    base_names: list[str] = []
    for base in owner.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
    if not base_names:
        return ""

    class_symbols = {
        symbol.name: symbol
        for symbol in iter_symbols(checkout)
        if symbol.kind == "class" and symbol.name in base_names
    }
    priorities = {
        "fit": 0,
        "transform": 1,
        "predict": 2,
        "predict_for_fit": 3,
        "preprocess_input": 4,
        "postprocess_input": 5,
        "update_column_types": 6,
        "__init__": 7,
    }
    parts: list[str] = []
    for base_name in base_names:
        symbol = class_symbols.get(base_name)
        if symbol is None:
            continue
        base_path = checkout / symbol.file_path
        try:
            base_tree = ast.parse(base_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        class_node = next(
            (
                node
                for node in base_tree.body
                if isinstance(node, ast.ClassDef) and node.name == base_name
            ),
            None,
        )
        if class_node is None:
            continue
        methods = [
            item
            for item in class_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        methods.sort(key=lambda item: (priorities.get(item.name, 100), item.lineno))
        snippets = [
            show_source(base_path, checkout=checkout, around=item.lineno)
            for item in methods[: max(1, limit - len(parts))]
        ]
        snippets = [snippet for snippet in snippets if snippet]
        if snippets:
            parts.append(f"Base class {base_name}:\n" + "\n\n".join(snippets))
        if len(parts) >= limit:
            break
    return "\n\n".join(parts)


def _related_implementation_sources(
    checkout: Path,
    lead: MatchSite,
    *,
    limit: int = 4,
) -> str:
    """Retrieve sibling implementations that share the lead class' direct base.

    This is deterministic code retrieval, not semantic guesswork. It exposes
    the blast radius of an ancestor edit (for example PCA and FastICA sharing
    ComponentAnalysisImplementation) before the Fixer chooses patch scope.
    """

    from fedotllm.agents.evolve.discovery.repo_map import iter_symbols

    target = checkout / lead.file_path
    if deny_write(target, checkout=checkout) or not target.is_file():
        return ""
    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    owner = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.lineno <= lead.line <= (node.end_lineno or node.lineno)
        ),
        None,
    )
    if owner is None:
        return ""
    base_names = {
        name
        for base in owner.bases
        if (name := _ast_name(base)) is not None
    }
    if not base_names:
        return ""

    candidates: list[tuple[int, str, int, str]] = []
    parsed: dict[str, ast.Module] = {lead.file_path: tree}
    for symbol in iter_symbols(checkout):
        if symbol.kind != "class" or symbol.name == owner.name:
            continue
        path = checkout / symbol.file_path
        if symbol.file_path not in parsed:
            try:
                parsed[symbol.file_path] = ast.parse(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except (OSError, SyntaxError, ValueError):
                continue
        node = next(
            (
                item
                for item in parsed[symbol.file_path].body
                if isinstance(item, ast.ClassDef)
                and item.name == symbol.name
                and item.lineno == symbol.line
            ),
            None,
        )
        if node is None:
            continue
        shared = base_names.intersection(
            name for base in node.bases if (name := _ast_name(base)) is not None
        )
        if not shared:
            continue
        same_file = 0 if symbol.file_path == lead.file_path else 1
        candidates.append(
            (same_file, symbol.file_path, symbol.line, ", ".join(sorted(shared)))
        )

    parts: list[str] = []
    for _, file_path, line, shared in sorted(candidates)[: max(1, limit)]:
        snippet = show_source(file_path, checkout=checkout, around=line)
        if snippet:
            parts.append(f"Shares {shared}:\n{snippet}")
    return "\n\n".join(parts)


def _ast_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _callee_sources(
    checkout: Path,
    lead: MatchSite,
    *,
    skip_file: str | None = None,
    limit: int = 4,
) -> str:
    """Bodies of functions this site calls. Same-file callees skipped if the whole file is already in context."""

    from fedotllm.agents.evolve.discovery.repo_map import (
        _call_edges,
        _resolve_call,
        iter_symbols,
        skip_metric_noise,
    )

    rel = lead.file_path.replace("\\", "/")
    symbols = [
        item
        for item in iter_symbols(checkout)
        if item.kind in {"method", "function"} and not skip_metric_noise(item.file_path)
    ]
    current = None
    for item in symbols:
        if item.file_path != rel or item.line > lead.line:
            continue
        if current is None or item.line > current.line:
            current = item
    if current is None:
        return ""
    by_name: dict[str, list] = {}
    for item in symbols:
        by_name.setdefault(item.name, []).append(item)
    names = _call_edges(checkout).get((rel, current.line), ())
    parts: list[str] = []
    seen: set[tuple[str, int]] = set()
    for called in names:
        for nxt in _resolve_call(called, current, by_name):
            key = (nxt.file_path, nxt.line)
            if key in seen or (skip_file and nxt.file_path == skip_file):
                continue
            seen.add(key)
            snippet = show_source(checkout / nxt.file_path, checkout=checkout, around=nxt.line)
            if snippet:
                parts.append(snippet)
            if len(parts) >= limit:
                return "\n\n".join(parts)
    return "\n\n".join(parts)


def _window(lines: list[str], around: int, *, radius: int) -> tuple[int, int]:
    around = min(max(1, around), max(1, len(lines)))
    span = _enclosing_span(lines, around)
    if span is None:
        start = max(0, around - 1 - radius)
        end = min(len(lines), around + radius)
        return start, end
    start, end = span
    if end - start > _MAX_FUNC_LINES:
        mid = around - 1
        start = max(start, mid - radius)
        end = min(end, mid + radius + 1)
    return start, end


def _enclosing_span(lines: list[str], around: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return None
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        if start <= around <= end:
            width = end - start
            if best is None or width < (best[1] - best[0]):
                best = (start - 1, end)
    return best
