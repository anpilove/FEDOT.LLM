"""Small deterministic RAG over documentation shipped with the FEDOT checkout."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fedotllm.agents.evolve.types import PatchSite

MAX_CHUNK_CHARS = 2_400
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_RST_HEADING = re.compile(r"^[=\-~^\"'`:+*#]{3,}$")
_STOP = {
    "and", "are", "class", "def", "for", "from", "import", "none", "not",
    "return", "self", "that", "the", "this", "with", "fedot", "implementation",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    kind: str
    path: str
    title: str
    text: str


def _terms(text: str) -> set[str]:
    # Make PCAImplementation searchable as both `pca` and `implementation`.
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "").replace("_", " ")
    return {
        token.lower()
        for token in _WORD.findall(expanded)
        if token.lower() not in _STOP
    }


def _rst_chunks(root: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    docs = root / "docs" / "source"
    if not docs.is_dir():
        return chunks
    for path in sorted(docs.rglob("*.rst")):
        rel = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        title = path.stem.replace("_", " ")
        start = 0
        for index in range(1, len(lines)):
            if not _RST_HEADING.match(lines[index].strip()):
                continue
            heading = lines[index - 1].strip()
            if not heading:
                continue
            body = "\n".join(lines[start:index - 1]).strip()
            if body:
                chunks.append(KnowledgeChunk("documentation", rel, title, body[:MAX_CHUNK_CHARS]))
            title = heading
            start = index + 1
        body = "\n".join(lines[start:]).strip()
        if body:
            chunks.append(KnowledgeChunk("documentation", rel, title, body[:MAX_CHUNK_CHARS]))
    return chunks


def _docstring_chunks(root: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    runtime = root / "fedot"
    if not runtime.is_dir():
        return chunks
    for path in sorted(runtime.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc or len(doc) < 30:
                continue
            chunks.append(
                KnowledgeChunk(
                    "docstring",
                    f"{rel}:{node.lineno}",
                    node.name,
                    doc[:MAX_CHUNK_CHARS],
                )
            )
    return chunks


def _repository_chunks(root: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    folder = root / "fedot" / "core" / "repository" / "data"
    for path in sorted(folder.glob("*_repository.json")) if folder.is_dir() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata") or {}
        for operation, row in (payload.get("operations") or {}).items():
            if not isinstance(row, dict):
                continue
            meta_name = row.get("meta")
            meta = metadata.get(meta_name, {}) if isinstance(meta_name, str) else {}
            combined = {"operation": operation, **meta, **row}
            chunks.append(
                KnowledgeChunk(
                    "operation_metadata",
                    path.relative_to(root).as_posix(),
                    operation,
                    json.dumps(combined, ensure_ascii=False, indent=2)[:MAX_CHUNK_CHARS],
                )
            )
    return chunks


def _default_parameter_chunks(root: Path) -> list[KnowledgeChunk]:
    path = root / "fedot" / "core" / "repository" / "data" / "default_operation_params.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    rel = path.relative_to(root).as_posix()
    return [
        KnowledgeChunk(
            "operation_defaults",
            rel,
            str(operation),
            json.dumps(
                {"operation": operation, "default_parameters": params},
                ensure_ascii=False,
                indent=2,
            )[:MAX_CHUNK_CHARS],
        )
        for operation, params in payload.items()
        if isinstance(params, dict)
    ]


@lru_cache(maxsize=32)
def _corpus(root_text: str) -> tuple[KnowledgeChunk, ...]:
    root = Path(root_text)
    return tuple(
        [
            *_rst_chunks(root),
            *_docstring_chunks(root),
            *_repository_chunks(root),
            *_default_parameter_chunks(root),
        ]
    )


def retrieve_knowledge(
    checkout: Path,
    query: str,
    *,
    limit: int = 4,
    max_chars: int = 6_000,
) -> str:
    """Return lexically relevant public FEDOT docs; never index Evolve artifacts."""

    query_terms = _terms(query)
    if not query_terms:
        return "<no documentation query>"
    ranked: list[tuple[int, KnowledgeChunk]] = []
    for chunk in _corpus(str(checkout.resolve())):
        title_terms = _terms(f"{chunk.path} {chunk.title}")
        body_terms = _terms(chunk.text)
        title_hits = len(query_terms & title_terms)
        body_hits = len(query_terms & body_terms)
        if not title_hits and not body_hits:
            continue
        exact_title = 1 if chunk.title.lower() in (query or "").lower() else 0
        route_bonus = 0
        if chunk.kind == "documentation":
            rel = chunk.path.lower()
            if "pipeline" in query_terms and rel.endswith("advanced/architecture.rst"):
                route_bonus += 40
            if query_terms & {"forecast", "forecasting", "lagged", "timeseries"} and rel.endswith(
                "basics/ts_forecasting.rst"
            ):
                route_bonus += 80
            if query_terms & {"preprocessing", "categorical", "encoding"} and rel.endswith(
                "advanced/data_preprocessing.rst"
            ):
                route_bonus += 40
        score = title_hits * 8 + body_hits * 2 + exact_title * 12 + route_bonus
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1].path, item[1].title))
    # Pure top-k commonly returns three adjacent method docstrings. Preserve
    # semantic diversity so the model sees both local API meaning and the
    # framework/repository contract when those sources match the query.
    def metadata_named_in_query(chunk: KnowledgeChunk) -> bool:
        title_terms = _terms(chunk.title)
        return bool(title_terms) and title_terms.issubset(query_terms)

    selected: list[tuple[int, KnowledgeChunk]] = []
    for kind in ("docstring", "documentation", "operation_metadata", "operation_defaults"):
        match = next(
            (
                item
                for item in ranked
                if item[1].kind == kind
                and (
                    kind not in {"operation_metadata", "operation_defaults"}
                    or metadata_named_in_query(item[1])
                )
            ),
            None,
        )
        if match is not None and match not in selected:
            selected.append(match)
    for item in ranked:
        if len(selected) >= max(1, limit):
            break
        if (
            item[1].kind in {"operation_metadata", "operation_defaults"}
            and not metadata_named_in_query(item[1])
        ):
            continue
        if item not in selected:
            selected.append(item)
    selected.sort(key=lambda item: (-item[0], item[1].path, item[1].title))
    parts: list[str] = []
    used = 0
    for score, chunk in selected[: max(1, limit)]:
        block = (
            f"[{chunk.kind}] {chunk.title} ({chunk.path}, relevance={score})\n"
            f"{chunk.text.strip()}"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        parts.append(block[:remaining])
        used += min(len(block), remaining) + 2
    return "\n\n".join(parts) if parts else "<no relevant FEDOT documentation>"


def knowledge_for_lead(
    checkout: Path,
    lead: PatchSite,
    *,
    max_chars: int = 4_000,
) -> str:
    # Query with the semantic lead, not a raw source slice. Imports and nearby
    # unrelated classes used to dominate lexical retrieval (for example a
    # lagged-forecast lead retrieved SimpleImputer and GaussianFilter docs).
    path_terms = Path(lead.file_path).stem.replace("_", " ")
    query = "\n".join(
        [path_terms, lead.why, *lead.evidence]
    )
    result = retrieve_knowledge(checkout, query, limit=5, max_chars=max_chars)
    if result.startswith("<"):
        return ""
    return (
        "Retrieved FEDOT architecture/docstring context (public source, not evaluator):\n"
        + result
    )
