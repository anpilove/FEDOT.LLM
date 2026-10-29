"""Deterministic pre-score routing from changed FEDOT code to quality families."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterable

PROBLEM_FAMILIES = ("classification", "regression", "ts_forecasting")
_TS_MARKERS = (
    "time_series",
    "forecast",
    "ts_implementations",
    "ts_transformations",
    "/ts/",
    "/ts_",
)
_CLASSIFICATION_MARKERS = ("classification", "classifier", "classif_")
_REGRESSION_RE = re.compile(
    r"(?<![a-z0-9])(?:(?:regressions?|regressors?)(?![a-z0-9])|regress_)"
)
_TABULAR_MARKERS = ("boostings_implementations", "sklearn_eval", "models/sklearn")
_MODEL_PATH_FAMILIES = {
    # sklearn_regr is registered for both regression and lagged TS pipelines.
    "operations/evaluation/regression.py": ("regression", "ts_forecasting"),
    # These strategies explicitly reject forecasting tasks in FEDOT's fit path.
    "operations/evaluation/boostings.py": ("classification", "regression"),
    "operations/evaluation/tabpfn.py": ("classification", "regression"),
    "models/discriminant_analysis.py": ("classification",),
    "models/svc.py": ("classification",),
    "models/keras.py": ("classification",),
    "models/knn.py": ("classification", "regression"),
    "models/tabpfn.py": ("classification", "regression"),
    # Custom models can declare table, image, text, or time-series inputs.
    "models/custom_model.py": PROBLEM_FAMILIES,
}
_CATEGORICAL_ENCODER_PATH = (
    "fedot/core/operations/evaluation/operation_implementations/"
    "data_operations/categorical_encoders.py"
)
# FEDOT calls this helper only under categorical_ids.size > 0. Other methods
# in the encoder module also run on numerical data and must stay broad.
_CATEGORICAL_ONLY_SYMBOLS = frozenset({
    "OneHotEncodingImplementation._apply_one_hot_encoding",
})
_SHARED_MARKERS = (
    "fedot/core/data/",
    "fedot/core/pipelines/",
    "fedot/core/composer/",
    "data_operations/",
    "preprocessing/",
)


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            str(item) for item in value.values() if isinstance(item, (str, int, float))
        ]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


def _problem_hits(text: str) -> set[str]:
    blob = text.replace("\\", "/").lower()
    hits: set[str] = set()
    if any(marker in blob for marker in _TS_MARKERS):
        hits.add("ts_forecasting")
    if any(marker in blob for marker in _CLASSIFICATION_MARKERS):
        hits.add("classification")
    if _REGRESSION_RE.search(blob):
        hits.add("regression")
    for suffix, families in _MODEL_PATH_FAMILIES.items():
        if blob.endswith(suffix):
            hits.update(families)
            break
    if not hits and any(marker in blob for marker in _TABULAR_MARKERS):
        hits.update(("classification", "regression"))
    # A TS-specific implementation can live under the generic data_operations/
    # package. The specific signal is stronger than that directory name; only
    # widen to all families when the path itself has no task-family signal.
    if not hits and any(marker in blob for marker in _SHARED_MARKERS):
        hits.update(PROBLEM_FAMILIES)
    return hits


def extract_change_tags(code_fragments: Iterable[str]) -> list[str]:
    """Extract conservative dataset-trait signals from changed code only."""

    fragments = [str(fragment) for fragment in code_fragments]
    markers = ("categorical", "categoric", "category", "categories", "cat_features")
    identifier_matches: list[bool] = []
    for fragment in fragments:
        try:
            names = [
                token.string.lower()
                for token in tokenize.generate_tokens(io.StringIO(fragment).readline)
                if token.type == tokenize.NAME
            ]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return []
        identifier_matches.append(any(marker in name for name in names for marker in markers))
    if identifier_matches and all(identifier_matches):
        return ["categorical"]
    return []


def _changed_files(job: dict) -> list[str]:
    paths = [str(path) for path in _strings(job.get("changed_files")) if str(path)]
    if not paths and job.get("file_path"):
        paths = [str(job["file_path"])]
    normalized = [
        path.replace("\\", "/").removeprefix("a/").removeprefix("b/") for path in paths
    ]
    return list(dict.fromkeys(normalized))


def analyze_quality_scope(job: dict, registry=None) -> dict:
    """Return family scope and evidence, widening any ambiguity to all families."""

    from fedotllm.agents.evolve.evaluation.quality_registry import load_quality_registry

    registry = registry or load_quality_registry()
    files = _changed_files(job)
    symbols = [str(item) for item in _strings(job.get("changed_symbols"))]
    if not symbols and job.get("quality_scope"):
        symbols = [
            str(item) for item in _strings(job["quality_scope"].get("changed_symbols"))
        ]

    families: set[str] = set()
    reasons: list[str] = []
    confidence = "direct"
    if not files:
        families.update(PROBLEM_FAMILIES)
        reasons.append("no_changed_paths")
        confidence = "fallback"
    else:
        for path in files:
            matched = _problem_hits(path)
            if matched:
                families.update(matched)
                reasons.append(f"path:{path}")
            else:
                families.update(PROBLEM_FAMILIES)
                reasons.append(f"unmapped_path:{path}")
                confidence = "fallback"

    for symbol in symbols:
        matched = _problem_hits(symbol)
        if matched:
            families.update(matched)
            reasons.append(f"ast_symbol:{symbol}")
    if job.get("ast_errors"):
        families.update(PROBLEM_FAMILIES)
        reasons.extend(f"ast_error:{item}" for item in _strings(job["ast_errors"]))
        confidence = "fallback"

    # The hypothesis describes the expected mechanism, but must not widen a
    # task-specific source path based on words like "regression node". Shared
    # and unmapped paths already select the full suite, and multi-file patches
    # union the families evidenced by every changed path.
    hypothesis = job.get("hypothesis") or {}
    hypothesis_text = " ".join(_strings(hypothesis))

    stored_scope = job.get("quality_scope") or {}
    stored_families = set(_strings(stored_scope.get("problem_families")))
    if stored_families - families:
        families.update(stored_families)
        reasons.append("stored_scope_widened_scope")

    # Identifiers and hypothesis wording alone do not establish exclusivity.
    # Only an audited helper whose callers require categorical data may narrow
    # the dataset set; shared/unknown code and unresolved edits stay broad.
    changed_code_tags = {
        str(item).strip().lower() for item in _strings(job.get("changed_code_tags"))
    }
    hypothesis_lower = hypothesis_text.lower()
    dataset_traits = (
        ["categorical"]
        if "categorical" in changed_code_tags
        and files
        and all(path == _CATEGORICAL_ENCODER_PATH for path in files)
        and symbols
        and set(symbols) <= _CATEGORICAL_ONLY_SYMBOLS
        and not job.get("ast_errors")
        and any(word in hypothesis_lower for word in ("categorical", "category", "categories"))
        else []
    )
    if dataset_traits:
        reasons.append("guarded_encoder_helper:categorical")

    if not families or not families <= set(PROBLEM_FAMILIES):
        families.update(PROBLEM_FAMILIES)
        reasons.append("invalid_scope_fallback")
        confidence = "fallback"
    ordered = [family for family in PROBLEM_FAMILIES if family in families]
    return {
        "problem_families": ordered,
        "dataset_traits": dataset_traits,
        "changed_files": files,
        "changed_symbols": symbols,
        "reasons": reasons,
        "confidence": confidence,
    }


def extract_changed_symbols(
    source_root: Path, edits: Iterable[dict | object]
) -> tuple[list[str], list[str]]:
    """Use AST to find enclosing functions/classes for changed source snippets."""

    root = Path(source_root).resolve()
    symbols: list[str] = []
    errors: list[str] = []
    for edit in edits:
        raw_path = (
            edit.get("file_path", "")
            if isinstance(edit, dict)
            else getattr(edit, "file_path", "")
        )
        old_code = (
            edit.get("old_code", "")
            if isinstance(edit, dict)
            else getattr(edit, "old_code", "")
        )
        new_code = (
            edit.get("new_code")
            if isinstance(edit, dict)
            else getattr(edit, "new_code", None)
        )
        rel = Path(str(raw_path))
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            errors.append(f"{rel.as_posix()}:source_missing")
            continue
        text = target.read_text(encoding="utf-8")
        offset = text.find(str(old_code)) if old_code else -1
        if offset < 0:
            errors.append(f"{rel.as_posix()}:edit_not_found")
            continue
        line = text.count("\n", 0, offset) + 1
        end_line = line + str(old_code).rstrip("\r\n").count("\n")
        try:
            tree = ast.parse(text, filename=rel.as_posix())
        except SyntaxError:
            errors.append(f"{rel.as_posix()}:SyntaxError")
            continue
        enclosing = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno <= line
            and end_line <= (node.end_lineno or node.lineno)
        ]
        enclosing.sort(
            key=lambda node: (node.lineno, -(node.end_lineno or node.lineno))
        )
        if enclosing:
            symbol = ".".join(node.name for node in enclosing)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        else:
            errors.append(f"{rel.as_posix()}:edit_outside_symbol")
            continue
        if new_code is not None:
            # Added code can escape the original helper with a dedent. Check
            # the replacement's whole span and enclosing lineage as well.
            replacement = str(new_code)
            updated = text[:offset] + replacement + text[offset + len(str(old_code)):]
            try:
                new_tree = ast.parse(updated, filename=rel.as_posix())
            except SyntaxError:
                errors.append(f"{rel.as_posix()}:new_SyntaxError")
                continue
            new_end_line = line + replacement.rstrip("\r\n").count("\n")
            new_enclosing = [
                node for node in ast.walk(new_tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.lineno <= line
                and new_end_line <= (node.end_lineno or node.lineno)
            ]
            new_enclosing.sort(key=lambda node: (node.lineno, -(node.end_lineno or node.lineno)))
            if ".".join(node.name for node in new_enclosing) != symbol:
                errors.append(f"{rel.as_posix()}:new_edit_outside_symbol")
    return symbols, errors
