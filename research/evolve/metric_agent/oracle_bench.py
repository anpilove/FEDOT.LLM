"""Oracle repair benchmark. Harness-only. Not imported by scout/discover/fixer."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from research.evolve.metric_agent.checkout import snapshot_diff
from research.evolve.metric_agent.context import context_from_lead
from research.evolve.metric_agent.journal import sha256_text, write_artifact
from research.evolve.metric_agent.patch import apply_patch
from research.evolve.metric_agent.propose import build_prompt, propose_patch
from research.evolve.metric_agent.smoke import import_error
from research.evolve.metric_agent.types import Lead, PatchCandidate

PCA_FILE = "fedot/core/operations/evaluation/operation_implementations/data_operations/sklearn_transformations.py"
KNN_FILE = "fedot/core/operations/evaluation/operation_implementations/models/knn.py"

_GENERIC_CONTRACT = (
    "If you change feature selection, scaling, or column-index handling in fit, "
    "apply the same rule in transform and predict."
)

_FILTER_TOKENS = (
    "categorical_idx",
    "numerical_idx",
    "encoded_idx",
    "ids_to_process",
    "get_not_encoded",
    "non_categorical",
    "subset_features",
    "divide_data_categorical",
)
_SCALE_TOKENS = ("StandardScaler", "MinMaxScaler", "scaler", "scale_", ".scale(")
_BOUND_WIDTH = ("shape[1]", "n_features", "n_cols", "n_columns", "features.shape")
_BOUND_OP = ("clip", "minimum", "< ", "<= ", "valid", "bound", "min(", "intersect")


@dataclass(frozen=True)
class OracleCase:
    case_id: str
    file_path: str
    class_name: str
    method: str | None
    why: str
    run_dev: bool = False
    test_paths: tuple[str, ...] = ()


CASES: dict[str, OracleCase] = {
    "pca": OracleCase(
        "pca",
        PCA_FILE,
        "PCAImplementation",
        None,
        "method PCAImplementation",
        run_dev=True,
    ),
    "knn": OracleCase(
        "knn",
        KNN_FILE,
        "FedotKnnClassImplementation",
        "fit",
        "method FedotKnnClassImplementation.fit",
    ),
    "imputation": OracleCase(
        "imputation",
        PCA_FILE,
        "ImputationImplementation",
        "fit",
        "method ImputationImplementation.fit",
        test_paths=("test/unit/data_operations/test_data_operations_implementations.py",),
    ),
}

COMPACT_MATRIX: tuple[tuple[str, str, int, int, str], ...] = (
    ("pca", "auto", 1, 1, "stock"),
    ("pca", "auto", 3, 1, "stock"),
    ("pca", "auto", 1, 3, "stock"),
    ("pca", "slice", 1, 1, "stock"),
    ("pca", "dep", 1, 1, "stock"),
    ("pca", "whole", 1, 1, "stock"),
    ("knn", "auto", 1, 1, "stock"),
    ("knn", "auto", 3, 1, "stock"),
    ("knn", "dep", 1, 1, "stock"),
    ("imputation", "auto", 1, 1, "stock"),
    ("imputation", "auto", 1, 3, "stock"),
)


@dataclass
class AttemptRecord:
    case_id: str
    file_path: str
    line: int
    why: str
    mode: str
    sample: int
    max_edits: int
    prompt_arm: str
    context_chars: int
    prompt_chars: int
    key_missing: bool
    valid_patch: bool
    n_edits: int
    changed_lines: int
    patch_hash: str
    import_ok: bool
    tests_ok: bool | None
    ast_gold_like: bool
    semantic_repair_success: bool
    shallow: bool
    identical_to_earlier: bool
    dev_status: str | None = None
    dev_score: float | None = None
    dev_delta: float | None = None
    rationale: str = ""
    folder: str = ""
    notes: list[str] = field(default_factory=list)


def symbol_line(checkout: Path, rel: str, class_name: str, method: str | None = None) -> int:
    tree = ast.parse((checkout / rel).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        if method is None:
            return node.lineno
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method:
                return item.lineno
    raise LookupError(f"{rel} {class_name}.{method or '<class>'}")


def oracle_lead(checkout: Path, case: OracleCase) -> Lead:
    line = symbol_line(checkout, case.file_path, case.class_name, case.method)
    return Lead(channel="oracle", file_path=case.file_path, line=line, why=case.why)


def _parse_with_text(text: str) -> ast.Module:
    return ast.parse(text)


def pca_ast_gold_like(source: str) -> bool:
    """Eval-only: fit and transform both subset non-categorical columns for PCA."""

    tree = _parse_with_text(source)
    pca = _class_node(tree, "PCAImplementation")
    parent = _class_node(tree, "ComponentAnalysisImplementation")
    fit = _node_src(source, pca, "fit") or _node_src(source, parent, "fit")
    transform = _node_src(source, pca, "transform") or _node_src(source, parent, "transform")
    return _has_any(fit, _FILTER_TOKENS) and _has_any(transform, _FILTER_TOKENS)


def knn_ast_gold_like(source: str) -> bool:
    """Eval-only: fit scales features and predict uses the fitted scaler."""

    tree = _parse_with_text(source)
    knn = _class_node(tree, "FedotKnnClassImplementation")
    parent = _class_node(tree, "KNeighborsImplementation")
    fit = _node_src(source, knn, "fit")
    predict = _node_src(source, knn, "predict") or _node_src(source, parent, "predict")
    proba = _node_src(source, knn, "predict_proba")
    if not fit or not predict:
        return False
    if "classes_ = self.classes" in fit and not _has_any(fit, _SCALE_TOKENS):
        return False
    scaled_fit = _has_any(fit, _SCALE_TOKENS) and ("fit_transform" in fit or ".fit(" in fit)
    uses_scaler = _has_any(predict, _SCALE_TOKENS) or "scaler" in predict
    if proba and not (_has_any(proba, _SCALE_TOKENS) or "scaler" in proba):
        return False
    return bool(scaled_fit and uses_scaler)


def imputation_ast_gold_like(source: str) -> bool:
    """Eval-only: fit and transform bound column indices to feature width."""

    tree = _parse_with_text(source)
    impl = _class_node(tree, "ImputationImplementation")
    fit = _node_src(source, impl, "fit")
    transform = _node_src(source, impl, "transform")
    return _bounds_idx(fit) and _bounds_idx(transform)


def ast_gold_like(case_id: str, source: str) -> bool:
    if case_id == "pca":
        return pca_ast_gold_like(source)
    if case_id == "knn":
        return knn_ast_gold_like(source)
    if case_id == "imputation":
        return imputation_ast_gold_like(source)
    return False


def is_shallow(case_id: str, candidate: PatchCandidate | None, source: str) -> bool:
    if candidate is None:
        return True
    blob = "\n".join([candidate.old_code, candidate.new_code, candidate.rationale, source])
    if case_id == "knn" and "classes_ = self.classes" in (candidate.new_code or "") and not ast_gold_like(case_id, source):
        return True
    if case_id == "pca" and not ast_gold_like(case_id, source):
        if any(token in blob for token in ("n_components", "svd_solver", "random_state")) and not _has_any(blob, _FILTER_TOKENS):
            return True
        return not _has_any(source, _FILTER_TOKENS)
    if case_id == "imputation" and not ast_gold_like(case_id, source):
        return True
    return not ast_gold_like(case_id, source)


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _node_src(source: str, owner: ast.ClassDef | None, name: str) -> str:
    if owner is None:
        return ""
    for item in owner.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return ast.get_source_segment(source, item) or ""
    return ""


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in (text or "") for token in tokens)


def _bounds_idx(method_src: str) -> bool:
    if not method_src:
        return False
    has_idx = any(token in method_src for token in ("categorical", "numerical_idx", "encoded_idx", "_ids"))
    return has_idx and _has_any(method_src, _BOUND_WIDTH) and _has_any(method_src, _BOUND_OP)


def changed_line_count(hunks: list[tuple[str, str]]) -> int:
    from difflib import SequenceMatcher

    n = 0
    for old, new in hunks:
        matcher = SequenceMatcher(a=old.splitlines(), b=new.splitlines())
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                n += max(i2 - i1, j2 - j1)
    return n


def _read_gold_replacements() -> dict[str, str]:
    """Eval-only. Never called from fixer/propose. Folder may be absent."""

    root = Path(__file__).resolve().parents[3] / "_local_fedot_patches" / "replacements"
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in root.glob("*.py"):
        out[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return out


class RecordingInference:
    def __init__(self, inner):
        self.inner = inner
        self.last_raw = ""
        if hasattr(inner, "query"):
            orig = inner.query

            def wrapped(messages):
                out = orig(messages)
                self.last_raw = out if isinstance(out, str) else str(out)
                return out

            inner.query = wrapped

    def create(self, prompt, model):
        parsed = self.inner.create(prompt, model)
        if not self.last_raw and parsed is not None:
            dump = getattr(parsed, "model_dump_json", None)
            self.last_raw = dump() if dump else str(parsed)
        return parsed


def run_targeted_tests(checkout: Path, paths: tuple[str, ...], *, timeout_s: float = 45) -> dict:
    existing = [p for p in paths if (checkout / p).is_file()]
    if not existing:
        return {"ran": False, "ok": None, "output": ""}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *existing, "-q", "--tb=line", "--maxfail=3"],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ran": True, "ok": False, "output": (str(exc.stdout or "") + str(exc.stderr or ""))[-1500:]}
    text = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-1500:]
    return {"ran": True, "ok": proc.returncode == 0, "output": text}


def run_dev_pca(checkout: Path) -> dict:
    from research.evolve.metric_agent.eval import run_patched

    result = run_patched("pca->catboost", checkout=checkout)
    delta = None
    if result.status == "ok":
        delta = result.score - 0.5
    return {
        "status": result.status,
        "score": result.score,
        "delta": delta,
        "detail": (result.detail or "")[:300],
    }


def run_arm(
    *,
    checkout: Path,
    case: OracleCase,
    inference,
    workspace: Path,
    mode: str,
    samples: int,
    max_edits: int,
    prompt_arm: str,
    run_dev: bool,
) -> list[AttemptRecord]:
    lead = oracle_lead(checkout, case)
    ctx = context_from_lead(lead, checkout, mode=mode)
    contract = _GENERIC_CONTRACT if prompt_arm == "hypothesis" else ""
    prompt = build_prompt(ctx, max_edits=max_edits, contract=contract)
    target = checkout / case.file_path
    original = target.read_bytes()
    key_missing = inference is None
    recorder = RecordingInference(inference) if inference is not None else None
    hashes: list[str] = []
    records: list[AttemptRecord] = []
    gold_files = _read_gold_replacements()
    arm_dir = workspace / case.case_id / f"{mode}_s{samples}_e{max_edits}_{prompt_arm}"
    write_artifact(arm_dir, "location.txt", f"{case.file_path}:{lead.line} {case.why}\n")
    write_artifact(arm_dir, "context.txt", ctx)
    write_artifact(arm_dir, "prompt.txt", prompt)

    for sample in range(1, samples + 1):
        folder = arm_dir / f"sample_{sample}"
        target.write_bytes(original)
        notes: list[str] = []
        candidate: PatchCandidate | None = None
        raw = ""
        if recorder is None:
            notes.append("KEY_MISSING")
        else:
            recorder.last_raw = ""
            candidate = propose_patch(
                inference=recorder,
                context=ctx,
                max_edits=max_edits,
                contract=contract,
            )
            raw = recorder.last_raw or ""
            write_artifact(folder, "raw_model.txt", raw)
        valid = False
        n_edits = 0
        changed = 0
        patch_hash = ""
        import_ok = False
        tests_ok: bool | None = None
        source = original.decode("utf-8", errors="replace")
        if candidate is not None:
            write_artifact(folder, "rationale.txt", candidate.rationale)
            write_artifact(folder, "old.py", candidate.old_code)
            write_artifact(folder, "new.py", candidate.new_code)
            write_artifact(folder, "hunks.json", json.dumps(candidate.hunks or [], indent=2))
            try:
                valid = apply_patch(checkout, candidate)
            except PermissionError as exc:
                notes.append(str(exc))
                valid = False
            n_edits = len(candidate.hunks) if candidate.hunks else 1
            changed = changed_line_count(candidate.hunks or [(candidate.old_code, candidate.new_code)])
        if valid:
            source = target.read_text(encoding="utf-8", errors="replace")
            write_artifact(folder, "applied.diff", snapshot_diff(checkout, case.file_path))
            patch_hash = sha256_text(source)
            broken = import_error(checkout, case.file_path)
            import_ok = broken is None
            write_artifact(folder, "import.txt", "ok" if import_ok else (broken or "fail"))
            if import_ok and case.test_paths:
                test_info = run_targeted_tests(checkout, case.test_paths)
                tests_ok = test_info["ok"]
                write_artifact(folder, "unit_tests.txt", json.dumps(test_info, indent=2)[:4000])
            elif import_ok:
                tests_ok = True
        ast_ok = ast_gold_like(case.case_id, source) if valid else False
        if gold_files:
            notes.append(f"gold_replacements_present={sorted(gold_files)}")
        identical = patch_hash in hashes and bool(patch_hash)
        if patch_hash:
            hashes.append(patch_hash)
        dev: dict | None = None
        semantic = False
        if case.case_id == "pca":
            if run_dev and valid and import_ok and not identical:
                dev = run_dev_pca(checkout)
                write_artifact(folder, "dev.json", json.dumps(dev, indent=2, default=str))
                semantic = dev.get("status") == "ok"
            else:
                notes.append("dev_skipped" if not identical else "dev_skipped_identical")
        else:
            semantic = ast_ok
        rec = AttemptRecord(
            case_id=case.case_id,
            file_path=case.file_path,
            line=lead.line,
            why=case.why,
            mode=mode,
            sample=sample,
            max_edits=max_edits,
            prompt_arm=prompt_arm,
            context_chars=len(ctx),
            prompt_chars=len(prompt),
            key_missing=key_missing,
            valid_patch=valid,
            n_edits=n_edits,
            changed_lines=changed,
            patch_hash=patch_hash,
            import_ok=import_ok,
            tests_ok=tests_ok,
            ast_gold_like=ast_ok,
            semantic_repair_success=semantic,
            shallow=is_shallow(case.case_id, candidate, source) if valid else True,
            identical_to_earlier=identical,
            dev_status=None if dev is None else str(dev.get("status")),
            dev_score=None if dev is None else dev.get("score"),
            dev_delta=None if dev is None else dev.get("delta"),
            rationale="" if candidate is None else candidate.rationale,
            folder=str(folder),
            notes=notes,
        )
        write_artifact(folder, "result.json", json.dumps(asdict(rec), indent=2, default=str))
        records.append(rec)
        target.write_bytes(original)
        if samples > 1 and identical and sample == 1:
            notes.append("seed_may_repeat")
    write_artifact(
        arm_dir,
        "summary.json",
        json.dumps(
            {
                "unique_patch_hashes": sorted(set(hashes)),
                "identical_under_seed": len(set(hashes)) == 1 and len(hashes) > 1,
                "valid_rate": sum(r.valid_patch for r in records) / max(1, len(records)),
                "semantic_rate": sum(r.semantic_repair_success for r in records) / max(1, len(records)),
                "tests_ok_rate": sum(1 for r in records if r.tests_ok) / max(1, len(records)),
                "gold_like_only_after_sample_1": any(r.ast_gold_like and r.sample > 1 for r in records)
                and not any(r.ast_gold_like and r.sample == 1 for r in records),
            },
            indent=2,
        ),
    )
    return records


def compact_matrix(
    *,
    checkout: Path,
    inference,
    workspace: Path,
    cases: tuple[str, ...] | None = None,
    run_dev: bool | None = None,
) -> list[AttemptRecord]:
    wanted = set(cases or CASES)
    out: list[AttemptRecord] = []
    for case_id, mode, samples, max_edits, prompt_arm in COMPACT_MATRIX:
        if case_id not in wanted:
            continue
        case = CASES[case_id]
        dev = case.run_dev if run_dev is None else run_dev
        out.extend(
            run_arm(
                checkout=checkout,
                case=case,
                inference=inference,
                workspace=workspace,
                mode=mode,
                samples=samples,
                max_edits=max_edits,
                prompt_arm=prompt_arm,
                run_dev=dev,
            )
        )
    return out


def summarize(records: list[AttemptRecord]) -> dict:
    rows = []
    for rec in records:
        rows.append(
            {
                "case": rec.case_id,
                "exact_symbol": f"{rec.file_path}:{rec.line} {rec.why}",
                "context_arm": rec.mode,
                "samples": rec.sample,
                "edit_budget": rec.max_edits,
                "semantic_success": rec.semantic_repair_success,
                "tests": rec.tests_ok,
                "DEV_delta": rec.dev_delta,
                "valid_patch": rec.valid_patch,
                "shallow": rec.shallow,
                "ast_gold_like": rec.ast_gold_like,
                "context_chars": rec.context_chars,
                "n_edits": rec.n_edits,
                "patch_hash": rec.patch_hash,
                "identical_to_earlier": rec.identical_to_earlier,
                "key_missing": rec.key_missing,
                "prompt_arm": rec.prompt_arm,
            }
        )
    return {
        "n": len(records),
        "key_missing": all(r.key_missing for r in records) if records else True,
        "semantic_any": any(r.semantic_repair_success for r in records),
        "rows": rows,
    }


def dump_context_probe(checkout: Path, workspace: Path) -> dict:
    """No LLM: measure what each context arm actually includes at oracle sites."""

    probe = {}
    for case in CASES.values():
        lead = oracle_lead(checkout, case)
        probe[case.case_id] = {"line": lead.line, "why": lead.why, "arms": {}}
        for mode in ("auto", "slice", "whole", "dep"):
            ctx = context_from_lead(lead, checkout, mode=mode)
            markers = {
                "pca_fit_call": "self.pca.fit(" in ctx,
                "pca_transform_call": "self.pca.transform(" in ctx,
                "knn_parent_predict": "def predict(" in ctx and "self.model.predict(" in ctx,
                "init": "def __init__(" in ctx,
                "scaler_hint": "StandardScaler" in ctx or "MinMaxScaler" in ctx,
                "numerical_idx": "numerical_idx" in ctx,
                "chars": len(ctx),
            }
            probe[case.case_id]["arms"][mode] = markers
            write_artifact(workspace / "context_probe" / case.case_id, f"{mode}.txt", ctx)
    write_artifact(workspace, "context_probe.json", json.dumps(probe, indent=2))
    return probe
