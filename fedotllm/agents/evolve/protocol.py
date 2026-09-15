"""Stable identities for numeric scoring and candidate acceptance semantics.

Cross-run memory must be strict enough to reject stale measurements, but it
must not invalidate every experiment when a comment, prompt or controller-only
configuration field changes.  The two fingerprints below intentionally cover
different contracts:

* score: data, splits, workload construction and numeric metric execution;
* acceptance: score plus patch, probe, pytest and KEEP/DROP semantics.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SCORE_PROTOCOL_SCHEMA = "score-v1"
ACCEPTANCE_PROTOCOL_SCHEMA = (
    "acceptance-v2:full-dev-protect+2of3-dev+shadow+single-final+signal-not-keep+resume-evidence+shared-signal-full-scope"
)


class _StripDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def semantic_python_fingerprint(
    path: Path,
    *,
    names: tuple[str, ...] = (),
) -> str:
    """Hash executable AST while ignoring comments, formatting and docstrings."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return hashlib.sha256(f"missing-or-invalid:{path.name}:{exc}".encode()).hexdigest()
    if names:
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in names
        ]
        missing = sorted(set(names) - {node.name for node in selected})
        tree = ast.Module(body=selected, type_ignores=[])
        if missing:
            tree.body.append(ast.Expr(value=ast.Constant(value=f"missing:{','.join(missing)}")))
    tree = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _digest_parts(schema: str, parts: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256(schema.encode())
    for name, value in parts:
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()[:16]


def score_protocol_fingerprint() -> str:
    """Identity of rows, splits, models and numeric scores."""

    root = Path(__file__).resolve().parent
    parts = [
        (name, semantic_python_fingerprint(root / name))
        for name in (
            "evaluation/scorer.py",
            "evaluation/independent_data.py",
            "evaluation/tasks.py",
            "evaluation/_worker.py",
            "evaluation/_fedot_quality_worker.py",
            "evaluation/quality_registry.py",
            "evaluation/fedot_quality.py",
            "evaluation/openml_fold.py",
            "execution/process.py",
        )
    ]
    parts.append(
        (
            "evaluation/eval.py",
            semantic_python_fingerprint(
                root / "evaluation/eval.py",
                names=("run_stock", "run_patched", "_run", "_env_hash"),
            ),
        )
    )
    manifest = root / "benchmark_manifest.json"
    manifest_hash = hashlib.sha256(
        manifest.read_bytes() if manifest.is_file() else b"<missing>"
    ).hexdigest()
    parts.append(("benchmark_manifest.json", manifest_hash))
    quality_registry = root / "evaluation" / "quality_registry.json"
    quality_hash = hashlib.sha256(
        quality_registry.read_bytes() if quality_registry.is_file() else b"<missing>"
    ).hexdigest()
    parts.append(("evaluation/quality_registry.json", quality_hash))
    return _digest_parts(SCORE_PROTOCOL_SCHEMA, parts)


def acceptance_protocol_fingerprint() -> str:
    """Identity required before an exact historical patch may be skipped."""

    root = Path(__file__).resolve().parent
    parts: list[tuple[str, str]] = [("score", score_protocol_fingerprint())]
    for name in (
        "evaluation/compare.py",
        "evaluation/test_contracts.py",
        "execution/patch.py",
        "evaluation/affected_eval.py",
        "evaluation/uncertainty.py",
        "execution/smoke.py",
        "execution/run_code.py",
        "execution/_snippet_worker.py",
        "controller/ablation.py",
        "controller/probes.py",
        "controller/metric_study.py",
        "controller/transfer.py",
        "controller/campaign.py",
        "controller/quality_executor.py",
        "controller/quality_queue.py",
        "storage/hypothesis.py",
        "agents/verifier.py",
        "discovery/targets.py",
        "discovery/research_tools.py",
    ):
        parts.append((name, semantic_python_fingerprint(root / name)))
    parts.extend(
        (
            name,
            semantic_python_fingerprint(root / file_name, names=functions),
        )
        for name, file_name, functions in (
            (
                "judge-tests",
                "evaluation/judge.py",
                (
                    "normalize_test_result",
                    "measure_fedot_tests",
                    "_baseline_test_fingerprint",
                    "measure_baseline_fedot_tests",
                    "tests_regressed",
                    "confirm_candidate_tests",
                    "verdict",
                ),
            ),
            (
                "pytest-runner",
                "discovery/signals.py",
                (
                    "failed_pytest_nodes",
                    "pytest_failure_excerpt",
                    "pytest_result",
                    "pytest_snapshot",
                ),
            ),
            (
                "probe-deduplication",
                "storage/replay.py",
                ("_matching_patch_findings", "_probe_only_rejection",
                 "tried_patch_hashes_from_findings", "rejected_probe_hashes_from_findings"),
            ),
            (
                "resume-verification",
                "storage/replay.py",
                ("load_resume_branch",),
            ),
            (
                "controller-feedback",
                "controller/feedback.py",
                ("_blocking_lift_crash_ids",),
            ),
            (
                "controller-confirmation",
                "controller/confirmation.py",
                ("confirm_dev", "quick_quality_screen"),
            ),
            (
                "quality-queue",
                "controller/quality_queue.py",
                ("enqueue_quality_job", "queue_priority"),
            ),
            (
                "controller-final",
                "controller/finalization.py",
                ("record_final",),
            ),
        )
    )
    return _digest_parts(ACCEPTANCE_PROTOCOL_SCHEMA, parts)
