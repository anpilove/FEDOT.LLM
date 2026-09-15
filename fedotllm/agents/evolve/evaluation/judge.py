"""Judge: DEV validation suite for KEEP/DROP. FINAL is scored separately after search. No LLM."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from fedotllm.log import logger
from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src, source_commit, source_fingerprint
from fedotllm.agents.evolve.evaluation.compare import compare_pack
from fedotllm.agents.evolve.discovery.discover import pytest_failure_excerpt, pytest_snapshot
from fedotllm.agents.evolve.evaluation.eval import run_patched, run_stock
from fedotllm.agents.evolve.execution.guard import repo_root
from fedotllm.agents.evolve.evaluation.tasks import coverage_task_limit, hidden_exam, load_task
from fedotllm.agents.evolve.types import Decision, PatchSite, ScoreResult, TestResult
from fedotllm.agents.evolve.evaluation.test_contracts import run_with_test_repairs


def normalize_test_result(value: TestResult) -> TestResult:
    if not isinstance(value, TestResult):
        raise TypeError(f"unsupported FEDOT test result: {type(value).__name__}")
    return value


def measure_stock(
    exam_ids: tuple[str, ...],
    *,
    checkout: Path | None = None,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
) -> dict[str, ScoreResult]:
    tree = checkout or resolve_fedot_src()
    return _run_suite(
        "stock",
        exam_ids,
        run_stock,
        tree,
        split=split,
        seed=seed,
        collect_coverage=collect_coverage,
    )


def measure_patched(
    exam_ids: tuple[str, ...],
    *,
    checkout: Path,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
) -> dict[str, ScoreResult]:
    return _run_suite(
        "patched",
        exam_ids,
        run_patched,
        checkout,
        split=split,
        seed=seed,
        collect_coverage=collect_coverage,
    )


def _run_suite(
    label: str,
    exam_ids: tuple[str, ...],
    run_one,
    checkout: Path,
    *,
    split: str = "dev",
    seed: int | None = None,
    collect_coverage: bool = False,
) -> dict[str, ScoreResult]:
    out: dict[str, ScoreResult] = {}
    n = len(exam_ids)
    logger.info("evolve %s 0/%s", label, n)
    for i, task_id in enumerate(exam_ids, start=1):
        cover = collect_coverage and i <= coverage_task_limit()
        result = run_one(
            task_id,
            checkout=checkout,
            split=split,
            seed=seed,
            collect_coverage=cover,
        )
        out[task_id] = result
        logger.info("evolve %s %s/%s %s %s", label, i, n, task_id, result.status)
    return out


def measure_fedot_tests(checkout: Path) -> TestResult:
    """FEDOT unit tests after a patch. Not an agent hunt signal."""

    # Discovery may stop early, acceptance must compare complete suites.
    return run_with_test_repairs(checkout, lambda tree: pytest_snapshot(tree, maxfail=0))


def _baseline_test_fingerprint(checkout: Path) -> str:
    """Fingerprint runtime, FEDOT tests and the environment used by pytest."""

    checkout = checkout.resolve()
    digest = hashlib.sha256()
    digest.update(Path(__file__).with_name("test_contracts.py").read_bytes())
    evolve = Path(__file__).resolve().parents[1]
    for relative in ("evaluation/judge.py", "discovery/signals.py", "execution/process.py"):
        digest.update((evolve / relative).read_bytes())
    digest.update(source_fingerprint(checkout).encode())
    digest.update(source_commit(checkout).encode())
    digest.update(sys.version.encode())
    digest.update(str(Path(sys.executable).resolve()).encode())
    for rel in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = checkout / rel
        digest.update(rel.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    lock = repo_root() / "uv.lock"
    digest.update(lock.read_bytes() if lock.is_file() else b"<missing-lock>")
    for test_root_name in ("test", "tests"):
        test_root = checkout / test_root_name
        if not test_root.is_dir():
            continue
        for path in sorted(test_root.rglob("*.py")):
            digest.update(path.relative_to(checkout).as_posix().encode())
            digest.update(path.read_bytes())
    # Data and even an empty directory affect archive extraction in FEDOT tests.
    # A source-only cache can otherwise compare different test inputs.
    for relative in ("examples/data", "test/data", "tests/data"):
        directory = checkout / relative
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(checkout).as_posix().encode())
            digest.update(b"<dir>" if path.is_dir() else path.read_bytes())
    return digest.hexdigest()


def measure_baseline_fedot_tests(checkout: Path, *, runner=None) -> TestResult:
    """Cache only immutable stock pytest, never a patched candidate result."""

    fingerprint = _baseline_test_fingerprint(checkout)
    use_cache = runner is None or runner is measure_fedot_tests
    cache_root = Path(
        os.environ.get(
            "EVOLVE_AGENT_TEST_BASELINE_CACHE",
            "/tmp/evolve-agent-test-baseline-cache",
        )
    )
    cache_file = cache_root / f"{fingerprint}.json"
    if use_cache:
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint:
                result = dict(payload["result"])
                result["failed_nodes"] = set(result.get("failed_nodes") or ())
                result["leads"] = [
                    item if isinstance(item, PatchSite) else PatchSite(**item)
                    for item in result.get("leads") or ()
                ]
                return TestResult(**result)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    run_tests = runner or measure_fedot_tests
    result = normalize_test_result(run_tests(checkout))
    if use_cache and result.completed:
        cache_root.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        temp.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "result": asdict(result),
                },
                ensure_ascii=False,
                default=lambda value: sorted(value) if isinstance(value, set) else str(value),
            ),
            encoding="utf-8",
        )
        temp.replace(cache_file)
    return result


def tests_regressed(
    before: TestResult | set[str],
    after: TestResult | set[str],
) -> Decision | None:
    if isinstance(before, TestResult):
        if not before.completed:
            return Decision(
                keep=False,
                reason=f"fedot_baseline_tests_{before.status}",
                target_delta=None,
                infrastructure_error=True,
            )
        before_nodes = before.failed_nodes
    else:
        before_nodes = before
    if isinstance(after, TestResult):
        if not after.completed:
            return Decision(
                keep=False,
                reason=f"fedot_tests_{after.status}",
                target_delta=None,
                infrastructure_error=True,
            )
        after_nodes = after.failed_nodes
    else:
        after_nodes = after
    extra = after_nodes - before_nodes
    if extra:
        sample = ", ".join(sorted(extra)[:3])
        return Decision(
            keep=False,
            reason=f"fedot_tests_regressed {sample}",
            target_delta=None,
            regression_deltas={},
        )
    if isinstance(before, TestResult) and isinstance(after, TestResult):
        changed = [
            node
            for node in sorted(before_nodes & after_nodes)
            if _failure_signature(before, node) != _failure_signature(after, node)
        ]
        # Empty signatures mean an old cache/test fixture did not retain the
        # actual traceback. Do not invent a difference without evidence.
        changed = [
            node
            for node in changed
            if _failure_signature(before, node) and _failure_signature(after, node)
        ]
        if changed:
            return Decision(
                keep=False,
                reason=f"fedot_tests_changed_failure {', '.join(changed[:3])}",
                target_delta=None,
                regression_deltas={},
            )
    return None


def _failure_signature(result: TestResult, node: str) -> str:
    """Normalize one pytest failure enough to compare its failure mechanism."""

    excerpt = pytest_failure_excerpt(result.output, {node}, max_chars=8_000)
    if not excerpt.strip():
        return ""
    text = re.sub(r"\x1b\[[0-9;]*m", "", excerpt)
    text = re.sub(
        r"(?m)(?<!\S)\S*site-packages/",
        "<site-packages>/",
        text,
    )
    text = re.sub(
        r"(?m)(?<!\S)(?:\S*/)?(?=(?:test|fedot)/)",
        "<checkout>/",
        text,
    )
    text = re.sub(r"(?<=\.py):\d+", ":<line>", text)
    text = re.sub(r"\bline\s+\d+\b", "line <n>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", text)
    # Linux/lbfgs stock failures include process-local 0x addresses and
    # rtol-scale floats; those must not look like a new patched mechanism.
    text = re.sub(r"0x[0-9a-fA-F]+", "0x<addr>", text)
    text = re.sub(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", "<float>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def confirm_candidate_tests(
    baseline: TestResult,
    checkout: Path,
    *,
    first: TestResult | None = None,
    runner=None,
    max_attempts: int = 2,
) -> tuple[TestResult, Decision | None, list[TestResult]]:
    """Require a new ordinary pytest failure to reproduce once.

    A single frozen FEDOT test is order/data dependent and occasionally fails
    only in the full suite.  Treating one non-reproducible node as a deterministic
    patch regression wastes the revision feedback loop.  Infrastructure errors
    remain fail-closed and are never hidden by retries.
    """

    run = runner or measure_fedot_tests
    attempts: list[TestResult] = []
    current = normalize_test_result(first if first is not None else run(checkout))
    while True:
        attempts.append(current)
        blocked = tests_regressed(baseline, current)
        if blocked is None or blocked.infrastructure_error:
            return current, blocked, attempts
        if len(attempts) >= max(1, max_attempts):
            return current, blocked, attempts
        current = normalize_test_result(run(checkout))


def verdict(
    stock: dict[str, ScoreResult],
    patched: dict[str, ScoreResult],
    *,
    lift_ids: tuple[str, ...] | None = None,
    protect_ids: tuple[str, ...] | None = None,
    evidence_only: bool = False,
) -> Decision:
    exam_lift, exam_protect = hidden_exam()
    lift_ids = lift_ids or exam_lift
    protect_ids = protect_ids or exam_protect
    spec = load_task(lift_ids[0])
    return compare_pack(
        stock,
        patched,
        lift_ids=lift_ids,
        protect_ids=protect_ids,
        higher_is_better=spec.higher_is_better,
        min_delta=spec.min_delta,
        sentinel=spec.sentinel,
        evidence_only=evidence_only,
    )
