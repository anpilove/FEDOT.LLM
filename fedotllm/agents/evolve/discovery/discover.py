"""Walk FEDOT source for a patch site. No tests, no exam, no gym."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from fedotllm.agents.evolve.discovery.context import (
    inspect_trace,
)
from fedotllm.agents.evolve.storage.replay import semantic_site_id
from fedotllm.agents.evolve.discovery.invariants import (
    invariant_leads,
    row_identity_leads,
)
from fedotllm.agents.evolve.discovery.registry import registry_leads
from fedotllm.agents.evolve.discovery.repo_map import (
    _area,
    fit_neighborhood,
    in_metric_scan,
    looks_metric,
    repo_map,
)
from fedotllm.agents.evolve.discovery.signals import (
    DEFAULT_PYTEST_TIMEOUT_S,
    collect_lint,
    failed_pytest_nodes,
    format_lint_for_llm,
    lint_leads,
    parse_lint,
    parse_pytest_output,
    pytest_contract_source,
    pytest_failure_excerpt,
    pytest_result,
    pytest_snapshot,
)
from fedotllm.agents.evolve.discovery.selection import (
    SiteProposal,
    _compact_line_numbers,
    _execution_causal_priority,
    _impact,
    _llm_pick,
    _metric_path,
    _unique,
    annotate_pool_rows,
    localization,
    pool_rows,
)
from fedotllm.agents.evolve.types import PatchSite, ScoreResult

__all__ = [
    "DEFAULT_PYTEST_TIMEOUT_S",
    "SiteProposal",
    "annotate_pool_rows",
    "collect_lint",
    "default_parameter_leads",
    "discover_leads",
    "failed_pytest_nodes",
    "format_lint_for_llm",
    "leads_from_scores",
    "lint_leads",
    "localization",
    "parse_lint",
    "parse_pytest_output",
    "pool_rows",
    "pytest_contract_source",
    "pytest_failure_excerpt",
    "pytest_result",
    "pytest_snapshot",
]

EXCLUDED_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "docs",
    "examples",
    "jupyter_notebooks",
    "caching",
    "visualisation",
    "visualization",
    "explainability",
    "remote",
    "structural_analysis",
}
_LINT_LIMIT = 40
_FILE_LIMIT = 400
_TEST_LEAD_LIMIT = 8
LOGGING_VERSION = 2


def _lead_name(lead: PatchSite) -> str:
    token = (lead.why or "").strip().split()[-1] if (lead.why or "").strip() else ""
    return token.rsplit(".", 1)[-1] if token else ""


def _rank_leads(leads: list[PatchSite]) -> list[PatchSite]:
    return sorted(leads, key=lambda lead: (-_impact(lead), lead.file_path, lead.line))


def _spread_leads(
    leads: list[PatchSite], *, limit: int | None = None
) -> list[PatchSite]:
    buckets: dict[str, list[PatchSite]] = {}
    for lead in leads:
        buckets.setdefault(_area(lead.file_path), []).append(lead)
    out: list[PatchSite] = []
    cap = len(leads) if limit is None else max(1, limit)
    while len(out) < cap:
        progressed = False
        for key in list(buckets):
            group = buckets[key]
            if not group:
                continue
            out.append(group.pop(0))
            progressed = True
            if len(out) >= cap:
                return out
        if not progressed:
            break
    return out


def _order_leads(leads: list[PatchSite]) -> list[PatchSite]:
    preferred = ("fit", "transform", "predict", "predict_proba")
    by_file: dict[str, list[PatchSite]] = {}
    for lead in leads:
        by_file.setdefault(lead.file_path, []).append(lead)
    picked: list[PatchSite] = []
    for group in by_file.values():
        names = {_lead_name(item): item for item in group}
        chosen = next((names[name] for name in preferred if name in names), group[0])
        picked.append(chosen)
    tiers: dict[int, list[PatchSite]] = {}
    for lead in picked:
        tiers.setdefault(-_impact(lead), []).append(lead)
    ordered: list[PatchSite] = []
    for key in sorted(tiers):
        ordered.extend(_spread_leads(tiers[key]))
    return ordered


def _crash_why(result: ScoreResult) -> str:
    """Exception text only. Never the exam task_id."""

    detail = (result.detail or "").strip().splitlines()
    if detail:
        return detail[0][:200]
    for line in reversed((result.traceback or "").splitlines()):
        text = line.strip()
        if (
            text
            and not text.startswith("File ")
            and "Traceback" not in text
            and 'File "' not in text
        ):
            return text[:200]
    return "crash"


def leads_from_scores(
    stock: dict[str, ScoreResult] | None,
    checkout: Path,
    *,
    operation_hints: dict[str, tuple[str, ...]] | None = None,
) -> list[PatchSite]:
    """Localize from execution evidence. Ignores dict keys (those are harness ids)."""

    if not stock:
        return []
    leads: list[PatchSite] = []
    seen: set[tuple[str, int]] = set()
    for task_key, result in stock.items():
        if result.status != "crash":
            continue
        why = _crash_why(result)
        operation_evidence: list[str] = []
        operation_rows: list[tuple[str, dict]] = []
        hinted_operations = (operation_hints or {}).get(task_key, ())
        for operation in hinted_operations:
            compact = operation.lower().replace("_", "").replace("-", "")
            match = next(
                (
                    row
                    for row in result.coverage
                    if compact
                    and compact
                    in str(row.get("symbol") or "")
                    .lower()
                    .replace("_", "")
                    .replace("-", "")
                    and str(row.get("file_path") or "").startswith("fedot/")
                ),
                None,
            )
            if match is None:
                operation_evidence.append(f"workload operation: {operation}")
            else:
                operation_rows.append((operation, match))
                operation_evidence.append(
                    "workload operation "
                    f"{operation}: {match['file_path']}:{int(match.get('line') or 1)} "
                    f"({match.get('symbol') or '<symbol>'})"
                )
        hinted_compact = {
            operation.lower().replace("_", "").replace("-", "")
            for operation in hinted_operations
        }
        for row in result.dataflow:
            operation = str(row.get("operation") or "")
            compact = operation.lower().replace("_", "").replace("-", "")
            if compact not in hinted_compact:
                continue
            input_snapshot = row.get("input") or {}
            output_snapshot = row.get("output") or {}
            operation_evidence.append(
                "runtime data flow "
                f"{operation}/{row.get('stage') or 'unknown'}: "
                f"input={json.dumps(input_snapshot, sort_keys=True)}; "
                f"output={json.dumps(output_snapshot, sort_keys=True)}"
            )
        frames = inspect_trace(result.traceback or "", checkout=checkout)
        frame_chain = "\n".join(
            f"{frame['file']}:{frame['line']} in {frame['func']}"
            for frame in frames
            if frame["file"].startswith("fedot/")
        )
        shared_evidence = (
            f"stock runtime crash: {why}",
            f"FEDOT frame chain:\n{frame_chain}",
            *operation_evidence,
        )
        # Tracebacks identify where invalid data was consumed. For a pipeline
        # workload the quality-changing cause is frequently the upstream
        # operation that produced the data/metadata, so rank its actually
        # executed implementation before the downstream exception leaf.
        for operation, row in operation_rows:
            rel = str(row["file_path"])
            line = int(row.get("line") or 1)
            key = (rel, line)
            if key in seen:
                continue
            seen.add(key)
            leads.append(
                PatchSite(
                    channel="operation",
                    file_path=rel,
                    line=line,
                    why=f"executed workload operation {operation} before downstream crash",
                    evidence=shared_evidence,
                    signals=("executed", "upstream_of_crash"),
                )
            )
            if len(leads) >= _TEST_LEAD_LIMIT:
                return leads
        for frame in reversed(frames):
            if not frame["file"].startswith("fedot/"):
                continue
            key = (frame["file"], int(frame["line"]))
            if key in seen:
                continue
            seen.add(key)
            leads.append(
                PatchSite(
                    channel="trace",
                    file_path=frame["file"],
                    line=int(frame["line"]),
                    why=f"{why} in {frame['func']}",
                    evidence=shared_evidence,
                )
            )
            if len(leads) >= _TEST_LEAD_LIMIT:
                return leads
    return leads


def default_parameter_leads(
    checkout: Path,
    operation_hints: dict[str, tuple[str, ...]],
    *,
    scores: dict[str, ScoreResult] | None = None,
) -> list[PatchSite]:
    """Expose defaults of actually evaluated operations as quality levers.

    This contains no DEV/FINAL values and no case labels.  It only connects the
    frozen workload's public operation ids to FEDOT's own default repository.
    """

    rel = "fedot/core/repository/data/default_operation_params.json"
    path = checkout / rel
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError):
        return []
    registry_rows: dict[str, tuple[str, int, dict]] = {}
    repository_dir = checkout / "fedot/core/repository/data"
    for registry_name in (
        "model_repository.json",
        "data_operation_repository.json",
        "gpu_models_repository.json",
    ):
        registry_path = repository_dir / registry_name
        try:
            registry_text = registry_path.read_text(encoding="utf-8")
            registry_payload = json.loads(registry_text)
        except (OSError, json.JSONDecodeError):
            continue
        operations = registry_payload.get("operations") or {}
        registry_lines = registry_text.splitlines()
        for operation, metadata in operations.items():
            pattern = re.compile(rf'^\s*"{re.escape(operation)}"\s*:')
            line = next(
                (
                    index
                    for index, text in enumerate(registry_lines, start=1)
                    if pattern.search(text)
                ),
                1,
            )
            registry_rows.setdefault(
                operation,
                (
                    f"fedot/core/repository/data/{registry_name}",
                    line,
                    metadata if isinstance(metadata, dict) else {},
                ),
            )

    tasks_by_operation: dict[str, list[str]] = {}
    for task_id, operations in operation_hints.items():
        for operation in operations:
            if operation in payload or operation in registry_rows:
                tasks_by_operation.setdefault(operation, []).append(task_id)
    runtime_by_operation: dict[str, list[str]] = {}
    for task_id, operations in operation_hints.items():
        result = (scores or {}).get(task_id)
        if result is None:
            continue
        wanted = set(operations)
        for row in result.dataflow:
            operation = str(row.get("operation") or "")
            if operation not in wanted:
                continue
            implementation = str(row.get("implementation") or "<implementation>")
            effective = json.dumps(
                row.get("params") or {},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            supported = ", ".join(
                str(name)
                for name in (
                    row.get("supported_parameters") or (row.get("params") or {}).keys()
                )
            )
            estimator_defaults = json.dumps(
                row.get("estimator_defaults") or {},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            text = f"runtime implementation: {implementation}; effective params: {effective}"
            if supported:
                text += f"; supported parameters: [{supported}]"
            if estimator_defaults != "{}":
                text += f"; estimator defaults: {estimator_defaults}"
            bucket = runtime_by_operation.setdefault(operation, [])
            if text not in bucket:
                bucket.append(text)
    leads: list[PatchSite] = []
    for operation, task_ids in sorted(
        tasks_by_operation.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if operation in payload:
            file_path = rel
            pattern = re.compile(rf'^\s*"{re.escape(operation)}"\s*:')
            line = next(
                (
                    index
                    for index, text in enumerate(lines, start=1)
                    if pattern.search(text)
                ),
                1,
            )
            current = payload[operation]
            registry_evidence = ""
        else:
            _registry_file, _registry_line, metadata = registry_rows[operation]
            # Repository JSON is descriptive metadata. A missing estimator
            # default must be inserted into the one file consumed by
            # DefaultOperationParamsRepository, not into the operation catalog.
            file_path, line = rel, len(lines)
            current = {}
            compact_metadata = {
                key: metadata[key]
                for key in ("meta", "presets", "tags")
                if key in metadata
            }
            registry_evidence = "operation registry metadata: " + json.dumps(
                compact_metadata,
                ensure_ascii=False,
                sort_keys=True,
            )
        leads.append(
            PatchSite(
                channel="configuration",
                file_path=file_path,
                line=line,
                why=f"default parameters for executed operation {operation}",
                evidence=tuple(
                    filter(
                        None,
                        (
                            f"executed operation: {operation}",
                            f"number of frozen workloads using this operation: {len(task_ids)}",
                            "current FEDOT defaults: "
                            + json.dumps(current, ensure_ascii=False, sort_keys=True),
                            (
                                "No explicit entry exists in "
                                f"{rel}; a supported mild default may be added there."
                                if operation not in payload
                                else ""
                            ),
                            registry_evidence,
                            *runtime_by_operation.get(operation, ())[:4],
                            "Explicit user parameters must remain authoritative; DEV judges only the default path.",
                        ),
                    )
                ),
                signals=("executed", "operation_defaults"),
            )
        )
    return leads


def static_leads(
    checkout: Path, *, rank_metadata_stale: bool = False
) -> list[PatchSite]:
    """Registry files first, then the rest of runtime fedot/. One site per file."""

    structural = [
        lead for lead in registry_leads(checkout) if _metric_path(lead.file_path)
    ]
    if not structural:
        tagged = fit_neighborhood(checkout, limit=_FILE_LIMIT)
        if not tagged:
            tagged = [
                (item, ("reachable",))
                for item in repo_map(checkout, (), limit=_FILE_LIMIT)
                if _metric_path(item.file_path)
            ]
        for symbol, sigs in tagged:
            if not in_metric_scan(symbol.file_path) or not looks_metric(
                checkout, symbol
            ):
                continue
            structural.append(
                PatchSite(
                    channel="repo_map",
                    file_path=symbol.file_path,
                    line=symbol.line,
                    why=f"{symbol.kind} {symbol.parent + '.' if symbol.parent else ''}{symbol.name}",
                    signals=sigs,
                )
            )
    structural = _unique([*row_identity_leads(checkout), *structural])
    seen = {lead.file_path for lead in structural}
    structural.extend(_extra_fedot_sites(checkout, seen))
    tree_files = [lead.file_path for lead in structural]
    if not rank_metadata_stale:
        return _order_leads(structural)
    from fedotllm.agents.evolve.discovery.repo_map import Symbol

    tree = [Symbol(path, "", "file", 1) for path in tree_files]
    hints = invariant_leads(checkout, tree)
    return _order_leads(_unique([*hints, *structural]))


def _extra_fedot_sites(checkout: Path, seen: set[str]) -> list[PatchSite]:
    """Runtime fedot/ files not already in the registry/fallback catalog."""

    from fedotllm.agents.evolve.discovery.repo_map import _py_files

    out: list[PatchSite] = []
    for path in _py_files(checkout):
        rel = path.relative_to(checkout).as_posix()
        if rel in seen:
            continue
        out.append(
            PatchSite(
                channel="core_scan",
                file_path=rel,
                line=1,
                why="file",
                signals=("core_scan",),
            )
        )
    return out


def discover_leads(
    checkout: Path,
    *,
    inference=None,
    limit: int = _LINT_LIMIT,
    max_picks: int | None = None,
    trace: dict | None = None,
    execution: list[dict] | None = None,
    trace_leads: list[PatchSite] | None = None,
    max_actions: int | None = None,
    max_runs_per_file: int = 2,
    excluded_files: set[str] | None = None,
    excluded_sites: set[tuple[str, int]] | None = None,
    prior_hypotheses: list[dict] | None = None,
    excluded_semantic_sites: set[str] | None = None,
    present_full_catalog: bool = False,
    on_pick: Callable[[list[PatchSite]], None] | None = None,
) -> list[PatchSite]:
    pooled = static_leads(checkout)
    executed_by_file: dict[str, PatchSite] = {}
    # File-level metric reachability must come only from function/method bodies
    # that actually ran.  A class statement executes all nested ``def`` lines
    # during import, which previously let Scout select an untouched sibling
    # method and call it metric-linked.
    executed_metric_lines: dict[str, set[int]] = {}
    for row in execution or []:
        rel = str(row.get("file_path") or "")
        kind = str(row.get("kind") or "")
        if kind not in {"method", "function"} or row.get("body_executed") is False:
            continue
        lines = executed_metric_lines.setdefault(rel, set())
        for pair in row.get("line_ranges") or ():
            try:
                first, last = int(pair[0]), int(pair[1])
            except (IndexError, TypeError, ValueError):
                continue
            lines.update(range(first, last + 1))
    for row in execution or []:
        rel = str(row.get("file_path") or "")
        try:
            line = int(row.get("line") or 1)
            count = int(row.get("count") or 0)
        except (TypeError, ValueError):
            continue
        if not rel.startswith("fedot/") or not (checkout / rel).is_file():
            continue
        ranges: list[str] = []
        for pair in row.get("line_ranges") or ():
            try:
                first, last = int(pair[0]), int(pair[1])
            except (IndexError, TypeError, ValueError):
                continue
            ranges.append(str(first) if first == last else f"{first}-{last}")
        symbol_kind = str(row.get("kind") or "symbol")
        # A function definition line is executed when its module is imported.
        # Keep it in the catalog for reachability, but do not rank it as a
        # metric-bearing call unless at least one body statement ran.
        if symbol_kind in {"method", "function"} and row.get("body_executed") is False:
            symbol_kind = "definition"
        runtime = str(row.get("runtime") or "")
        candidate = PatchSite(
            channel="execution",
            file_path=rel,
            line=line,
            why=f"executed symbol {row.get('symbol') or '<module>'}",
            evidence=tuple(
                filter(
                    None,
                    (
                        f"runtime line hits: {count}",
                        f"executed lines in this symbol: {','.join(ranges)}"
                        if ranges
                        else "",
                        f"runtime symbol kind: {symbol_kind}",
                        runtime if symbol_kind != "definition" else "",
                        str(row.get("workload") or "")
                        if symbol_kind != "definition"
                        else "",
                    ),
                )
            ),
            signals=("executed", symbol_kind),
        )
        current = executed_by_file.get(rel)
        current_count = (
            int(current.evidence[0].rsplit(" ", 1)[-1]) if current is not None else -1
        )
        if current is None:
            executed_by_file[rel] = candidate
        else:
            candidate_specific = symbol_kind in {"method", "function"}
            current_specific = any(
                item in {"method", "function"} for item in current.signals
            )
            winner = (
                candidate
                if (candidate_specific, count) > (current_specific, current_count)
                else current
            )
            same_symbol = current.why == candidate.why
            if not same_symbol:
                # Coverage also records module/class definitions reached only
                # because a file was imported.  Once a metric-bearing method
                # wins this file, attaching workloads and operation traces from
                # those unrelated symbols invents causal links and can multiply
                # the Scout prompt size.  Evidence from repeated executions of
                # the same symbol is still merged below.
                executed_by_file[rel] = winner
                continue
            workload_evidence = tuple(
                dict.fromkeys(
                    item
                    for item in (*current.evidence, *candidate.evidence)
                    if not item.startswith(
                        (
                            "runtime line hits:",
                            "executed lines in this symbol:",
                            "runtime symbol kind:",
                        )
                    )
                )
            )
            winner_runtime = tuple(
                item
                for item in winner.evidence
                if item.startswith(
                    (
                        "runtime line hits:",
                        "executed lines in this symbol:",
                        "runtime symbol kind:",
                    )
                )
            )
            merged_evidence = (*winner_runtime, *workload_evidence)
            executed_by_file[rel] = PatchSite(
                channel=winner.channel,
                file_path=winner.file_path,
                line=winner.line,
                why=winner.why,
                evidence=merged_evidence,
                signals=winner.signals,
            )
    executed: list[PatchSite] = []
    for rel, lead in executed_by_file.items():
        metric_lines = executed_metric_lines.get(rel) or set()
        file_evidence = (
            f"executed metric-bearing lines in file: {_compact_line_numbers(metric_lines)}"
            if metric_lines
            else ""
        )
        executed.append(
            PatchSite(
                channel=lead.channel,
                file_path=lead.file_path,
                line=lead.line,
                why=lead.why,
                evidence=tuple(filter(None, (*lead.evidence, file_evidence))),
                signals=lead.signals,
            )
        )
    if executed:
        executed.sort(
            key=lambda lead: (
                _execution_causal_priority(lead),
                -int(lead.evidence[0].rsplit(" ", 1)[-1]),
            )
        )
        pooled = _unique([*executed, *pooled])
    if trace_leads:
        urgent = [
            lead
            for lead in trace_leads
            if any(item.startswith("stock runtime crash:") for item in lead.evidence)
            or "observed_contract_failure" in lead.signals
        ]
        exploratory = [lead for lead in trace_leads if lead not in urgent]
        # A measured crash remains first.  Otherwise, prefer source that was
        # actually executed by the quality workloads over generic parameter
        # opportunities.  Previously one shared defaults JSON dominated every
        # campaign even after many distinct scalar proposals were rejected.
        pooled = _unique([*urgent, *pooled, *exploratory])
    excluded = set(excluded_files or ())
    if excluded:
        pooled = [lead for lead in pooled if lead.file_path not in excluded]
    excluded_exact = set(excluded_sites or ())
    if excluded_exact:
        pooled = [
            lead for lead in pooled if (lead.file_path, lead.line) not in excluded_exact
        ]
    excluded_semantic = set(excluded_semantic_sites or ())
    if excluded_semantic:
        pooled = [
            lead for lead in pooled if semantic_site_id(lead) not in excluded_semantic
        ]
    if trace is not None:
        hits = invariant_leads(checkout)
        trace["pool_rows_static"] = pool_rows(pooled)
        trace["metadata_stale_hits"] = [
            {"file_path": hit.file_path, "line": hit.line, "why": hit.why}
            for hit in hits
        ]
        trace["execution_symbols"] = execution or []
        trace["crash_leads"] = [
            {
                "file_path": lead.file_path,
                "line": lead.line,
                "why": lead.why,
                "evidence": list(lead.evidence),
            }
            for lead in trace_leads or []
        ]
        trace["cross_run_excluded_files"] = sorted(excluded)
        trace["cross_run_excluded_sites"] = [
            {"file_path": file_path, "line": line}
            for file_path, line in sorted(excluded_exact)
        ]
        trace["cross_run_excluded_semantic_sites"] = sorted(excluded_semantic)
        trace["llm_pick"] = None
        trace["llm_picks"] = []
        trace["llm_pick_raw"] = None
    if inference is not None:
        want = (
            max_picks
            if max_picks is not None
            else int(os.environ.get("EVOLVE_AGENT_MAX_LEADS", "3"))
        )
        found = _llm_pick(
            inference,
            checkout,
            pooled,
            max_picks=max(1, want),
            trace=trace,
            max_actions=max_actions,
            max_runs_per_file=max_runs_per_file,
            excluded_sites=excluded_exact,
            prior_hypotheses=prior_hypotheses,
            excluded_semantic_sites=excluded_semantic,
            present_full_catalog=present_full_catalog,
            on_pick=on_pick,
        )
        if trace is not None:
            trace["llm_picks"] = [
                {
                    "file_path": item.file_path,
                    "line": item.line,
                    "why": item.why,
                    "mechanism": item.mechanism,
                    "proposed_change": item.proposed_change,
                    "expected_metric_effect": item.expected_metric_effect,
                }
                for item in found
            ]
            if found:
                first = found[0]
                trace["llm_pick"] = {
                    "file_path": first.file_path,
                    "line": first.line,
                    "why": first.why,
                    "mechanism": first.mechanism,
                    "proposed_change": first.proposed_change,
                    "expected_metric_effect": first.expected_metric_effect,
                }
        for picked in reversed(found):
            if not _metric_path(picked.file_path):
                continue
            for lead in pooled:
                if lead.file_path == picked.file_path and lead.line == picked.line:
                    picked = PatchSite(
                        channel="llm",
                        file_path=picked.file_path,
                        line=picked.line,
                        why=picked.why or lead.why,
                        evidence=lead.evidence,
                        signals=lead.signals,
                        mechanism=picked.mechanism,
                        proposed_change=picked.proposed_change,
                        expected_metric_effect=picked.expected_metric_effect,
                        hypothesis_kind=picked.hypothesis_kind,
                    )
                    break
            pooled = _unique([picked] + pooled)
        # The catalog is context, not an approved hypothesis queue. Otherwise
        # skip/reject (or an exhausted Scout budget) silently reintroduces the
        # same static entries into Verifier and spends the remaining budget.
        selected = {(lead.file_path, lead.line) for lead in found}
        pooled = [lead for lead in pooled if (lead.file_path, lead.line) in selected]
    return pooled[: max(1, limit)]


_FILE_COVERAGE_PREFIX = "executed metric-bearing lines in file:"
