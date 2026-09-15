from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from fedotllm.log import logger
from fedotllm.llm import LLMRequestTimeout
from fedotllm.agents.evolve.evaluation.affected_eval import (
    affected_feedback,
    confirm_affected_metric,
    evaluate_affected_metric,
)
from fedotllm.agents.evolve.execution.checkout import (
    create_experiment_checkout,
    discard_experiment_checkout,
    resolve_fedot_src,
    snapshot_diff,
    source_commit,
    source_fingerprint,
)
from fedotllm.agents.evolve.discovery.discover import (
    LOGGING_VERSION,
    annotate_pool_rows,
    default_parameter_leads,
    leads_from_scores,
    localization,
    pool_rows,
    pytest_failure_excerpt,
    pytest_contract_source,
)
from fedotllm.agents.evolve.discovery.contracts import (
    discover_contract_violations,
    supports_public_contracts,
    verification_from_contract_lead,
)
from fedotllm.agents.evolve.evaluation.eval import run_stock
from fedotllm.agents.evolve.evaluation.independent_data import PROTOCOL as INDEPENDENT_DATA_PROTOCOL
from fedotllm.agents.evolve.storage.findings import append_final_outcome, begin_run
from fedotllm.agents.evolve.agents.fixer import fix_lead
from fedotllm.agents.evolve.agents.failures import AgentModelFailure
from fedotllm.agents.evolve.discovery.targets import apply_verified_target
from fedotllm.agents.evolve.execution.guard import guard_path
from fedotllm.agents.evolve.storage.hypothesis import (
    Hypothesis,
    as_row,
    behavior_probe_fingerprint,
    normalized_patch_hash,
)
from fedotllm.agents.evolve.storage.journal import append_journal
from fedotllm.agents.evolve.storage.checkpoint import save_checkpoint
from fedotllm.agents.evolve.storage.run_budget import EvolveBudgetExhausted
from fedotllm.agents.evolve.controller.measurement_budget import MeasurementBudget
from fedotllm.agents.evolve.evaluation.manifest import verify_manifest
from fedotllm.agents.evolve.evaluation.judge import (
    confirm_candidate_tests,
    measure_fedot_tests,
    measure_baseline_fedot_tests,
    measure_patched,
    measure_stock,
    normalize_test_result,
    tests_regressed,
    verdict,
)
from fedotllm.agents.evolve.protocol import (
    acceptance_protocol_fingerprint,
    score_protocol_fingerprint,
)
from fedotllm.agents.evolve.storage.replay import (
    patch_feedback_from_findings,
    recent_completed_semantic_sites_from_findings,
    recent_completed_sites_from_findings,
    recent_completed_hypotheses_from_findings,
    rejected_probe_hashes_from_findings,
    skip_tried,
    solved_lift_tasks_from_findings,
    tried_patch_hashes_from_findings,
    tried_sites,
)
from fedotllm.agents.evolve.execution.run_code import run_fedot_snippet
from fedotllm.agents.evolve.storage.scoreboard import append_attempt
from fedotllm.agents.evolve.agents.scout import scout
from fedotllm.agents.evolve.execution.smoke import import_error
from fedotllm.agents.evolve.controller.feedback import (
    _blocking_lift_crash_ids,
    _candidate_confirmation_scope,
    _candidate_patch_text,
    _compact_reproduction_feedback,
    _dev_feedback,
    _revision_feedback_context,
    _runtime_operation_evidence,
    _test_result_diagnostics,
)
from fedotllm.agents.evolve.controller.confirmation import (
    affected_metric_moved,
    confirm_dev,
    quick_quality_screen,
)
from fedotllm.agents.evolve.controller.artifacts import _record_attempt, _score_log
from fedotllm.agents.evolve.controller.probes import (
    _behavior_probe_blocks_candidate as behavior_probe_blocks_candidate,
)
from fedotllm.agents.evolve.controller.probes import (
    compare_behavior_probe as run_behavior_probe,
    compare_with_prior_probe,
)
from fedotllm.agents.evolve.controller.finalization import (
    record_final,
    record_final_skipped,
)
from fedotllm.agents.evolve.controller.quality_queue import (
    enqueue_quality_job,
    queue_priority,
)
from fedotllm.agents.evolve.controller.localization import (
    hydrate_configuration_surfaces,
    start_revision,
)
from fedotllm.agents.evolve.controller.patches import (
    candidate_diff,
    render_candidate_patch,
)
from fedotllm.agents.evolve.controller.ablation import ablate_candidate
from fedotllm.agents.evolve.controller.protection import (
    enforce_global_dev_protect as run_global_dev_protect,
)
from fedotllm.agents.evolve.controller.session import finish_campaign
from fedotllm.agents.evolve.model_contract import build_model_contract
from fedotllm.agents.evolve.evaluation.tasks import (
    coverage_task_limit,
    hidden_exam,
    load_task,
    quality_suite,
)
from fedotllm.agents.evolve.types import (
    Decision,
    EvolveAgentConfig,
    EvolveRunPolicy,
    PatchCandidate,
    PatchSite,
    ScoreResult,
    VerificationResult,
)
from fedotllm.agents.evolve.agents.verifier import (
    is_controller_observed_crash,
    replay_reproduction,
    verification_from_observed_crash,
    verification_context,
    verify_lead,
)


def compare_behavior_probe(source: Path, experiment: Path, code: str) -> dict:
    """Run a causal probe using the campaign's replaceable snippet runner."""

    return run_behavior_probe(
        source,
        experiment,
        code,
        run_snippet_fn=run_fedot_snippet,
    )


def _behavior_probe_blocks_candidate(result: dict) -> bool:
    return behavior_probe_blocks_candidate(result)


def _controller_crashes_for_lead(
    stock: dict[str, ScoreResult], lead: PatchSite
) -> tuple[str, ...]:
    """Select stock crashes that independently executed the proposed site."""

    target = Path(lead.file_path).as_posix()
    matched: list[str] = []
    for task_id, result in stock.items():
        if result.status != "crash":
            continue
        traceback = (result.traceback or "").replace("\\", "/")
        covered = any(
            Path(str(row.get("file_path") or "")).as_posix() == target
            for row in result.coverage
        )
        if target in traceback or covered:
            matched.append(task_id)
    return tuple(matched)


def _start_revision(*args, **kwargs) -> Hypothesis:
    return start_revision(*args, **kwargs)


def _hydrate_configuration_surfaces(*args, **kwargs) -> int:
    return hydrate_configuration_surfaces(
        *args,
        **kwargs,
        measure_stock_fn=measure_stock,
    )


def _candidate_diff(
    experiment: Path,
    source: Path,
    candidate: PatchCandidate,
) -> str:
    return candidate_diff(
        experiment,
        source,
        candidate,
        snapshot_diff_fn=snapshot_diff,
    )


def _render_candidate_patch(*args, **kwargs) -> str:
    return render_candidate_patch(
        *args,
        **kwargs,
        snapshot_diff_fn=snapshot_diff,
    )


def run_once(
    *,
    checkout: Path | None = None,
    inference=None,
    scout_inference=None,
    verifier_inference=None,
    fixer_inference=None,
    workspace: Path | None = None,
    findings_path: Path | None = None,
    lift_ids: tuple[str, ...] | None = None,
    protect_ids: tuple[str, ...] | None = None,
    max_leads: int | None = None,
    max_revisions: int | None = None,
    max_edits: int | None = None,
    max_actions: int | None = None,
    site_cooldown_campaigns: int | None = None,
    resume_lead: PatchSite | None = None,
    resume_verification: VerificationResult | None = None,
    resume_candidate: PatchCandidate | None = None,
    resume_feedback: str = "",
    policy: EvolveRunPolicy | None = None,
    run_id: str | None = None,
    config: EvolveAgentConfig | None = None,
) -> Decision:
    """Agent walks the library; a technical screen rejects crashes, FINAL judges quality."""

    config = config or EvolveAgentConfig()
    policy = policy or EvolveRunPolicy()
    scout_inference = inference if scout_inference is None else scout_inference
    fixer_inference = inference if fixer_inference is None else fixer_inference
    max_revisions = max_revisions if max_revisions is not None else config.max_revisions
    max_edits = max_edits if max_edits is not None else config.max_edits
    max_actions = max_actions if max_actions is not None else config.max_actions
    site_cooldown_campaigns = (
        site_cooldown_campaigns
        if site_cooldown_campaigns is not None
        else config.site_cooldown_campaigns
    )
    site_cooldown_campaigns = max(0, int(site_cooldown_campaigns))
    exam_lift, exam_protect = hidden_exam()
    requested_lift_ids = lift_ids or exam_lift
    protect_ids = protect_ids or exam_protect
    cap = (
        max_leads
        if max_leads is not None
        else int(os.environ.get("EVOLVE_AGENT_MAX_LEADS", str(config.max_hypotheses)))
    )
    source = (checkout or resolve_fedot_src()).resolve()
    # A real FEDOT campaign always retains the complete 24-workload safety
    # suite. Explicit lift/protect subsets remain useful for partial unit-test
    # trees and component fixtures, but cannot narrow production acceptance.
    if supports_public_contracts(source):
        protect_ids = tuple(dict.fromkeys((*protect_ids, *exam_protect)))
    workspace = workspace or Path(
        os.environ.get("EVOLVE_AGENT_WORK", "/tmp/evolve-agent-run")
    )
    workspace.mkdir(parents=True, exist_ok=True)
    measurement_budget = MeasurementBudget(
        max_pairs=config.max_measurement_pairs,
        reserve_final_pairs=config.reserve_final_measurement_pairs,
        max_seconds=config.max_measurement_seconds,
        reserve_final_seconds=config.reserve_final_measurement_seconds,
    )
    if config.metric_only and not config.metric_study_path:
        raise ValueError("metric-only runs require a persistent metric_study_path")
    if config.metric_only and resume_verification is not None:
        # Historical verifier contexts may be wrong; re-establish the contract.
        resume_verification = None
    model_contract = build_model_contract(
        {
            "scout": scout_inference,
            "verifier": verifier_inference,
            "fixer": fixer_inference,
        }
    )
    (workspace / "model_contract.json").write_text(
        json.dumps(model_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not model_contract["ok"]:
        return Decision(
            keep=False,
            reason="model_contract_invalid: "
            + "; ".join(model_contract["diagnostics"]),
            target_delta=None,
            stage="infrastructure",
            infrastructure_error=True,
        )
    if policy.verify_manifest:
        try:
            manifest_errors = verify_manifest(source)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifest_errors = [
                f"manifest verification failed: {type(exc).__name__}: {exc}"
            ]
        if manifest_errors:
            reason = "manifest_invalid: " + "; ".join(manifest_errors)
            (workspace / "manifest_error.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": str(source),
                        "errors": manifest_errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return Decision(
                keep=False,
                reason=reason[:1000],
                target_delta=None,
                stage="infrastructure",
                infrastructure_error=True,
            )
    journal = workspace / "journal.jsonl"
    run_id = run_id or uuid.uuid4().hex[:12]
    source_hash = source_fingerprint(source)
    score_protocol_hash = score_protocol_fingerprint()
    # Keep the legacy field name in artifacts for schema compatibility.  Its
    # value is now explicitly the stricter candidate-acceptance identity;
    # numeric score caches use ``score_protocol_hash`` independently.
    evaluation_protocol_hash = acceptance_protocol_fingerprint()
    source_commit_value = source_commit(source)
    findings_path = findings_path or (workspace / "findings.jsonl")
    solved_lift_ids = solved_lift_tasks_from_findings(
        findings_path,
        source_hash=source_hash,
    )
    lift_ids = tuple(
        task_id for task_id in requested_lift_ids if task_id not in solved_lift_ids
    )
    campaign_config = {
        "lift_ids": list(lift_ids),
        "requested_lift_ids": list(requested_lift_ids),
        "known_solved_lift_ids": sorted(set(requested_lift_ids) & solved_lift_ids),
        "protect_ids": list(protect_ids),
        "discovery_ids": list(lift_ids),
        "coverage_tasks": coverage_task_limit(),
        "max_hypotheses": cap,
        "max_revisions": max_revisions,
        "max_edits": max_edits,
        "max_actions": max_actions,
        "score_protocol_hash": score_protocol_hash,
        "evaluation_protocol_hash": evaluation_protocol_hash,
        "max_configuration_operations": config.max_configuration_operations,
        "max_signal_confirmations": config.max_signal_confirmations,
        "site_cooldown_campaigns": site_cooldown_campaigns,
        "resumed_branch": bool(resume_lead),
        "metric_only": config.metric_only,
        "metric_study_path": config.metric_study_path,
        "fedot_quality_jobs": policy.fedot_quality_jobs,
        "measurement_budget": measurement_budget.snapshot(),
        "transfer_screen_sources": config.transfer_screen_sources,
    }
    run_number = begin_run(
        findings_path,
        run_id=run_id,
        workspace=workspace,
        source=source,
        source_commit=source_commit_value,
        source_hash=source_hash,
        campaign_config=campaign_config,
    )

    patch_outcomes: dict[str, str] = {}

    def record_attempt(*args, **kwargs) -> None:
        kwargs.setdefault("score_protocol_hash", score_protocol_hash)
        kwargs.setdefault("evaluation_protocol_hash", evaluation_protocol_hash)
        kwargs.setdefault("resumed_branch", bool(resume_lead))
        _record_attempt(*args, **kwargs)
        decision = kwargs.get("decision") or (args[6] if len(args) > 6 else None)
        patch_hash = kwargs.get("patch_hash")
        if (
            patch_hash
            and isinstance(decision, Decision)
            and not decision.reason.startswith("duplicate_")
        ):
            patch_outcomes[patch_hash] = decision.reason

    previous_trace = os.environ.get("EVOLVE_AGENT_TRACE")
    previous_llm_audit = os.environ.get("EVOLVE_AGENT_LLM_AUDIT")
    os.environ["EVOLVE_AGENT_TRACE"] = str(workspace / "trace.jsonl")
    os.environ["EVOLVE_AGENT_LLM_AUDIT"] = str(workspace / "llm_calls.jsonl")
    append_journal(
        workspace / "trace.jsonl",
        {
            "event": "campaign_start",
            "run_id": run_id,
            "run_number": run_number,
            "source_hash": source_hash,
            "lift_ids": list(lift_ids),
            "protect_ids": list(protect_ids),
            "limits": {
                "hypotheses": cap,
                "revisions": max_revisions,
                "edits": max_edits,
                "actions": max_actions,
            },
        },
    )
    save_checkpoint(
        workspace,
        stage="campaign_started",
        run_id=run_id,
        campaign_config=campaign_config,
        source_hash=source_hash,
        selected_leads=[],
    )

    def finish(decision: Decision) -> Decision:
        campaign_config["measurement_budget"] = measurement_budget.snapshot()
        append_journal(
            workspace / "trace.jsonl",
            {"event": "measurement_budget", **measurement_budget.snapshot()},
        )
        return finish_campaign(
            decision,
            source=source,
            source_hash=source_hash,
            source_commit=source_commit_value,
            source_fingerprint_fn=source_fingerprint,
            scout_inference=scout_inference,
            verifier_inference=verifier_inference,
            fixer_inference=fixer_inference,
            workspace=workspace,
            findings_path=findings_path,
            run_id=run_id,
            run_number=run_number,
            campaign_config=campaign_config,
            previous_trace=previous_trace,
            previous_llm_audit=previous_llm_audit,
        )

    if not lift_ids:
        decision = Decision(
            keep=False,
            reason="no_novel_lift_tasks",
            target_delta=None,
            stage="discovery",
        )
        append_journal(
            journal,
            {
                "event": "decision",
                **asdict(decision),
                "known_solved_lift_ids": sorted(
                    set(requested_lift_ids) & solved_lift_ids
                ),
            },
        )
        return finish(decision)

    exam_ids = tuple(dict.fromkeys((*lift_ids, *protect_ids)))
    signal_confirmations_used = 0
    pool_n = max(cap * 8, 40)
    trace: dict = {}
    # Reserve half the campaign budget for repairs, but let a larger explicit
    # budget expand source exploration instead of silently stopping at 20.
    scout_action_limit = max(1, max_actions // 2)
    logger.info("evolve run stock+pick cap=%s tasks=%s", cap, len(exam_ids))
    scout_checkout = create_experiment_checkout(
        source, workspace, run_id=run_id, candidate_id="scout"
    )
    try:
        if config.metric_only:
            stock = {
                task: run_stock(
                    task, checkout=source, seed=config.dev_seed, collect_coverage=True,
                    task_override={"evaluation_protocol": INDEPENDENT_DATA_PROTOCOL},
                ) for task in exam_ids
            }
        else:
            stock = measure_stock(
                exam_ids, checkout=source, seed=config.dev_seed, collect_coverage=True,
            )
        # The suite intentionally instruments only its first task to keep the
        # normal baseline cheap. If a later task is the one that crashes, rerun
        # just that task with coverage; otherwise crash localization silently
        # loses the executed operation and collapses back to the traceback leaf.
        for task_id, result in tuple(stock.items()):
            if task_id not in lift_ids:
                continue
            if result.status != "crash" or (result.coverage and result.dataflow):
                continue
            refreshed = measure_stock(
                (task_id,),
                checkout=source,
                seed=config.dev_seed,
                collect_coverage=True,
            ).get(task_id)
            if refreshed is not None and refreshed.coverage:
                stock[task_id] = refreshed
        operation_hints: dict[str, tuple[str, ...]] = {}
        for task_id in exam_ids:
            try:
                operation_hints[task_id] = tuple(load_task(task_id).nodes)
            except (KeyError, ValueError, OSError):
                operation_hints[task_id] = ()
        # ``protect_ids`` are a safety contract, not search supervision.  Their
        # scores remain in every DEV verdict, but their crashes, coverage and
        # parameter surfaces must not steer Scout when the caller deliberately
        # excluded them from ``lift_ids`` (for example, a known benchmark case
        # in an unknown-discovery campaign).
        discovery_stock = {
            task_id: stock[task_id] for task_id in lift_ids if task_id in stock
        }
        discovery_operation_hints = {
            task_id: operation_hints.get(task_id, ()) for task_id in discovery_stock
        }
        configuration_surface_refreshes = _hydrate_configuration_surfaces(
            discovery_stock,
            discovery_operation_hints,
            source=source,
            seed=config.dev_seed,
            max_operations=config.max_configuration_operations,
        )
        stock.update(discovery_stock)
        trace["configuration_surface_refreshes"] = configuration_surface_refreshes
        execution: list[dict] = []
        for task_id, result in discovery_stock.items():
            try:
                spec = load_task(task_id)
                workload = (
                    f"workload family={spec.problem}; "
                    f"pipeline operations={','.join(spec.nodes)}; "
                    "evaluation objective withheld during discovery"
                )
            except (KeyError, ValueError, OSError):
                workload = "workload metadata unavailable"
            for row in result.coverage:
                runtime = _runtime_operation_evidence(
                    result.dataflow,
                    symbol=str(row.get("symbol") or ""),
                )
                execution.append({**row, "workload": workload, "runtime": runtime})
        crash_leads = leads_from_scores(
            discovery_stock,
            source,
            operation_hints=discovery_operation_hints,
        )
        parameter_leads = default_parameter_leads(
            source,
            discovery_operation_hints,
            scores=discovery_stock,
        )
        # A rejected hypothesis must not hide an important file or method from
        # every future campaign. Current-workspace attempts are excluded, while
        # exact sites from only the most recent completed campaign receive a
        # short cooldown. Cross-run memory permanently deduplicates exact
        # patches below. A failed scalar setting must not permanently blacklist
        # every future algorithm/default change for that entire operation.
        durable_tried = tried_sites(workspace)
        recent_site_cooldown = recent_completed_sites_from_findings(
            findings_path,
            source_hash=source_hash,
            evaluation_protocol_hash=evaluation_protocol_hash,
            score_protocol_hash=score_protocol_hash,
            campaigns=site_cooldown_campaigns,
        )
        recent_semantic_cooldown = recent_completed_semantic_sites_from_findings(
            findings_path,
            source_hash=source_hash,
            evaluation_protocol_hash=evaluation_protocol_hash,
            score_protocol_hash=score_protocol_hash,
            campaigns=site_cooldown_campaigns,
        )
        recent_hypotheses = recent_completed_hypotheses_from_findings(
            findings_path,
            source_hash=source_hash,
            score_protocol_hash=score_protocol_hash,
        )
        from fedotllm.agents.evolve.benchmark.known_mechanisms import (
            combined_known_mechanisms,
        )

        confirmed_hypotheses = combined_known_mechanisms(findings_path)
        prior_hypotheses = [*confirmed_hypotheses, *recent_hypotheses]
        trace["recent_source_hypotheses"] = prior_hypotheses
        trace["recent_site_cooldown"] = [
            {"file_path": file_path, "line": line}
            for file_path, line in sorted(recent_site_cooldown)
        ]
        trace["recent_semantic_site_cooldown"] = sorted(recent_semantic_cooldown)
        trace["confirmed_defect_hypotheses"] = confirmed_hypotheses
        contract_leads, contract_rows = (
            discover_contract_violations(scout_checkout)
            if supports_public_contracts(scout_checkout)
            else ([], [])
        )
        confirmed_contract_ids = {
            str(item.get("contract_id") or "")
            for item in confirmed_hypotheses
            if item.get("contract_id")
        }
        for row in contract_rows:
            row["known_confirmed"] = (
                str(row.get("contract_id") or "") in confirmed_contract_ids
            )
        contract_leads = [
            lead
            for lead in contract_leads
            if not any(
                item.startswith("observed contract id:")
                and item.partition(":")[2].strip() in confirmed_contract_ids
                for item in lead.evidence
            )
        ]
        trace["public_contract_checks"] = contract_rows
        if contract_rows:
            append_journal(
                journal,
                {
                    "event": "public_contract_checks",
                    "run_id": run_id,
                    "checks": contract_rows,
                    "violations": len(contract_leads),
                },
            )
        if resume_lead is not None:
            sites = [resume_lead]
            leads = [resume_lead]
            trace["scout_actions"] = 0
            trace["resumed_branch"] = True
        else:
            try:
                def persist_scout_picks(selected: list[PatchSite]) -> None:
                    save_checkpoint(
                        workspace,
                        stage="scout_pick",
                        run_id=run_id,
                        selected_leads=[asdict(item) for item in selected],
                    )

                sites = scout(
                    scout_checkout,
                    inference=scout_inference,
                    max_leads=pool_n,
                    max_picks=cap,
                    trace=trace,
                    execution=execution,
                    trace_leads=[*contract_leads, *crash_leads, *parameter_leads],
                    max_actions=scout_action_limit,
                    max_runs_per_file=2,
                    excluded_sites=durable_tried | recent_site_cooldown,
                    excluded_semantic_sites=recent_semantic_cooldown,
                    prior_hypotheses=prior_hypotheses,
                    present_full_catalog=True,
                    on_pick=persist_scout_picks,
                )
            except AgentModelFailure as exc:
                return finish(
                    Decision(
                        keep=False,
                        reason=f"llm_{exc.category}: {exc.detail}"[:500],
                        target_delta=None,
                        stage=(
                            "infrastructure" if exc.infrastructure else "model_response"
                        ),
                        infrastructure_error=exc.infrastructure,
                    )
                )
            except (EvolveBudgetExhausted, LLMRequestTimeout) as exc:
                return finish(
                    Decision(
                        keep=False,
                        reason=f"llm_infrastructure: {type(exc).__name__}: {exc}"[:500],
                        target_delta=None,
                        stage="infrastructure",
                        infrastructure_error=True,
                    )
                )
            leads = skip_tried(
                sites,
                workspace,
            )[:cap]
            # Repair reproducible contract violations before speculative metric
            # alternatives. This preserves Scout diversity while preventing a
            # small budget from being consumed by defaults before real defects.
            leads.sort(
                key=lambda item: (
                    0
                    if "upstream_of_crash" in item.signals
                    else 1
                    if item.hypothesis_kind == "correctness"
                    else 2
                )
            )
        static_rows = list(trace.get("pool_rows_static") or [])
        pick = trace.get("llm_pick")
        final_rows = annotate_pool_rows(
            pool_rows(sites), static_rows=static_rows, llm_pick=pick
        )

        def loc(lead: PatchSite | None) -> dict:
            if lead is None:
                return {}
            return localization(
                lead.file_path,
                lead.line,
                static_rows=static_rows,
                final_rows=final_rows,
                llm_pick=pick,
            )

        append_journal(
            journal,
            {
                "event": "scout",
                "schema_version": 1,
                "run_id": run_id,
                "logging_version": LOGGING_VERSION,
                "leads": [{**asdict(lead), **loc(lead)} for lead in leads],
                "pool": len(sites),
                "pool_rows_static": static_rows,
                "llm_pick": pick,
                "llm_picks": trace.get("llm_picks") or [],
                "llm_pick_raw": trace.get("llm_pick_raw"),
                "llm_pick_rounds": trace.get("llm_pick_rounds") or [],
                "scout_actions": int(trace.get("scout_actions") or 0),
                "scout_action_limit": scout_action_limit,
                "scout_budget_exhausted": bool(trace.get("scout_budget_exhausted")),
                "scout_external_stop": trace.get("scout_external_stop"),
                "pool_rows": final_rows,
                "metadata_stale_hits": trace.get("metadata_stale_hits") or [],
            },
        )
        save_checkpoint(
            workspace,
            stage="scout_complete",
            run_id=run_id,
            selected_leads=[asdict(item) for item in leads],
            scout_actions=int(trace.get("scout_actions") or 0),
        )
    finally:
        discard_experiment_checkout(scout_checkout, workspace=workspace, source=source)
    logger.info("evolve scout %s sites, stock %s tasks", len(leads), len(stock))
    if not leads:
        external_stop = str(trace.get("scout_external_stop") or "")
        failed_reads = sum(
            row.get("status") == "error" for row in trace.get("llm_pick_rounds", [])
        )
        reason = external_stop or (
            f"scout_incomplete: {failed_reads} failed reads"
            if failed_reads
            else "no_lead"
        )
        decision = Decision(
            keep=False,
            reason=reason,
            target_delta=None,
            regression_deltas={},
            stage="infrastructure" if failed_reads or external_stop else "dev",
            infrastructure_error=bool(failed_reads or external_stop),
        )
        append_journal(journal, {"event": "decision", **asdict(decision)})
        append_attempt(
            workspace,
            lead=None,
            candidate=None,
            stock=stock,
            patched=None,
            decision=decision,
        )
        _record_final_skipped(journal, reason=decision.reason)
        return finish(decision)

    baseline_tests = normalize_test_result(
        measure_baseline_fedot_tests(source, runner=measure_fedot_tests)
    )
    baseline_blocked = tests_regressed(baseline_tests, baseline_tests)
    if baseline_blocked is not None:
        append_journal(journal, {"event": "decision", **asdict(baseline_blocked)})
        append_attempt(
            workspace,
            lead=None,
            candidate=None,
            stock=stock,
            patched=None,
            decision=baseline_blocked,
        )
        _record_final_skipped(journal, reason=baseline_blocked.reason)
        return finish(baseline_blocked)

    def enforce_global_dev_protect(
        experiment: Path,
        local_stock: dict[str, ScoreResult],
        local_patched: dict[str, ScoreResult],
    ) -> tuple[Decision, dict[str, ScoreResult], dict[str, ScoreResult]]:
        return run_global_dev_protect(
            experiment,
            local_stock,
            local_patched,
            lift_ids=lift_ids,
            protect_ids=exam_protect,
            source=source,
            seed=config.dev_seed,
            journal=journal,
            measure_stock_fn=measure_stock,
            measure_patched_fn=measure_patched,
            verdict_fn=verdict,
        )

    confirmation_failures: dict[str, dict] = {}

    def finalize_dev_keep(
        experiment: Path,
        candidate: PatchCandidate,
        decision: Decision,
        *,
        confirmation_ids: tuple[str, ...] | None = None,
        verification_result: VerificationResult | None = None,
    ) -> Decision | None:
        """Confirm DEV, ablate, run FINAL once, and always discard executor."""

        try:
            if confirmation_ids is None:
                observed_signal_ids = tuple(
                    task_id
                    for task_id in exam_ids
                    if abs(float(decision.regression_deltas.get(task_id) or 0.0))
                    > 1e-12
                )
                confirmation_ids = observed_signal_ids or None
            scoped_ids = confirmation_ids or exam_ids
            scoped_lift = tuple(
                task_id for task_id in lift_ids if task_id in scoped_ids
            )
            scoped_protect = tuple(
                task_id for task_id in protect_ids if task_id in scoped_ids
            )
            if not scoped_lift:
                scoped_lift = scoped_ids
            if not scoped_protect:
                scoped_protect = scoped_ids
            append_journal(
                journal,
                {
                    "event": "confirmation_scope",
                    "candidate": candidate.candidate_id,
                    "tasks": list(scoped_ids),
                    "full_dev_tasks": list(exam_ids),
                    "reason": (
                        "observed-signal scoped confirmation; full DEV already protected every workload"
                        if confirmation_ids is not None
                        else "general code patch"
                    ),
                },
            )
            skip_dev_keep_gate = (
                decision.reason == "affected_metric_moved_pending_final"
                or policy.fedot_quality_jobs
            )
            if policy.confirm_and_ablate and not skip_dev_keep_gate:
                confirmed, confirmation = _confirm_dev(
                    source,
                    experiment,
                    scoped_ids,
                    scoped_lift,
                    scoped_protect,
                    seeds=config.confirmation_seeds,
                )
                append_journal(
                    journal,
                    {
                        "event": "dev_confirmation",
                        "candidate": candidate.candidate_id,
                        **confirmation,
                    },
                )
                if not confirmed:
                    decision.keep = False
                    decision.dev_keep = False
                    decision.reason = "dev_confirmation_failed"
                    confirmation_failures[candidate.candidate_id] = confirmation
                    promising_dir = workspace / "promising"
                    promising_dir.mkdir(parents=True, exist_ok=True)
                    promising_patch = promising_dir / f"{candidate.candidate_id}.patch"
                    promising_patch.write_text(
                        _candidate_diff(experiment, source, candidate),
                        encoding="utf-8",
                    )
                    (promising_dir / f"{candidate.candidate_id}.json").write_text(
                        json.dumps(
                            {
                                "candidate_id": candidate.candidate_id,
                                "reason": decision.reason,
                                "initial_dev_delta": decision.target_delta,
                                "confirmation": confirmation,
                            },
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    append_journal(
                        journal,
                        {
                            "event": "promising_candidate",
                            "candidate": candidate.candidate_id,
                            "path": str(promising_patch),
                            "reason": decision.reason,
                            "initial_dev_delta": decision.target_delta,
                            "confirmation": confirmation,
                        },
                    )
                    return None
                candidate = _ablate_candidate(
                    source,
                    workspace,
                    run_id,
                    candidate,
                    scoped_ids,
                    scoped_lift,
                    scoped_protect,
                    journal,
                    confirmation_seeds=config.confirmation_seeds,
                    baseline_tests=baseline_tests,
                    verification=verification_result,
                )
            if policy.evaluate_final and not policy.fedot_quality_jobs:
                from fedotllm.agents.evolve.controller.transfer import evaluate_transfer
                from fedotllm.agents.evolve.execution.patch import apply_patch
                transfer_tree = create_experiment_checkout(
                    source, workspace, run_id=run_id, candidate_id=f"transfer-{candidate.candidate_id}",
                )
                try:
                    if not apply_patch(transfer_tree, candidate):
                        decision.keep = False
                        decision.final_keep = False
                        decision.reason = "transfer_patch_replay_failed"
                        return finish(decision)
                    checks = [("dev", 42, "screen"), ("shadow", 42, "full")]
                    for transfer_split, transfer_seed, transfer_scope in checks:
                        transferred, transfer_report = evaluate_transfer(
                            source, transfer_tree, metric_plan.get("transfer"),
                            params=metric_plan.get("operation_params"), split=transfer_split,
                            seed=transfer_seed, scope=transfer_scope,
                            budget=measurement_budget,
                        )
                        append_journal(journal, {"event": "benchmark_transfer", "candidate": candidate.candidate_id,
                                                 **transfer_report})
                        if not transferred:
                            decision.keep = False
                            decision.final_keep = False
                            decision.reason = transfer_report["reason"]
                            return finish(decision)
                finally:
                    discard_experiment_checkout(transfer_tree, workspace=workspace, source=source)
            diff = _render_candidate_patch(source, workspace, run_id, candidate)
            (workspace / "final_candidate.patch").write_text(diff, encoding="utf-8")
            if policy.fedot_quality_jobs:
                job_path = enqueue_quality_job(
                    workspace,
                    candidate=candidate,
                    patch_text=diff,
                    hint="technically_valid_from_keep_path",
                    priority="high",
                    probe_status="",
                    toy_metric="priority_only",
                )
                final_decision = Decision(
                    False,
                    "queued_for_fedot_quality",
                    decision.target_delta,
                    stage="quality_queue",
                )
                append_journal(
                    journal,
                    {
                        "event": "quality_queue",
                        "candidate": candidate.candidate_id,
                        "path": str(job_path),
                        "priority": "high",
                    },
                )
            else:
                final_decision = _record_final(
                source,
                workspace,
                journal,
                candidate,
                decision,
                run_id=run_id,
                seeds=config.confirmation_seeds,
                lift_ids=lift_ids,
                protect_ids=exam_protect,
                enabled=policy.evaluate_final,
                transfer_plan=metric_plan.get("transfer"),
                operation_params=metric_plan.get("operation_params"),
                measurement_budget=measurement_budget,
            )
            if final_decision is not None:
                append_final_outcome(
                    findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    workspace=workspace,
                    decision=asdict(final_decision),
                    patch_hash=normalized_patch_hash(candidate, source_hash),
                    source_hash=source_hash,
                    score_protocol_hash=score_protocol_hash,
                    evaluation_protocol_hash=evaluation_protocol_hash,
                )
                decision.keep = final_decision.keep
                decision.final_keep = final_decision.keep
                decision.stage = "final"
                decision.reason = final_decision.reason
                decision.target_delta = final_decision.target_delta
                decision.regression_deltas = final_decision.regression_deltas
                decision.infrastructure_error = final_decision.infrastructure_error
                if final_decision.keep:
                    (workspace / "winner.patch").write_text(diff, encoding="utf-8")
            return finish(decision)
        finally:
            if experiment.exists():
                discard_experiment_checkout(
                    experiment,
                    workspace=workspace,
                    source=source,
                )

    last = Decision(
        keep=False, reason="no_patch", target_delta=None, regression_deltas={}
    )
    n_leads = len(leads)
    # Exact source changes are the durable identity of a tried experiment. A
    # future campaign may revisit the same high-leverage method with a different
    # mechanism, but must not pay to evaluate an identical normalized patch.
    historical_patch_hashes: set[str] = tried_patch_hashes_from_findings(
        findings_path,
        source_hash=source_hash,
        evaluation_protocol_hash=evaluation_protocol_hash,
    )
    seen_patches: set[str] = set(historical_patch_hashes)
    rejected_probes = rejected_probe_hashes_from_findings(
        findings_path,
        source_hash=source_hash,
        evaluation_protocol_hash=evaluation_protocol_hash,
    )
    local_seen_patches: set[str] = set()
    actions = int(trace.get("scout_actions") or 0)
    for i, lead in enumerate(leads, start=1):
        safety_extension_granted = False
        prior_behavior_probe: tuple[str, str] | None = None
        hypothesis = Hypothesis(
            id=f"h-{i}-{uuid.uuid4().hex[:8]}",
            parent_id=None,
            lead=asdict(lead),
            claim=(
                "; ".join(filter(None, (lead.mechanism, lead.proposed_change)))
                or lead.why
                or f"quality-affecting change near {lead.file_path}:{lead.line}"
            ),
            expected_effect=(
                lead.expected_metric_effect
                or "improve at least one DEV workload without regressions"
            ),
        )
        append_journal(
            workspace / "hypotheses.jsonl",
            {"event": "hypothesis", **as_row(hypothesis)},
        )
        save_checkpoint(
            workspace,
            stage="hypothesis",
            run_id=run_id,
            active_lead=asdict(lead),
            hypothesis=as_row(hypothesis),
            lead_index=i,
        )
        observed_crash = verification_from_observed_crash(lead)
        metric_plan = None
        observed_contract = verification_from_contract_lead(lead)
        if i == 1 and resume_verification is not None:
            verification = resume_verification
        elif observed_contract is not None:
            verification = observed_contract
        elif observed_crash is not None:
            verification = observed_crash
        elif verifier_inference is None or lead.hypothesis_kind != "correctness":
            verification = VerificationResult(
                "quality_hypothesis",
                claim=hypothesis.claim,
                expected=hypothesis.expected_effect,
                detail="verifier disabled by direct run_once caller",
            )
        else:
            verification_checkout = create_experiment_checkout(
                source,
                workspace,
                run_id=run_id,
                candidate_id=f"verify-{i}-{uuid.uuid4().hex[:8]}",
            )
            try:
                try:
                    verification = verify_lead(
                        verification_checkout,
                        lead,
                        inference=verifier_inference,
                        workspace=workspace,
                        max_model_calls=3,
                        correctness_only=True,
                    )
                except AgentModelFailure as exc:
                    verification = VerificationResult(
                        "infrastructure_error" if exc.infrastructure else "inconclusive",
                        claim=hypothesis.claim,
                        detail=f"llm_{exc.category}: {exc.detail}"[:500],
                    )
                except (EvolveBudgetExhausted, LLMRequestTimeout) as exc:
                    verification = VerificationResult(
                        "infrastructure_error",
                        detail=f"{type(exc).__name__}: {exc}"[:500],
                    )
            finally:
                discard_experiment_checkout(
                    verification_checkout,
                    workspace=workspace,
                    source=source,
                )
        original_lead = lead
        try:
            lead = apply_verified_target(source, lead, verification)
        except (OSError, KeyError, TypeError, ValueError, SyntaxError) as exc:
            verification = replace(
                verification, status="inconclusive",
                detail=f"verified target identity could not be preserved: {exc}",
            )
        if lead != original_lead:
            hypothesis = replace(hypothesis, lead=asdict(lead))
            append_journal(journal, {
                "event": "verified_target_refined", "hypothesis_id": hypothesis.id,
                "original_lead": asdict(original_lead), "lead": asdict(lead),
                "resolved_target": verification.resolved_target,
            })
            append_journal(workspace / "hypotheses.jsonl", {
                "event": "hypothesis_target_refined", **as_row(hypothesis),
                "original_lead": asdict(original_lead),
            })
        append_journal(
            journal,
            {
                "event": "verification",
                "run_id": run_id,
                "hypothesis_id": hypothesis.id,
                "lead": asdict(lead),
                "status": verification.status,
                "claim": verification.claim,
                "expected": verification.expected,
                "observed": verification.observed,
                "reproduction_code": verification.reproduction_code,
                "evidence": list(verification.evidence),
                "detail": verification.detail,
                "current_approach": verification.current_approach,
                "proposed_approach": verification.proposed_approach,
                "alternatives_considered": list(verification.alternatives_considered),
                "generality": verification.generality,
                "risks": list(verification.risks),
                "resolved_target": verification.resolved_target,
            },
        )
        save_checkpoint(
            workspace,
            stage="contract_verification",
            run_id=run_id,
            active_lead=asdict(lead),
            hypothesis=as_row(hypothesis),
            verification={
                "status": verification.status,
                "claim": verification.claim,
                "expected": verification.expected,
                "observed": verification.observed,
                "reproduction_code": verification.reproduction_code,
                "evidence": list(verification.evidence),
                "detail": verification.detail,
                "resolved_target": verification.resolved_target,
            },
        )
        if not verification.proceed:
            last = Decision(
                keep=False,
                reason=f"verification_{verification.status}: {verification.detail}"[
                    :500
                ],
                target_delta=None,
                stage="verification",
                infrastructure_error=verification.status == "infrastructure_error",
            )
            reproduction = {
                "status": verification.status,
                "stock": None,
                "patched": None,
                "claim": verification.claim,
                "expected": verification.expected,
                "observed": verification.observed,
                "detail": verification.detail,
            }
            record_attempt(
                journal,
                workspace,
                lead,
                None,
                stock,
                None,
                last,
                loc(lead),
                0,
                hypothesis_id=hypothesis.id,
                reproduction=reproduction,
                findings_path=findings_path,
                run_number=run_number,
                run_id=run_id,
                source_commit_value=source_commit_value,
                source_hash=source_hash,
            )
            if last.infrastructure_error:
                _record_final_skipped(journal, reason=last.reason)
                return finish(last)
            continue
        initial_feedback = resume_feedback if i == 1 else ""
        feedback_history = [initial_feedback] if initial_feedback.strip() else []
        feedback = _revision_feedback_context(feedback_history)
        verification_record = {
            "status": verification.status,
            "stock": (
                "failed_as_predicted" if verification.status == "verified_bug" else None
            ),
            "patched": None,
            "claim": verification.claim,
            "expected": verification.expected,
            "observed": verification.observed,
            "current_approach": verification.current_approach,
            "proposed_approach": verification.proposed_approach,
            "alternatives_considered": list(verification.alternatives_considered),
            "generality": verification.generality,
            "risks": list(verification.risks),
            "resolved_target": verification.resolved_target,
        }
        # One additional attempt may be unlocked only by a local KEEP that the
        # mandatory global protect suite rejects. It repairs patch scope; it is
        # not an extra random metric-search sample.
        metric_plan = {}
        if config.metric_only or policy.evaluate_final:
            from fedotllm.agents.evolve.controller.metric_study import preregister
            from fedotllm.agents.evolve.evaluation.affected_eval import extract_operation_overlays
            metric_plan = preregister(
                lead, stock, lift_ids, workspace / "metric-plans" / f"lead-{i}.json",
                operation_params=extract_operation_overlays(verification.reproduction_code),
                source=source,
                screen_sources=config.transfer_screen_sources,
            )
            append_journal(journal, {"event": "metric_target_frozen_before_patch", **metric_plan})
        retry_saved_candidate: PatchCandidate | None = None
        for revision in range(1, max(1, max_revisions) + 2):
            if revision > max(1, max_revisions) and not safety_extension_granted:
                break
            actions += 1
            if actions > max_actions:
                return finish(
                    Decision(
                        keep=False,
                        reason="controller_action_limit",
                        target_delta=None,
                        stage="infrastructure",
                        infrastructure_error=True,
                    )
                )
            logger.info(
                "evolve fix %s/%s revision %s %s:%s",
                i,
                n_leads,
                revision,
                lead.file_path,
                lead.line,
            )
            experiment_id = f"lead-{i}-r{revision}-{uuid.uuid4().hex[:8]}"
            experiment = create_experiment_checkout(
                source,
                workspace,
                run_id=run_id,
                candidate_id=experiment_id,
            )
            try:
                candidate = fix_lead(
                    experiment,
                    lead,
                    inference=fixer_inference,
                    workspace=workspace,
                    max_edits=max_edits,
                    feedback=feedback,
                    verification=verification_context(verification),
                    validate_behavior_probe=verification.status == "quality_hypothesis",
                    hypothesis_id=hypothesis.id,
                    saved_candidate=(
                        resume_candidate
                        if i == 1 and revision == 1
                        else retry_saved_candidate
                    ),
                )
                retry_saved_candidate = None
            except AgentModelFailure as exc:
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                last = Decision(
                    keep=False,
                    reason=f"llm_{exc.category}: {exc.detail}"[:500],
                    target_delta=None,
                    stage="infrastructure" if exc.infrastructure else "model_response",
                    experiment_id=experiment_id,
                    infrastructure_error=exc.infrastructure,
                )
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    None,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    hypothesis_id=hypothesis.id,
                    reproduction=verification_record,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                if exc.infrastructure:
                    return finish(last)
                break
            except (EvolveBudgetExhausted, LLMRequestTimeout) as exc:
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                return finish(
                    Decision(
                        keep=False,
                        reason=f"llm_infrastructure: {type(exc).__name__}: {exc}"[:500],
                        target_delta=None,
                        stage="infrastructure",
                        experiment_id=experiment_id,
                        infrastructure_error=True,
                    )
                )
            if candidate is None:
                reason = "no_patch"
                status = (
                    workspace
                    / "context"
                    / f"{Path(lead.file_path).stem}-{lead.line}"
                    / "fix_status.txt"
                )
                if status.is_file():
                    reason = status.read_text(encoding="utf-8").strip() or reason
                last = Decision(
                    keep=False,
                    reason=reason,
                    target_delta=None,
                    regression_deltas={},
                    experiment_id=experiment_id,
                )
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    None,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    hypothesis_id=hypothesis.id,
                    reproduction=verification_record,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                break

            save_checkpoint(
                workspace,
                stage="patch",
                run_id=run_id,
                active_lead=asdict(lead),
                hypothesis=as_row(hypothesis),
                candidate_id=candidate.candidate_id,
                candidate_artifact=f"candidates/{candidate.candidate_id}/candidate.json",
                revision=revision,
            )
            legacy_patch_hash = normalized_patch_hash(candidate, source_hash)
            patch_hash = normalized_patch_hash(
                candidate,
                source_hash,
                checkout=experiment,
            )
            candidate_patch_hashes = (patch_hash, legacy_patch_hash)
            probe_hash = behavior_probe_fingerprint(candidate.behavior_probe)
            reproduction = replay_reproduction(experiment, verification)
            save_checkpoint(
                workspace,
                stage="reproduction",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                reproduction=reproduction,
            )
            duplicate_probe = verification.status == "quality_hypothesis" and any(
                probe_hash in rejected_probes.get(candidate_hash, set())
                for candidate_hash in candidate_patch_hashes
            )
            matched_patch_hash = next(
                (
                    candidate_hash
                    for candidate_hash in candidate_patch_hashes
                    if candidate_hash in seen_patches
                ),
                patch_hash,
            )
            if matched_patch_hash in seen_patches or duplicate_probe:
                historical_feedback = (
                    patch_feedback_from_findings(
                        findings_path,
                        source_hash=source_hash,
                        patch_hash=matched_patch_hash,
                        evaluation_protocol_hash=evaluation_protocol_hash,
                    )
                    if (
                        matched_patch_hash in historical_patch_hashes or duplicate_probe
                    )
                    and matched_patch_hash not in local_seen_patches
                    else ""
                )
                prior_reason = patch_outcomes.get(matched_patch_hash, last.reason)
                duplicate_reason = (
                    "duplicate_historical_patch"
                    if historical_feedback
                    else f"duplicate_patch_after:{prior_reason}"
                    if prior_reason and prior_reason != "no_patch"
                    else "duplicate_patch"
                )
                last = Decision(
                    False,
                    duplicate_reason,
                    None,
                    experiment_id=experiment_id,
                )
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    candidate,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    hypothesis_id=hypothesis.id,
                    patch_hash=patch_hash,
                    feedback=historical_feedback,
                    reproduction=reproduction,
                    # The original finding remains the durable record. Avoid
                    # duplicating it merely because retrieval worked.
                    findings_path=None,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                local_seen_patches.update(candidate_patch_hashes)
                if historical_feedback and revision < max(1, max_revisions):
                    feedback_history.append(historical_feedback)
                    feedback = _revision_feedback_context(feedback_history)
                    hypothesis = _start_revision(
                        workspace,
                        hypothesis,
                        lead,
                        revision + 1,
                        duplicate_reason,
                    )
                    continue
                break
            seen_patches.update(candidate_patch_hashes)
            local_seen_patches.update(candidate_patch_hashes)

            behavior_probe: dict = {}
            if verification.status == "quality_hypothesis":
                behavior_probe = compare_with_prior_probe(
                    source,
                    experiment,
                    candidate.behavior_probe,
                    prior_probe=prior_behavior_probe,
                    compare_fn=compare_behavior_probe,
                )
                candidate_dir = workspace / "candidates" / candidate.candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                if behavior_probe.get("reused_probe_from_candidate"):
                    (candidate_dir / "submitted_behavior_probe.py").write_text(
                        candidate.behavior_probe, encoding="utf-8"
                    )
                    candidate.behavior_probe = behavior_probe["code"]
                    (candidate_dir / "behavior_probe.py").write_text(
                        candidate.behavior_probe, encoding="utf-8"
                    )
                if (
                    behavior_probe.get("status") == "changed"
                    and candidate.behavior_probe
                ):
                    prior_behavior_probe = (
                        candidate.candidate_id,
                        candidate.behavior_probe,
                    )
                (candidate_dir / "behavior_probe_result.json").write_text(
                    json.dumps(behavior_probe, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                append_journal(
                    workspace / "trace.jsonl",
                    {
                        "event": "behavior_probe",
                        "candidate": candidate.candidate_id,
                        **behavior_probe,
                    },
                )
                save_checkpoint(
                    workspace,
                    stage="behavior_probe",
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    revision=revision,
                    behavior_probe=behavior_probe,
                )
                behavior_status = behavior_probe.get("status")
                if _behavior_probe_blocks_candidate(behavior_probe):
                    # No metric trial has happened yet. A new diagnostic may
                    # expose the effect of these same edits; the same pair may not.
                    for candidate_hash in candidate_patch_hashes:
                        seen_patches.discard(candidate_hash)
                    rejected_probes.setdefault(patch_hash, set()).add(probe_hash)
                    reason = f"behavior_probe_{behavior_status}"
                    last = Decision(
                        False,
                        reason,
                        None,
                        experiment_id=experiment_id,
                    )
                    if behavior_status in {"invalid", "missing"}:
                        diagnosis = (
                            "The hypothesis remains unjudged because the diagnostic "
                            "test is invalid. Keep the source mechanism frozen when it "
                            "is still supported and repair only the probe. The exact "
                            "probe code, exit status, traceback, stdout, stderr, and "
                            "observations from both runs follow."
                        )
                    elif behavior_status == "no_change":
                        diagnosis = (
                            "A valid probe ran successfully on both checkouts and its "
                            "actual observations were equal. This disproves the claimed "
                            "runtime effect measured by this hypothesis; revise the "
                            "source mechanism before trying another probe."
                        )
                    else:
                        diagnosis = (
                            "Stock completed the diagnostic but the patched checkout "
                            "failed. Treat the patched traceback as a patch regression "
                            "and repair the source change."
                        )
                    feedback = (
                        f"outcome={reason}; interpretation="
                        f"{behavior_probe.get('hypothesis_result')}. {diagnosis}\n"
                        + json.dumps(behavior_probe, ensure_ascii=False)[:7_000]
                    )
                    feedback_history.append(feedback)
                    feedback = _revision_feedback_context(feedback_history)
                    if behavior_status in {"invalid", "missing"}:
                        # The test failed to judge the source hypothesis. Reuse
                        # the exact source edits and spend the next turn on a
                        # corrected diagnostic instead of silently replacing
                        # the candidate.
                        retry_saved_candidate = candidate
                    record_attempt(
                        journal,
                        workspace,
                        lead,
                        candidate,
                        stock,
                        None,
                        last,
                        loc(lead),
                        revision,
                        diff=_candidate_diff(experiment, source, candidate),
                        hypothesis_id=hypothesis.id,
                        patch_hash=patch_hash,
                        feedback=feedback,
                        reproduction=reproduction,
                        behavior_probe=behavior_probe,
                        findings_path=findings_path,
                        run_number=run_number,
                        run_id=run_id,
                        source_commit_value=source_commit_value,
                        source_hash=source_hash,
                    )
                    discard_experiment_checkout(
                        experiment, workspace=workspace, source=source
                    )
                    if revision < max_revisions:
                        hypothesis = _start_revision(
                            workspace, hypothesis, lead, revision + 1, reason
                        )
                        continue
                    break

            if (
                verification.status == "verified_bug"
                and not is_controller_observed_crash(verification)
                and reproduction.get("patched") != "resolved"
            ):
                last = Decision(
                    False,
                    "verified_bug_not_resolved",
                    None,
                    experiment_id=experiment_id,
                )
                feedback = (
                    "The independent stock reproduction still fails after your patch. "
                    "Fix the verified mechanism, not an adjacent symptom.\n"
                    + _compact_reproduction_feedback(reproduction)
                )
                feedback_history.append(feedback)
                feedback = _revision_feedback_context(feedback_history)
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    candidate,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    diff=_candidate_diff(experiment, source, candidate),
                    hypothesis_id=hypothesis.id,
                    patch_hash=patch_hash,
                    feedback=feedback,
                    reproduction=reproduction,
                    behavior_probe=behavior_probe,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                continue

            broken = next(
                (
                    error
                    for rel in dict.fromkeys(edit.file_path for edit in candidate.edits)
                    if (error := import_error(experiment, rel))
                ),
                None,
            )
            if broken:
                last = Decision(
                    False,
                    f"patch_unimportable {broken}",
                    None,
                    experiment_id=experiment_id,
                )
                feedback = (
                    f"outcome=import_error; error={broken}\n\nPrevious evaluated patch:\n"
                    + _candidate_patch_text(candidate)
                    + "\nRepair the same mechanism so every edited module imports."
                )
                feedback_history.append(feedback)
                feedback = _revision_feedback_context(feedback_history)
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    candidate,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    diff=_candidate_diff(experiment, source, candidate),
                    hypothesis_id=hypothesis.id,
                    patch_hash=patch_hash,
                    feedback=feedback,
                    reproduction=reproduction,
                    behavior_probe=behavior_probe,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                if revision >= max(1, max_revisions) and not safety_extension_granted:
                    safety_extension_granted = True
                    append_journal(
                        journal,
                        {
                            "event": "targeted_revision_extension",
                            "gate": "import",
                            "candidate": candidate.candidate_id,
                            "reason": last.reason,
                        },
                    )
                if revision < max_revisions or (
                    safety_extension_granted and revision == max(1, max_revisions)
                ):
                    hypothesis = _start_revision(
                        workspace, hypothesis, lead, revision + 1, last.reason
                    )
                    continue
                break

            # A concrete leaf-operation patch is checked for a deterministic
            # launch failure on affected workloads before pytest. Missing
            # short-horizon gain only queues the candidate; it does not DROP.
            quick_scope = None
            maintenance_screen = False
            quick_result: dict = {}
            if verification.status == "quality_hypothesis" and not config.metric_only:
                quick_scope = _candidate_confirmation_scope(
                    source,
                    candidate,
                    stock,
                    operation_hints,
                    exam_ids,
                )
            if quick_scope is not None and set(quick_scope) != set(exam_ids):
                quick_passed, quick_result = _quick_quality_screen(
                    source,
                    experiment,
                    quick_scope,
                    lift_ids,
                    protect_ids,
                    stock,
                    seed=config.dev_seed,
                )
                append_journal(
                    workspace / "trace.jsonl",
                    {
                        "event": "quick_quality_screen",
                        "candidate": candidate.candidate_id,
                        "tasks": list(quick_scope),
                        **quick_result,
                    },
                )
                if not quick_passed:
                    shadow_screen = quick_result.get("shadow") or {}
                    maintenance_screen = bool(
                        verification.status == "quality_hypothesis"
                        and quick_result.get("reason") == "no_affected_metric_signal"
                        and shadow_screen.get("evaluated")
                        and not (quick_result.get("dev") or {}).get(
                            "infrastructure_error"
                        )
                        and not shadow_screen.get("infrastructure_error")
                        and not str(
                            (quick_result.get("dev") or {}).get("reason", "")
                        ).startswith("regression")
                        and not str(shadow_screen.get("reason", "")).startswith(
                            "regression"
                        )
                    )
                if not quick_passed and not maintenance_screen:
                    quick_decision = Decision(
                        False,
                        str(quick_result.get("reason") or "no_affected_metric_signal"),
                        quick_result.get("target_delta"),
                        experiment_id=experiment_id,
                    )
                    feedback = (
                        "outcome=quick_quality_drop; stock ran on the affected "
                        "workloads, but the patch crashed, timed out, or returned "
                        "an invalid score. Short-horizon metric movement is not a "
                        "DROP. Refine the same mechanism before requesting the "
                        "full protect suite.\n"
                        + json.dumps(quick_result, ensure_ascii=False, default=str)[
                            :5_000
                        ]
                    )
                    feedback_history.append(feedback)
                    feedback = _revision_feedback_context(feedback_history)
                    record_attempt(
                        journal,
                        workspace,
                        lead,
                        candidate,
                        {task_id: stock[task_id] for task_id in quick_scope},
                        None,
                        quick_decision,
                        loc(lead),
                        revision,
                        diff=_candidate_diff(experiment, source, candidate),
                        hypothesis_id=hypothesis.id,
                        patch_hash=patch_hash,
                        feedback=feedback,
                        reproduction=reproduction,
                        behavior_probe=behavior_probe,
                        findings_path=findings_path,
                        run_number=run_number,
                        run_id=run_id,
                        source_commit_value=source_commit_value,
                        source_hash=source_hash,
                    )
                    last = quick_decision
                    discard_experiment_checkout(
                        experiment, workspace=workspace, source=source
                    )
                    if revision < max_revisions:
                        hypothesis = _start_revision(
                            workspace, hypothesis, lead, revision + 1, last.reason
                        )
                        continue
                    break

            first_tests = normalize_test_result(measure_fedot_tests(experiment))
            after_tests, blocked, test_attempts = confirm_candidate_tests(
                baseline_tests,
                experiment,
                first=first_tests,
                runner=measure_fedot_tests,
            )
            save_checkpoint(
                workspace,
                stage="pytest_gate",
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                revision=revision,
                tests={
                    "status": after_tests.status,
                    "completed": after_tests.completed,
                    "failed_nodes": sorted(after_tests.failed_nodes),
                    "duration_s": after_tests.duration_s,
                    "blocked": blocked.reason if blocked is not None else None,
                    "attempts": len(test_attempts),
                },
            )
            if len(test_attempts) > 1:
                append_journal(
                    journal,
                    {
                        "event": "pytest_gate_retry",
                        "candidate": candidate.candidate_id,
                        "attempts": [
                            {
                                "status": item.status,
                                "failed_nodes": sorted(item.failed_nodes),
                                "duration_s": item.duration_s,
                            }
                            for item in test_attempts
                        ],
                        "passed_after_retry": blocked is None,
                    },
                )
            if blocked is not None:
                blocked.experiment_id = experiment_id
                last = blocked
                new_failed_nodes = (
                    after_tests.failed_nodes - baseline_tests.failed_nodes
                )
                failed_nodes = ", ".join(sorted(new_failed_nodes)[:20])
                failure_details = pytest_failure_excerpt(
                    after_tests.output,
                    new_failed_nodes,
                    max_chars=5_000,
                )
                if not after_tests.completed:
                    failure_details = _test_result_diagnostics(
                        after_tests,
                        max_chars=5_000,
                    )
                contract_source = pytest_contract_source(
                    experiment,
                    new_failed_nodes,
                    max_chars=3_000,
                )
                feedback = (
                    f"outcome=test_failures; reason={blocked.reason}; "
                    f"failed_nodes={failed_nodes}\n"
                    "pytest_failure_details (ground truth; fix these exact contracts, "
                    "do not invent an unrelated cause):\n"
                    f"{failure_details}"
                    + (
                        "\n\nfailing_test_contract_source (exact frozen FEDOT "
                        "tests; preserve these assertions):\n"
                        f"{contract_source}"
                        if contract_source
                        else ""
                    )
                    + "\n\nPrevious evaluated patch:\n"
                    + _candidate_patch_text(candidate)
                    + "\nPreserve the causal hypothesis, but correct the implementation "
                    "and all existing FEDOT contracts before DEV evaluation."
                )
                feedback_history.append(feedback)
                feedback = _revision_feedback_context(feedback_history)
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    candidate,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    tests=after_tests,
                    hypothesis_id=hypothesis.id,
                    patch_hash=patch_hash,
                    feedback=feedback,
                    reproduction=reproduction,
                    behavior_probe=behavior_probe,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                if blocked.infrastructure_error:
                    _record_final_skipped(journal, reason=blocked.reason)
                    return finish(blocked)
                if revision >= max(1, max_revisions) and not safety_extension_granted:
                    safety_extension_granted = True
                    append_journal(
                        journal,
                        {
                            "event": "targeted_revision_extension",
                            "gate": "pytest",
                            "candidate": candidate.candidate_id,
                            "reason": last.reason,
                        },
                    )
                if revision < max_revisions or (
                    safety_extension_granted and revision == max(1, max_revisions)
                ):
                    hypothesis = _start_revision(
                        workspace, hypothesis, lead, revision + 1, last.reason
                    )
                    continue
                break

            if (
                policy.fedot_quality_jobs
                and verification.status == "quality_hypothesis"
                and not config.metric_only
            ):
                probe_status = str((behavior_probe or {}).get("status") or "")
                toy_moved = str(quick_result.get("reason") or "") == "early_gain"
                priority = queue_priority(
                    probe_status=probe_status,
                    toy_metric_moved=toy_moved,
                )
                job_path = enqueue_quality_job(
                    workspace,
                    candidate=candidate,
                    patch_text=_candidate_diff(experiment, source, candidate),
                    hint=(
                        "probe_changed"
                        if probe_status == "changed"
                        else "technically_valid_metric_unclear"
                    ),
                    priority=priority,
                    probe_status=probe_status,
                    toy_metric="early_gain" if toy_moved else "no_veto",
                )
                last = Decision(
                    False,
                    "queued_for_fedot_quality",
                    None,
                    stage="quality_queue",
                    experiment_id=experiment_id,
                )
                append_journal(
                    journal,
                    {
                        "event": "quality_queue",
                        "candidate": candidate.candidate_id,
                        "path": str(job_path),
                        "priority": priority,
                        "probe_status": probe_status,
                        "toy_metric": "priority_only",
                    },
                )
                record_attempt(
                    journal,
                    workspace,
                    lead,
                    candidate,
                    stock,
                    None,
                    last,
                    loc(lead),
                    revision,
                    diff=_candidate_diff(experiment, source, candidate),
                    tests=after_tests,
                    hypothesis_id=hypothesis.id,
                    patch_hash=patch_hash,
                    reproduction=reproduction,
                    behavior_probe=behavior_probe,
                    findings_path=findings_path,
                    run_number=run_number,
                    run_id=run_id,
                    source_commit_value=source_commit_value,
                    source_hash=source_hash,
                )
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                break

            if config.metric_only:
                from fedotllm.agents.evolve.controller.metric_study import evaluate_candidate
                candidate_dir = workspace / "candidates" / candidate.candidate_id
                try:
                    last = evaluate_candidate(
                        source, experiment, candidate, verification, metric_plan,
                        lead=lead, tasks=exam_ids, study=Path(config.metric_study_path),
                        workspace=candidate_dir, measurement_budget=measurement_budget,
                    )
                    last.experiment_id = experiment_id
                    record_attempt(
                        journal, workspace, lead, candidate, stock, None, last, loc(lead), revision,
                        diff=_candidate_diff(experiment, source, candidate), tests=after_tests,
                        hypothesis_id=hypothesis.id, patch_hash=patch_hash,
                        reproduction=reproduction, findings_path=findings_path,
                        run_number=run_number, run_id=run_id,
                        source_commit_value=source_commit_value, source_hash=source_hash,
                    )
                finally:
                    discard_experiment_checkout(experiment, workspace=workspace, source=source)
                if last.final_keep is not None or "FINAL_already_sealed" in last.reason:
                    return finish(last)
                # Only DEV feedback may lead to a revision. Held-out decisions
                # end this hypothesis; their numeric evidence never reaches Fixer.
                if last.reason.startswith("metric_DEV_rejected") and revision < max_revisions:
                    feedback = last.reason + "\nKeep the preregistered target and causal mechanism."
                    hypothesis = _start_revision(workspace, hypothesis, lead, revision + 1, last.reason)
                    continue
                break

            affected_metric: dict = {}
            if verification.status == "verified_bug" and verification.reproduction_code:
                affected_metric = evaluate_affected_metric(
                    source,
                    experiment,
                    verification,
                    lead,
                    seed=config.dev_seed,
                )
                candidate_dir = workspace / "candidates" / candidate.candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "affected_metric.json").write_text(
                    json.dumps(affected_metric, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                append_journal(
                    workspace / "trace.jsonl",
                    {
                        "event": "affected_metric",
                        "candidate": candidate.candidate_id,
                        **affected_metric,
                    },
                )
                affected_rejected = affected_metric.get("status") == "regressed"
                if affected_metric.get("status") == "improved":
                    dev_confirmation = confirm_affected_metric(
                        source,
                        experiment,
                        verification,
                        lead,
                        seeds=config.confirmation_seeds,
                        split="dev",
                        initial=affected_metric,
                    )
                    affected_metric["dev_confirmation"] = dev_confirmation
                    affected_rejected = not dev_confirmation["confirmed"]
                    (candidate_dir / "affected_metric.json").write_text(
                        json.dumps(affected_metric, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    append_journal(
                        workspace / "trace.jsonl",
                        {
                            "event": "affected_metric_confirmation",
                            "candidate": candidate.candidate_id,
                            "stage": "dev",
                            **dev_confirmation,
                        },
                    )
                if affected_rejected:
                    reason = (
                        "affected_metric_regression"
                        if affected_metric.get("status") == "regressed"
                        else "affected_metric_not_confirmed"
                    )
                    last = Decision(
                        False,
                        reason,
                        None,
                        stage="affected",
                        experiment_id=experiment_id,
                    )
                    feedback = (
                        "The patch resolves the invariant but worsens a frozen real-data "
                        "metric on a workload where the exact verified line was executed. "
                        "Try a different deterministic policy; do not optimize against "
                        "dataset values.\n"
                        + affected_feedback(affected_metric)
                        + (
                            "\nconfirmation="
                            + json.dumps(
                                {
                                    key: affected_metric["dev_confirmation"].get(key)
                                    for key in (
                                        "improved_seeds",
                                        "regressed_task_seed_pairs",
                                        "infrastructure_failures",
                                    )
                                },
                                ensure_ascii=False,
                            )
                            if affected_metric.get("dev_confirmation")
                            else ""
                        )
                        + "\n\nPrevious evaluated patch:\n"
                        + _candidate_patch_text(candidate)
                    )
                    feedback_history.append(feedback)
                    feedback = _revision_feedback_context(feedback_history)
                    record_attempt(
                        journal,
                        workspace,
                        lead,
                        candidate,
                        stock,
                        None,
                        last,
                        loc(lead),
                        revision,
                        diff=_candidate_diff(experiment, source, candidate),
                        tests=after_tests,
                        hypothesis_id=hypothesis.id,
                        patch_hash=patch_hash,
                        feedback=feedback,
                        reproduction=reproduction,
                        behavior_probe=behavior_probe,
                        affected_metric=affected_metric,
                        findings_path=findings_path,
                        run_number=run_number,
                        run_id=run_id,
                        source_commit_value=source_commit_value,
                        source_hash=source_hash,
                    )
                    discard_experiment_checkout(
                        experiment, workspace=workspace, source=source
                    )
                    if revision < max_revisions:
                        hypothesis = _start_revision(
                            workspace, hypothesis, lead, revision + 1, last.reason
                        )
                        continue
                    break

            blocking_crash_ids = _blocking_lift_crash_ids(stock, lift_ids)
            controller_crash_ids = (
                _controller_crashes_for_lead(stock, lead)
                if is_controller_observed_crash(verification)
                else ()
            )
            crash_ids = tuple(
                dict.fromkeys((*blocking_crash_ids, *controller_crash_ids))
            )
            crash_probe: dict[str, ScoreResult] = {}
            if crash_ids:
                crash_probe = measure_patched(
                    crash_ids,
                    checkout=experiment,
                    split="dev",
                    seed=config.dev_seed,
                    collect_coverage=True,
                )
            if controller_crash_ids:
                transitions = {
                    task_id: {
                        "stock_status": stock[task_id].status,
                        "patched_status": (
                            crash_probe[task_id].status
                            if task_id in crash_probe
                            else "missing"
                        ),
                        "stock_output": (
                            stock[task_id].detail
                            or stock[task_id].log_tail
                            or stock[task_id].traceback
                        )[-2_000:],
                        "patched_output": (
                            (
                                crash_probe[task_id].detail
                                or crash_probe[task_id].log_tail
                                or crash_probe[task_id].traceback
                            )[-2_000:]
                            if task_id in crash_probe
                            else ""
                        ),
                    }
                    for task_id in controller_crash_ids
                }
                reproduction["controller_workloads"] = transitions
                reproduction["patched"] = (
                    "resolved"
                    if all(
                        task_id in crash_probe
                        and crash_probe[task_id].status == "ok"
                        for task_id in controller_crash_ids
                    )
                    else "still_failing"
                )
            unresolved_crashes = bool(blocking_crash_ids) and all(
                crash_probe.get(task_id) is not None
                and crash_probe[task_id].status == "crash"
                for task_id in blocking_crash_ids
            )
            if unresolved_crashes:
                # No lift is possible only when the complete lift set consists
                # of unresolved crashes. A known crash in the protect-only set
                # is an unchanged baseline, not a reason to skip healthy lift
                # workloads. The old all-stock shortcut silently evaluated
                # unrelated PolyFeatures patches only on the known PCA crash.
                # Reuse stock protect results and return the instrumented causal
                # feedback immediately instead of spending minutes on a full
                # suite that cannot produce KEEP.
                patched = dict(stock)
                patched.update(crash_probe)
            else:
                patched = measure_patched(
                    exam_ids,
                    checkout=experiment,
                    split="dev",
                    seed=config.dev_seed,
                )
            last = verdict(stock, patched, lift_ids=lift_ids, protect_ids=protect_ids)
            last.experiment_id = experiment_id
            if (
                verification.status == "quality_hypothesis"
                and not last.keep
                and not last.infrastructure_error
                and str(last.reason).startswith("target_delta")
            ):
                affected_ids = (
                    _candidate_confirmation_scope(
                        source,
                        candidate,
                        stock,
                        operation_hints,
                        exam_ids,
                    )
                    or exam_ids
                )
                if affected_metric_moved(stock, patched, affected_ids):
                    last.keep = True
                    last.reason = "affected_metric_moved_pending_final"
                else:
                    last.reason = "no_affected_metric_signal"
            record_stock = stock
            global_protect_failed = False
            if last.keep:
                global_decision, global_stock, global_patched = (
                    enforce_global_dev_protect(
                        experiment,
                        stock,
                        patched,
                    )
                )
                record_stock = global_stock
                patched = global_patched
                if not global_decision.keep:
                    if (
                        verification.status == "quality_hypothesis"
                        and str(global_decision.reason).startswith("target_delta")
                    ):
                        last.keep = True
                        last.reason = "affected_metric_moved_pending_final"
                    else:
                        last = global_decision
                        global_protect_failed = True
                        if revision >= max(1, max_revisions):
                            safety_extension_granted = True
            metric_reason = last.reason
            correctness_confirmed = bool(
                not last.keep
                and not last.infrastructure_error
                and verification.status == "verified_bug"
                and reproduction.get("stock") == "failed_as_predicted"
                and reproduction.get("patched") == "resolved"
                and not metric_reason.startswith("regression")
            )
            maintenance_confirmed = bool(
                maintenance_screen
                and not last.keep
                and not last.infrastructure_error
                and not metric_reason.startswith("regression")
                and last.regression_deltas
                and all(
                    delta is not None and delta >= -1e-12
                    for delta in last.regression_deltas.values()
                )
            )
            candidate_diff = _candidate_diff(experiment, source, candidate)
            signal_confirmation: dict | None = None
            small_metric_confirmed = False
            if (
                not last.keep
                and not last.infrastructure_error
                and last.target_delta is not None
                and last.target_delta > 0
                and last.reason.startswith("target_delta")
                and signal_confirmations_used < config.max_signal_confirmations
                and policy.confirm_small_signals
            ):
                signal_scope = (
                    _candidate_confirmation_scope(
                        source,
                        candidate,
                        stock,
                        operation_hints,
                        exam_ids,
                    )
                    or exam_ids
                )
                signal_lift = tuple(
                    task_id for task_id in lift_ids if task_id in signal_scope
                )
                signal_protect = tuple(
                    task_id for task_id in protect_ids if task_id in signal_scope
                )
                if not signal_lift:
                    signal_lift = signal_scope
                if not signal_protect:
                    signal_protect = signal_scope
                signal_confirmations_used += 1
                signal_confirmed, signal_confirmation = _confirm_dev(
                    source,
                    experiment,
                    signal_scope,
                    signal_lift,
                    signal_protect,
                    seeds=config.confirmation_seeds,
                    evidence_only=True,
                )
                append_journal(
                    journal,
                    {
                        "event": "metric_signal_confirmation",
                        "candidate": candidate.candidate_id,
                        "tasks": list(signal_scope),
                        "confirmation_index": signal_confirmations_used,
                        **signal_confirmation,
                    },
                )
                if signal_confirmed:
                    small_metric_confirmed = True
                    promising_dir = workspace / "promising"
                    promising_dir.mkdir(parents=True, exist_ok=True)
                    (promising_dir / f"{candidate.candidate_id}.patch").write_text(
                        candidate_diff,
                        encoding="utf-8",
                    )
            if correctness_confirmed:
                # A reproduced correctness defect is useful even when it does
                # not move this frozen metric. It is a separate successful
                # outcome, not a weak metric candidate.
                affected_dev = affected_metric.get("dev_confirmation") or {}
                if affected_dev.get("confirmed"):
                    # A small DEV signal is evidence attached to the
                    # correctness finding, not permission to consume FINAL.
                    # FINAL is reserved for the single broad DEV winner that
                    # terminates the campaign.
                    last.reason = "correctness_keep_with_metric_signal"
                    last.stage = "correctness"
                    (candidate_dir / "affected_metric.json").write_text(
                        json.dumps(affected_metric, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    last.reason = "correctness_keep"
                    last.stage = "correctness"
                last.correctness_keep = True
                correctness_dir = workspace / "correctness_fixes"
                correctness_dir.mkdir(parents=True, exist_ok=True)
                correctness_path = correctness_dir / f"{candidate.candidate_id}.patch"
                correctness_path.write_text(candidate_diff, encoding="utf-8")
                append_journal(
                    journal,
                    {
                        "event": "correctness_keep",
                        "candidate": candidate.candidate_id,
                        "path": str(correctness_path),
                        "metric_reason": metric_reason,
                        "target_delta": last.target_delta,
                        "affected_metric_status": affected_metric.get("status"),
                        "affected_dev_signal": bool(affected_dev.get("confirmed")),
                    },
                )
                feedback = (
                    f"outcome={last.reason}; the independent stock failure is "
                    "resolved, comparative tests pass, and the complete DEV protect "
                    "suite has no regression. Accept this as a library bug fix; "
                    "metric improvement is a separate outcome."
                )
            elif small_metric_confirmed:
                last.reason = "confirmed_small_metric_keep"
                last.stage = "metric_signal"
                last.metric_signal_keep = True
                metric_dir = workspace / "metric_signal_fixes"
                metric_dir.mkdir(parents=True, exist_ok=True)
                metric_path = metric_dir / f"{candidate.candidate_id}.patch"
                metric_path.write_text(candidate_diff, encoding="utf-8")
                append_journal(
                    journal,
                    {
                        "event": "metric_signal_keep",
                        "candidate": candidate.candidate_id,
                        "path": str(metric_path),
                        "target_delta": last.target_delta,
                        "confirmation": signal_confirmation,
                    },
                )
                feedback = (
                    "outcome=confirmed_small_metric_keep; the metric gain is below "
                    "the single-run practical threshold, but it stayed positive "
                    "across confirmation seeds and SHADOW with no protected regression."
                )
            elif maintenance_confirmed:
                last.reason = "maintenance_keep"
                last.stage = "maintenance"
                last.maintenance_keep = True
                maintenance_dir = workspace / "maintenance_fixes"
                maintenance_dir.mkdir(parents=True, exist_ok=True)
                maintenance_path = maintenance_dir / f"{candidate.candidate_id}.patch"
                maintenance_path.write_text(candidate_diff, encoding="utf-8")
                append_journal(
                    journal,
                    {
                        "event": "maintenance_keep",
                        "candidate": candidate.candidate_id,
                        "path": str(maintenance_path),
                        "metric_reason": metric_reason,
                        "target_delta": last.target_delta,
                        "behavior_probe_status": behavior_probe.get("status"),
                    },
                )
                feedback = (
                    "outcome=maintenance_keep; Verifier justified a general runtime "
                    "improvement, the causal probe changed, comparative tests pass, "
                    "DEV and SHADOW were neutral, and the complete DEV protect suite "
                    "has no regression. Preserve it as a non-metric library improvement."
                )
            else:
                feedback = _dev_feedback(
                    last,
                    patched,
                    candidate,
                    stock=record_stock,
                )
                if signal_confirmation is not None:
                    shadow = signal_confirmation.get("shadow") or {}
                    if signal_confirmation.get("confirmed"):
                        feedback += (
                            "\nThis is a confirmed small metric signal: it is positive "
                            "across model seeds and SHADOW without a protected regression, "
                            "but remains below the practical KEEP threshold. Preserve the "
                            "causal mechanism and seek a larger effect."
                        )
                    else:
                        feedback += (
                            "\nThe small positive DEV signal did not generalize under the "
                            "evidence gate. Diagnose the paired task trade-off before "
                            "refining it. confirmation="
                            + json.dumps(
                                {
                                    "improved_seeds": signal_confirmation.get(
                                        "improved_seeds"
                                    ),
                                    "regressed_task_seed_pairs": signal_confirmation.get(
                                        "regressed_task_seed_pairs"
                                    ),
                                    "shadow": {
                                        "keep": shadow.get("keep"),
                                        "reason": shadow.get("reason"),
                                        "target_delta": shadow.get("target_delta"),
                                    },
                                },
                                sort_keys=True,
                            )
                        )
                if global_protect_failed:
                    feedback += (
                        "\nMandatory global protect suite rejected this focused KEEP. "
                        "Narrow any shared-base edit to the operation named by the lead "
                        "unless evidence shows sibling operations need the same behavior."
                    )
            feedback_history.append(feedback)
            feedback = _revision_feedback_context(feedback_history)
            record_attempt(
                journal,
                workspace,
                lead,
                candidate,
                record_stock,
                patched,
                last,
                loc(lead),
                revision,
                diff=candidate_diff,
                tests=after_tests,
                hypothesis_id=hypothesis.id,
                patch_hash=patch_hash,
                feedback=feedback,
                reproduction=reproduction,
                behavior_probe=behavior_probe,
                affected_metric=affected_metric,
                findings_path=findings_path,
                run_number=run_number,
                run_id=run_id,
                source_commit_value=source_commit_value,
                source_hash=source_hash,
            )
            logger.info(
                "evolve fix %s/%s revision %s %s %s",
                i,
                n_leads,
                revision,
                "KEEP"
                if (
                    last.keep
                    or last.correctness_keep
                    or last.maintenance_keep
                    or last.metric_signal_keep
                )
                else "drop",
                last.reason,
            )
            if maintenance_confirmed or small_metric_confirmed:
                # Confirmed secondary acceptance tracks do not consume FINAL.
                discard_experiment_checkout(
                    experiment, workspace=workspace, source=source
                )
                last.keep = True
                break
            if last.keep:
                confirmation_scope = _candidate_confirmation_scope(
                    source,
                    candidate,
                    stock,
                    operation_hints,
                    exam_ids,
                )
                completed = finalize_dev_keep(
                    experiment,
                    candidate,
                    last,
                    confirmation_ids=confirmation_scope,
                    verification_result=verification,
                )
                if completed is not None:
                    return completed
                confirmation_failure = confirmation_failures.pop(
                    candidate.candidate_id,
                    None,
                )
                if confirmation_failure is not None and revision < max_revisions:
                    shadow = confirmation_failure.get("shadow") or {}
                    confirmation_feedback = (
                        "outcome=dev_confirmation_failed; the patch improved initial "
                        "DEV but did not survive independent confirmation. Refine the "
                        "same causal mechanism; do not switch to an unrelated file.\n"
                        f"improved_seeds={confirmation_failure.get('improved_seeds')}; "
                        "regressed_task_seed_pairs="
                        f"{confirmation_failure.get('regressed_task_seed_pairs')}; "
                        f"infrastructure_failures={confirmation_failure.get('infrastructure_failures')}; "
                        f"shadow_task_result={json.dumps(shadow, ensure_ascii=False, default=str)}\n\n"
                        "Previous evaluated patch:\n" + _candidate_patch_text(candidate)
                    )
                    feedback_history.append(confirmation_feedback)
                    feedback = _revision_feedback_context(feedback_history)
                    hypothesis = _start_revision(
                        workspace,
                        hypothesis,
                        lead,
                        revision + 1,
                        last.reason,
                    )
                    continue
                break

            discard_experiment_checkout(experiment, workspace=workspace, source=source)
            if last.infrastructure_error:
                break
            if last.reason == "no_affected_metric_signal":
                break
            if last.reason == "queued_for_fedot_quality":
                break
            if correctness_confirmed:
                # Correctness and metric quality are separate acceptance tracks.
                # Set campaign success only after bypassing metric finalization:
                # a correctness patch must never consume the hidden FINAL split.
                last.keep = True
                break
            if revision < max_revisions or (
                safety_extension_granted and revision == max(1, max_revisions)
            ):
                hypothesis = _start_revision(
                    workspace, hypothesis, lead, revision + 1, last.reason
                )
    if last.reason != "queued_for_fedot_quality":
        _record_final_skipped(journal, reason=last.reason)
    return finish(last)


def _quick_quality_screen(*args, **kwargs):
    """Run the cheap gate with the campaign's replaceable evaluator bindings."""

    return quick_quality_screen(
        *args,
        **kwargs,
        measure_stock_fn=measure_stock,
        measure_patched_fn=measure_patched,
        verdict_fn=verdict,
    )


def _confirm_dev(*args, **kwargs):
    """Run confirmation with the campaign's replaceable evaluator bindings."""

    return confirm_dev(
        *args,
        **kwargs,
        measure_stock_fn=measure_stock,
        measure_patched_fn=measure_patched,
        verdict_fn=verdict,
    )


def _ablate_candidate(*args, **kwargs) -> PatchCandidate:
    return ablate_candidate(
        *args,
        **kwargs,
        compare_behavior_probe_fn=compare_behavior_probe,
        behavior_probe_blocks_fn=_behavior_probe_blocks_candidate,
        measure_fedot_tests_fn=measure_fedot_tests,
        confirm_dev_fn=_confirm_dev,
    )


def eval_contract(checkout: Path | None = None) -> dict:
    checkout = checkout or resolve_fedot_src()
    ids = quality_suite()
    results = {task_id: run_stock(task_id, checkout=checkout) for task_id in ids}
    return {
        "guard_cases": guard_path("data/cases.json"),
        "guard_scorer": guard_path("fedotllm/agents/evolve/evaluation/scorer.py"),
        "scores": {key: _score_log(value) for key, value in results.items()},
    }


def _record_final_skipped(journal: Path, *, reason: str) -> None:
    record_final_skipped(journal, reason=reason)


def _record_final(*args, **kwargs) -> Decision | None:
    """Run FINAL using the campaign's replaceable evaluator bindings."""

    return record_final(
        *args,
        **kwargs,
        measure_stock_fn=measure_stock,
        measure_patched_fn=measure_patched,
        verdict_fn=verdict,
    )
