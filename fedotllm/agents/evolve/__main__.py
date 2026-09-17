from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

DEFAULT_PRESETS = "fedotllm:openrouter"
# Stage output budgets. max_tokens also pays for hidden reasoning on providers
# such as OpenRouter; an explicit preset budget always wins.
STAGE_MAX_TOKENS = {"scout": 3000, "verifier": 4000, "fixer": 8000}
EXIT_INFRASTRUCTURE = 3


def _bind_fedot(path: Path | None) -> None:
    if path is None:
        cache = Path(__file__).resolve().parents[3] / ".repo_cache" / "FEDOT"
        if cache.is_dir() and not os.environ.get("FEDOTLLM_REPO_PATH"):
            os.environ["FEDOTLLM_REPO_PATH"] = str(cache)
        return
    os.environ["FEDOTLLM_REPO_PATH"] = str(path.resolve())


def _stage_inferences(presets: str):
    """Build independent Scout, Verifier and Fixer clients from one preset.

    Returns ``(None, None, None)`` when no API key is available. The parent
    ``.env`` is loaded first so credentials kept there count as available.
    """
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        return None, None, None
    from fedotllm.configs.loader import load_config
    from fedotllm.llm import AIInference

    config = load_config(presets=presets)
    model_names = {
        "scout": config.evolve.reader_model,
        "verifier": config.evolve.reader_model,
        "fixer": config.evolve.fixer_model,
    }
    clients = []
    for stage, max_tokens in STAGE_MAX_TOKENS.items():
        stage_config = config.llm.model_copy(deep=True)
        stage_config.model_name = model_names[stage] or config.llm.model_name
        # Evolve experiments are intentionally single-model. A preset-wide
        # fallback must not change the treatment halfway through a run.
        stage_config.fallback_models = ""
        stage_config.completion_params = dict(stage_config.completion_params)
        stage_config.completion_params.setdefault("max_tokens", max_tokens)
        clients.append(AIInference(stage_config))
    return tuple(clients)


def _csv(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated CLI value, dropping blanks and duplicates."""
    return tuple(
        dict.fromkeys(part.strip() for part in (raw or "").split(",") if part.strip())
    )


def _task_ids(raw: str) -> tuple[str, ...] | None:
    values = _csv(raw)
    if not values:
        return None
    from fedotllm.agents.evolve.evaluation.tasks import load_task

    unknown: list[str] = []
    for task_id in values:
        try:
            load_task(task_id)
        except KeyError:
            unknown.append(task_id)
    if unknown:
        raise ValueError(f"unknown Evolve task ids: {', '.join(unknown)}")
    return values


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str), flush=True)


def _decision_exit_code(decision) -> int:
    if decision.infrastructure_error:
        return EXIT_INFRASTRUCTURE
    if (
        decision.correctness_keep
        or decision.maintenance_keep
        or decision.metric_signal_keep
    ):
        return 0
    if decision.dev_keep and decision.final_keep is not True:
        return 2
    return 0 if decision.dev_keep else 1


def _bounded_fixer_reserve(max_queries: int | None, requested: int) -> int:
    """Keep repair/probe capacity while always leaving discovery one request."""

    if requested < 0:
        raise ValueError("--llm-reserve-fixer-queries must be non-negative")
    if max_queries is None:
        return requested
    return min(requested, max(0, max_queries - 1))


def _campaign_reserves(
    max_queries: int | None,
    fixer_requested: int,
    *,
    has_verifier: bool,
    has_fixer: bool,
) -> dict[str, int]:
    """Reserve the bounded correctness gate and repair capacity."""

    if fixer_requested < 0:
        raise ValueError("--llm-reserve-fixer-queries must be non-negative")
    available = None if max_queries is None else max(0, max_queries - 1)
    verifier = 3 if has_verifier else 0
    if available is not None:
        verifier = min(verifier, available)
        available -= verifier
    fixer = fixer_requested if has_fixer else 0
    if available is not None:
        fixer = min(fixer, available)
    return {
        stage: amount
        for stage, amount in (("verifier", verifier), ("fixer", fixer))
        if amount > 0
    }


def _stage_clients(scout, verifier, fixer) -> dict[str, Any]:
    return {
        role: client
        for role, client in (("scout", scout), ("verifier", verifier), ("fixer", fixer))
        if client is not None
    }


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def _add_fedot(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fedot", type=Path, default=None)


def _add_presets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--presets", default=DEFAULT_PRESETS)


def _add_workspace(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--workspace", type=Path, default=Path(default))


def _add_llm_budget(parser: argparse.ArgumentParser, *, max_queries: int) -> None:
    parser.add_argument("--llm-max-queries", type=int, default=max_queries)
    parser.add_argument("--llm-max-cost-usd", type=float, default=0.25)
    parser.add_argument(
        "--unlimited-llm-budget",
        action="store_true",
        help=(
            "Record LLM usage without enforcing aggregate query or cost limits; "
            "per-request timeout and campaign action limits still apply"
        ),
    )
    parser.add_argument("--llm-reserve-fixer-queries", type=int, default=12)


def _add_quality_args(parser: argparse.ArgumentParser, *, workspace: str, tasks_help: str) -> None:
    _add_fedot(parser)
    parser.add_argument("--tasks", default="", help=tasks_help)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--cpu-quota", type=int, default=None)
    _add_workspace(parser, workspace)
    parser.add_argument(
        "--stock-cache",
        type=Path,
        default=None,
        help="Explicit EVOLVE_QUALITY_STOCK_CACHE directory; reuse cached stock runs",
    )


def _add_campaign_args(parser: argparse.ArgumentParser, *, max_hypotheses: int) -> None:
    """Arguments shared by ``run`` and ``continue``."""
    from fedotllm.agents.evolve.storage.findings import default_findings_path

    _add_presets(parser)
    _add_fedot(parser)
    parser.add_argument("--max-hypotheses", type=int, default=max_hypotheses)
    parser.add_argument(
        "--metric-only",
        action="store_true",
        help="Preregistered metric-uplift study; correctness-only findings do not count",
    )
    parser.add_argument(
        "--metric-study",
        type=Path,
        help="Persistent study directory shared across campaigns; one frozen FINAL batch",
    )
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=30)
    parser.add_argument(
        "--max-measurement-pairs", type=int, default=160,
        help="Hard cap on stock/patched benchmark pairs; -1 disables this cap",
    )
    parser.add_argument(
        "--reserve-final-measurement-pairs", type=int, default=54,
        help="Pairs reserved for the frozen FINAL batch",
    )
    parser.add_argument(
        "--max-measurement-seconds", type=float, default=None,
        help="Hard wall-clock ceiling for paired measurements; omitted means no time ceiling",
    )
    parser.add_argument(
        "--reserve-final-measurement-seconds", type=float, default=0.0,
        help="Wall-clock time reserved for the frozen FINAL batch",
    )
    parser.add_argument(
        "--transfer-screen-sources", type=int, default=3,
        help="Pre-registered dataset sources in the cheap transfer screen",
    )
    _add_llm_budget(parser, max_queries=30)
    parser.add_argument(
        "--site-cooldown-campaigns", type=int, default=1,
        help="Temporarily skip exact sites from this many completed campaigns",
    )
    parser.add_argument(
        "--lift-tasks", default="",
        help="Comma-separated frozen DEV workloads allowed to establish improvement",
    )
    parser.add_argument(
        "--protect-tasks", default="",
        help="Comma-separated frozen workloads that must not regress; default is the full suite",
    )
    parser.add_argument(
        "--coverage-tasks", type=int, default=None,
        help="Collect execution evidence for the first N selected workloads (default: 12 diverse workloads)",
    )
    parser.add_argument(
        "--findings", type=Path, default=default_findings_path(),
        help="Append-only cross-run findings dataset",
    )


def _build_parser() -> argparse.ArgumentParser:
    from fedotllm.agents.evolve.storage.findings import default_findings_path

    parser = argparse.ArgumentParser(
        description=(
            "EvolveAgent: find verified FEDOT bug fixes or patches that improve "
            "full Fedot(best_quality, 1h) quality jobs."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval-contract", help="Frozen scorer contract, no LLM")
    _add_fedot(p)

    p = sub.add_parser(
        "quality-job",
        help="Stock vs patch Fedot(best_quality, timeout=3600s) on registered full OpenML tasks",
    )
    _add_quality_args(
        p,
        workspace="/tmp/evolve-fedot-quality",
        tasks_help="Comma-separated quality task ids; default is the registry pool (24 OpenML + public TS)",
    )
    p.add_argument("--patch-checkout", type=Path, default=None)
    p.add_argument(
        "--stock-only",
        action="store_true",
        help="Run only stock Fedot(1h) baselines on the registered tasks",
    )

    p = sub.add_parser(
        "quality-drain",
        help="Stock-cache applicable registry tasks, then score every queued patch",
    )
    _add_quality_args(
        p,
        workspace="/tmp/evolve-agent-quality-drain",
        tasks_help="Comma-separated quality task ids; default pool is the full registry, then tabular vs TS per patch",
    )

    p = sub.add_parser("metric-finalize", help="Resume only an already frozen metric FINAL batch; no LLM")
    _add_fedot(p)
    p.add_argument("--metric-study", type=Path, required=True)

    p = sub.add_parser("doctor", help="Validate frozen FEDOT, datasets, runner and evaluator")
    _add_fedot(p)
    _add_workspace(p, "/tmp/evolve-agent-doctor")
    p.add_argument("--task", default="catboost")
    p.add_argument("--skip-evaluator", action="store_true")

    p = sub.add_parser("benchmark", help="Run a deterministic Evolve component benchmark")
    p.add_argument(
        "--component",
        required=True,
        choices=(
            "micro-fast",
            "micro-agent",
            "micro-discovery",
            "hidden-controls",
            "hidden-controls-fresh",
            "hidden-controls-fresh2",
            "judge",
            "localization",
            "configuration",
            "configuration-search",
            "affected",
            "verification",
            "contract-support",
            "fixer",
            "e2e",
        ),
    )
    _add_fedot(p)
    _add_workspace(p, "/tmp/evolve-agent-benchmark")
    _add_presets(p)
    p.add_argument("--architecture", choices=("monolith", "staged", "committee"), default="staged")
    p.add_argument("--cases", default="", help="Comma-separated private micro case ids")
    p.add_argument("--committee-size", type=int, default=3)
    _add_llm_budget(p, max_queries=40)
    p.add_argument(
        "--allow-llm",
        action="store_true",
        help="Explicitly authorize paid LLM calls for fixer/e2e components",
    )

    p = sub.add_parser("leads", help="Walk FEDOT source via repo map. No tests.")
    _add_fedot(p)
    _add_presets(p)

    p = sub.add_parser("run", help="Discover a site, patch, KEEP/DROP on the quality suite")
    _add_workspace(p, "/tmp/evolve-agent-run")
    _add_campaign_args(p, max_hypotheses=5)

    p = sub.add_parser(
        "continue",
        help="Resume an agent-owned hypothesis with corrected measured feedback",
    )
    p.add_argument("--from-workspace", type=Path, required=True)
    p.add_argument(
        "--candidate",
        default=None,
        help="Saved candidate id; omit to resume the latest checkpointed patch or Scout lead",
    )
    p.add_argument("--workspace", type=Path, required=True)
    _add_campaign_args(p, max_hypotheses=1)

    p = sub.add_parser("findings", help="Summarize or import the cross-run findings dataset")
    p.add_argument("--dataset", type=Path, default=default_findings_path())
    p.add_argument(
        "--import-workspace",
        type=Path,
        action="append",
        default=[],
        help="Import a completed or interrupted campaign workspace (repeatable)",
    )

    p = sub.add_parser("scoreboard", help="DEV getting_better vs FINAL getting_better_final")
    _add_workspace(p, "/tmp/evolve-agent-run")

    p = sub.add_parser("replay", help="Dump cmd/log_tail/diff from journal. Harness-only.")
    _add_workspace(p, "/tmp/evolve-agent-run")
    p.add_argument("--candidate", default=None)

    p = sub.add_parser("noise", help="Phase 0: stock vs stock on DEV tasks. No LLM.")
    _add_workspace(p, "/tmp/evolve-agent-noise")
    _add_fedot(p)
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--tasks", default="catboost,lgbm,rf")

    p = sub.add_parser("recall", help="Phase 1: offline file Recall@k. LLM pick optional.")
    _add_fedot(p)
    _add_presets(p)
    _add_workspace(p, "/tmp/evolve-agent-recall")
    p.add_argument("--no-llm", action="store_true")

    p = sub.add_parser("repair", help="Phase 2: oracle-location fixer.")
    _add_fedot(p)
    _add_presets(p)
    _add_workspace(p, "/tmp/evolve-agent-oracle")
    p.add_argument("--split", default="dev", choices=("dev", "test"))
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--replay-tests", action="store_true", help="Gate saved patches with FEDOT unit tests")
    p.add_argument(
        "--replay-holdout",
        action="store_true",
        help="DEV metric on saved patches. Not the e2e scoreboard.",
    )
    p.add_argument("--files", default="", help="Comma-separated FEDOT paths; overrides split/limit")
    return parser


# --------------------------------------------------------------------------
# command handlers
# --------------------------------------------------------------------------

Handler = Callable[[argparse.Namespace, argparse.ArgumentParser], int]


def _cmd_quality_job(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.controller.quality_executor import measure_fedot_quality
    from fedotllm.agents.evolve.evaluation.fedot_quality import (
        bind_stock_cache,
        run_quality_stock_jobs,
    )
    from fedotllm.agents.evolve.evaluation.quality_registry import list_quality_task_ids
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
    from fedotllm.agents.evolve.storage.journal import append_journal

    if args.patch_checkout is None and not args.stock_only:
        parser.error("quality-job needs --patch-checkout or --stock-only")
    source = resolve_fedot_src()
    args.workspace.mkdir(parents=True, exist_ok=True)
    journal = args.workspace / "quality_jobs.jsonl"
    stock_cache = bind_stock_cache(args.stock_cache)
    task_ids = _csv(args.tasks) or list_quality_task_ids()
    print(
        json.dumps(
            {
                "event": "quality-job-start",
                "stock_only": bool(args.stock_only),
                "fedot": str(source),
                "task_ids": list(task_ids),
                "n_jobs": args.n_jobs,
                "cpu_quota": args.cpu_quota,
                "stock_cache": str(stock_cache),
                "data_cache": os.environ.get("EVOLVE_QUALITY_DATA_CACHE"),
            }
        ),
        flush=True,
    )
    if args.stock_only:
        print(f"quality-job stock-only writing cache under {stock_cache}", flush=True)
        rows = run_quality_stock_jobs(
            stock_checkout=source,
            task_ids=task_ids,
            n_jobs=args.n_jobs,
            cpu_quota=args.cpu_quota,
        )
        append_journal(
            journal,
            {
                "event": "fedot_quality_stock",
                "task_ids": list(task_ids),
                "jobs": [
                    {
                        "spec": row["spec"],
                        "search_ran": row["search_ran"],
                        "stock_status": row["stock"]["status"],
                        "stock_score": row["stock"]["score"],
                    }
                    for row in rows
                ],
            },
        )
        _print_json(
            {
                "mode": "stock-only",
                "task_ids": list(task_ids),
                "journal": str(journal),
                "jobs": [
                    {
                        "task_id": row["spec"]["source_dataset"],
                        "status": row["stock"]["status"],
                        "score": row["stock"]["score"],
                        "search_ran": row["search_ran"],
                    }
                    for row in rows
                ],
            }
        )
        return 0 if all(row["search_ran"] for row in rows) else EXIT_INFRASTRUCTURE
    decision = measure_fedot_quality(
        source,
        args.patch_checkout.resolve(),
        journal=journal,
        task_ids=task_ids,
        n_jobs=args.n_jobs,
        cpu_quota=args.cpu_quota,
    )
    _print_json(
        {
            "keep": decision.keep,
            "reason": decision.reason,
            "target_delta": decision.target_delta,
            "regression_deltas": decision.regression_deltas,
            "infrastructure_error": decision.infrastructure_error,
            "task_ids": list(task_ids),
            "journal": str(journal),
        }
    )
    if decision.infrastructure_error:
        return EXIT_INFRASTRUCTURE
    return 0 if decision.keep else 1


def _cmd_quality_drain(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    from fedotllm.agents.evolve.evaluation.fedot_quality import bind_stock_cache
    from fedotllm.agents.evolve.evaluation.quality_registry import list_quality_task_ids
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    args.workspace.mkdir(parents=True, exist_ok=True)
    stock_cache = bind_stock_cache(args.stock_cache)
    task_ids = _csv(args.tasks) or list_quality_task_ids()
    print(
        json.dumps(
            {
                "event": "quality-drain-start",
                "task_ids": list(task_ids),
                "stock_cache": str(stock_cache),
            }
        ),
        flush=True,
    )
    payload = drain_quality_queue(
        resolve_fedot_src(),
        args.workspace,
        journal=args.workspace / "quality_jobs.jsonl",
        task_ids=task_ids,
        n_jobs=args.n_jobs,
        cpu_quota=args.cpu_quota,
        stock_cache=stock_cache,
    )
    _print_json(payload)
    failed = any(
        job.get("status") in {"apply_failed", "infrastructure_error"}
        for job in payload["jobs"]
    )
    return EXIT_INFRASTRUCTURE if not payload["stock_ok"] or failed else 0


def _cmd_metric_finalize(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.controller.metric_study import BATCH_SIZE, finalize_batch
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    if not (args.metric_study / "final-batch.json").exists():
        parser.error("metric-finalize only resumes an already sealed FINAL batch")
    payload = finalize_batch(resolve_fedot_src(), args.metric_study)
    print(json.dumps(payload, indent=2))
    return 0 if payload["metric_confirmed_count"] == BATCH_SIZE else 2


def _cmd_eval_contract(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.controller.campaign import eval_contract

    payload = eval_contract()
    _print_json(payload)
    scores = payload["scores"]
    ok = (
        payload["guard_cases"] == "deny"
        and payload["guard_scorer"] == "deny"
        and all(
            scores.get(task_id, {}).get("status") == "ok"
            for task_id in ("catboost", "lgbm", "rf")
        )
    )
    return 0 if ok else 1


def _cmd_doctor(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.commands.doctor import run_doctor
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    payload = run_doctor(
        resolve_fedot_src(),
        args.workspace,
        evaluator_task=args.task,
        evaluator=not args.skip_evaluator,
    )
    _print_json(payload)
    return 0 if payload["ok"] else EXIT_INFRASTRUCTURE


def _cmd_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.benchmark.runner import run_component
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
    from fedotllm.agents.evolve.storage.run_budget import EvolveRunBudget

    if args.allow_llm:
        args.workspace.mkdir(parents=True, exist_ok=True)
        os.environ["EVOLVE_AGENT_LLM_AUDIT"] = str(
            (args.workspace / "llm_calls.jsonl").resolve()
        )
    scout, verifier, fixer = (
        _stage_inferences(args.presets) if args.allow_llm else (None, None, None)
    )
    clients = _stage_clients(scout, verifier, fixer)
    budget_context = nullcontext(None)
    if clients:
        unlimited = args.unlimited_llm_budget
        fixer_reserve = (
            0
            if unlimited
            else _bounded_fixer_reserve(args.llm_max_queries, args.llm_reserve_fixer_queries)
        )
        budget_context = EvolveRunBudget(
            clients,
            max_queries=None if unlimited else args.llm_max_queries,
            max_cost_usd=None if unlimited else args.llm_max_cost_usd,
            reserved_queries={"fixer": fixer_reserve} if fixer is not None else None,
        )
    with budget_context as budget:
        payload = run_component(
            args.component,
            resolve_fedot_src(),
            args.workspace,
            inference=fixer,
            scout_inference=scout,
            verifier_inference=verifier,
            fixer_inference=fixer,
            architecture=args.architecture,
            case_ids=_csv(args.cases) or None,
            committee_size=args.committee_size,
        )
    if budget is not None:
        budget_payload = budget.snapshot().to_dict()
        payload["run_budget"] = budget_payload
        _write_json(args.workspace / "run_budget.json", budget_payload)
    # Component functions may write their detailed artifact before the CLI
    # budget context closes. Always persist the final payload as the
    # authoritative benchmark-level summary so cost and execution status
    # cannot disappear from the saved result.
    _write_json(args.workspace / "benchmark_summary.json", payload)
    _print_json(payload)
    return 0 if payload.get("ok") else 1


def _cmd_leads(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.discovery.discover import discover_leads
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    scout, _, _ = _stage_inferences(args.presets)
    leads = discover_leads(resolve_fedot_src(), inference=scout)
    print(json.dumps([asdict(lead) for lead in leads], indent=2))
    return 0 if leads else 1


def _cmd_scoreboard(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.storage.scoreboard import summarize

    print(json.dumps(summarize(args.workspace), indent=2))
    return 0


def _cmd_findings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.storage.findings import import_workspace, summarize

    imported = [
        import_workspace(workspace, args.dataset) for workspace in args.import_workspace
    ]
    print(json.dumps({"imports": imported, **summarize(args.dataset)}, indent=2))
    return 0


def _cmd_replay(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.storage.replay import load_replay

    payload = load_replay(args.workspace, candidate=args.candidate)
    _print_json(payload)
    return 0 if payload else 1


def _cmd_noise(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.commands.calibrate import calibrate_stock
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    payload = calibrate_stock(
        checkout=resolve_fedot_src(),
        task_ids=_csv(args.tasks),
        seeds=tuple(int(part) for part in _csv(args.seeds)),
        workspace=args.workspace,
    )
    _print_json(payload["by_task"])
    return 0


def _cmd_recall(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.commands.recall import measure_localization
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    inference = None if args.no_llm else _stage_inferences(args.presets)[0]
    payload = measure_localization(
        resolve_fedot_src(),
        inference=inference,
        workspace=args.workspace,
    )
    _print_json(payload)
    return 0


def _cmd_repair(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from fedotllm.agents.evolve.commands.repair import (
        gate_saved_repairs,
        holdout_saved_repairs,
        measure_repair,
    )

    if args.replay_holdout:
        payload = holdout_saved_repairs(workspace=args.workspace)
    elif args.replay_tests:
        payload = gate_saved_repairs(workspace=args.workspace)
    else:
        _, _, fixer = _stage_inferences(args.presets)
        payload = measure_repair(
            inference=fixer,
            workspace=args.workspace,
            split=args.split,
            limit=args.limit,
            files=_csv(args.files) or None,
        )
    _print_json(payload)
    return 0


def _validate_campaign_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.metric_only and args.metric_study is None:
        parser.error("--metric-only requires --metric-study")
    if args.max_measurement_pairs < -1:
        parser.error("--max-measurement-pairs must be non-negative or -1")
    if args.reserve_final_measurement_pairs < 0:
        parser.error("--reserve-final-measurement-pairs must be non-negative")
    if args.max_measurement_seconds is not None and args.max_measurement_seconds < 0:
        parser.error("--max-measurement-seconds must be non-negative")
    if args.reserve_final_measurement_seconds < 0:
        parser.error("--reserve-final-measurement-seconds must be non-negative")
    if (
        args.max_measurement_seconds is not None
        and args.reserve_final_measurement_seconds > args.max_measurement_seconds
    ):
        parser.error("--reserve-final-measurement-seconds exceeds --max-measurement-seconds")
    if (
        args.max_measurement_pairs >= 0
        and args.reserve_final_measurement_pairs > args.max_measurement_pairs
    ):
        parser.error("--reserve-final-measurement-pairs exceeds --max-measurement-pairs")
    if args.transfer_screen_sources < 2:
        parser.error("--transfer-screen-sources must be at least 2")
    if args.coverage_tasks is not None and args.coverage_tasks < 1:
        parser.error("--coverage-tasks must be positive")


def _resume_branch(
    args: argparse.Namespace, parser: argparse.ArgumentParser, checkout: Path
) -> tuple[dict, str]:
    """Recover the candidate or Scout lead saved by an interrupted campaign."""
    from fedotllm.agents.evolve.execution.checkout import source_fingerprint
    from fedotllm.agents.evolve.protocol import acceptance_protocol_fingerprint
    from fedotllm.agents.evolve.storage.checkpoint import checkpoint_leads, load_checkpoint
    from fedotllm.agents.evolve.storage.journal import resolve_run_workspace
    from fedotllm.agents.evolve.storage.replay import (
        load_resume_branch,
        patch_feedback_from_findings,
    )

    interrupted_workspace = resolve_run_workspace(args.from_workspace)
    checkpoint = load_checkpoint(interrupted_workspace)
    candidate_id = args.candidate or checkpoint.get("candidate_id")
    branch = (
        load_resume_branch(interrupted_workspace, candidate_id=str(candidate_id))
        if candidate_id
        else None
    )
    if branch is None:
        selected = checkpoint_leads(interrupted_workspace)
        if selected:
            branch = {
                "lead": selected[0],
                "verification": None,
                "candidate": None,
                "candidate_status": "scout_checkpoint",
                "resume_diagnostic": (
                    "Resume the first accumulated Scout candidate after an "
                    "interrupted catalog walk."
                ),
                "patch_hash": "",
                "source_hash": str(checkpoint.get("source_hash") or ""),
                "candidate_id": "",
                "hypothesis_id": "",
            }
    if branch is None:
        parser.error("cannot recover a candidate or Scout lead from --from-workspace")
    feedback = ""
    if branch["patch_hash"]:
        feedback = patch_feedback_from_findings(
            args.findings,
            source_hash=source_fingerprint(checkout),
            patch_hash=branch["patch_hash"],
            evaluation_protocol_hash=acceptance_protocol_fingerprint(),
        )
    feedback = feedback or str(branch.get("resume_diagnostic") or "") or (
        "Resume the exact saved source candidate after interruption and "
        "complete its independent probe, test, and metric checks."
    )
    return branch, feedback


def _cmd_campaign(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """``run`` and ``continue``: one hunt campaign, optionally resuming a branch."""
    from fedotllm.agents.evolve.controller.campaign import run_once
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
    from fedotllm.agents.evolve.storage.run_budget import EvolveRunBudget
    from fedotllm.agents.evolve.types import EvolveAgentConfig

    _validate_campaign_args(args, parser)
    try:
        lift_ids = _task_ids(args.lift_tasks)
        protect_ids = _task_ids(args.protect_tasks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.coverage_tasks is not None:
        os.environ["EVOLVE_AGENT_COVERAGE_TASKS"] = str(args.coverage_tasks)

    scout, verifier, fixer = _stage_inferences(args.presets)
    if scout is None or fixer is None:
        print("FEDOTLLM_LLM_API_KEY is not set; propose_patch will no-op", file=sys.stderr)
    checkout = resolve_fedot_src()
    resume_branch: dict | None = None
    resume_feedback = ""
    if args.cmd == "continue":
        resume_branch, resume_feedback = _resume_branch(args, parser, checkout)

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    workspace_root = args.workspace.resolve()
    run_workspace = workspace_root / "runs" / run_id
    run_workspace.mkdir(parents=True, exist_ok=False)
    _write_json(
        workspace_root / "latest_run.json",
        {"run_id": run_id, "workspace": str(run_workspace)},
    )

    clients = _stage_clients(scout, verifier, fixer)
    budget_context = nullcontext(None)
    if clients:
        if args.unlimited_llm_budget:
            max_queries = max_cost = None
            reserves: dict[str, int] = {}
        else:
            max_queries = args.llm_max_queries
            max_cost = args.llm_max_cost_usd
            try:
                reserves = _campaign_reserves(
                    max_queries,
                    args.llm_reserve_fixer_queries,
                    has_verifier=verifier is not None,
                    has_fixer=fixer is not None,
                )
            except ValueError as exc:
                parser.error(str(exc))
        budget_context = EvolveRunBudget(
            clients,
            max_queries=max_queries,
            max_cost_usd=max_cost,
            reserved_queries=reserves or None,
        )
    with budget_context as budget:
        decision = run_once(
            checkout=checkout,
            scout_inference=scout,
            # Only correctness leads use the bounded three-call Verifier path.
            # Quality leads continue directly to DEV/SHADOW/FINAL evaluation.
            verifier_inference=verifier,
            fixer_inference=fixer,
            workspace=run_workspace,
            findings_path=args.findings,
            max_leads=args.max_hypotheses,
            max_revisions=args.max_revisions,
            max_edits=args.max_edits,
            max_actions=args.max_actions,
            site_cooldown_campaigns=args.site_cooldown_campaigns,
            lift_ids=lift_ids,
            protect_ids=protect_ids,
            resume_lead=resume_branch["lead"] if resume_branch else None,
            resume_verification=resume_branch["verification"] if resume_branch else None,
            resume_candidate=resume_branch.get("candidate") if resume_branch else None,
            resume_feedback=resume_feedback,
            run_id=run_id,
            config=EvolveAgentConfig(
                metric_only=args.metric_only,
                metric_study_path=str(args.metric_study.resolve()) if args.metric_study else "",
                max_measurement_pairs=(
                    None if args.max_measurement_pairs < 0 else args.max_measurement_pairs
                ),
                reserve_final_measurement_pairs=args.reserve_final_measurement_pairs,
                max_measurement_seconds=args.max_measurement_seconds,
                reserve_final_measurement_seconds=args.reserve_final_measurement_seconds,
                transfer_screen_sources=args.transfer_screen_sources,
            ),
        )
    budget_payload = budget.snapshot().to_dict() if budget is not None else None
    _write_json(run_workspace / "run_budget.json", budget_payload)
    summary_path = run_workspace / "campaign_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            summary = None
        if isinstance(summary, dict):
            summary["run_budget"] = budget_payload
            _write_json(summary_path, summary)
    _print_json(
        {
            "keep_dev": decision.dev_keep,
            "keep_final": decision.final_keep,
            "keep_correctness": decision.correctness_keep,
            "stage": decision.stage,
            "reason": decision.reason,
            "target_delta": decision.target_delta,
            "regression_deltas": decision.regression_deltas,
            "workspace": str(run_workspace),
            "run_budget": budget_payload,
        }
    )
    return _decision_exit_code(decision)


HANDLERS: dict[str, Handler] = {
    "quality-job": _cmd_quality_job,
    "quality-drain": _cmd_quality_drain,
    "metric-finalize": _cmd_metric_finalize,
    "eval-contract": _cmd_eval_contract,
    "doctor": _cmd_doctor,
    "benchmark": _cmd_benchmark,
    "leads": _cmd_leads,
    "scoreboard": _cmd_scoreboard,
    "findings": _cmd_findings,
    "replay": _cmd_replay,
    "noise": _cmd_noise,
    "recall": _cmd_recall,
    "repair": _cmd_repair,
    "run": _cmd_campaign,
    "continue": _cmd_campaign,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _bind_fedot(getattr(args, "fedot", None))
    return HANDLERS[args.cmd](args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
