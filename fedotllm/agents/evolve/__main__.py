from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path


def _bind_fedot(path: Path | None) -> None:
    if path is None:
        cache = Path(__file__).resolve().parents[3] / ".repo_cache" / "FEDOT"
        if cache.is_dir() and not os.environ.get("FEDOTLLM_REPO_PATH"):
            os.environ["FEDOTLLM_REPO_PATH"] = str(cache)
        return
    os.environ["FEDOTLLM_REPO_PATH"] = str(path.resolve())


def _inference(presets: str):
    # The project commonly keeps credentials in a parent `.env`.  Load it
    # before deciding whether LLM access is available; load_config() also loads
    # dotenv, but the old early return happened before load_config was called.
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        return None
    from fedotllm.configs.loader import load_config
    from fedotllm.llm import AIInference

    config = load_config(presets=presets)
    return AIInference(config.llm)


def _stage_inferences(presets: str):
    """Build independent Scout, Verifier and Fixer clients from one preset."""
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        return None, None, None
    from fedotllm.configs.loader import load_config
    from fedotllm.llm import AIInference

    config = load_config(presets=presets)
    scout_config = config.llm.model_copy(deep=True)
    verifier_config = config.llm.model_copy(deep=True)
    fixer_config = config.llm.model_copy(deep=True)
    scout_config.model_name = config.evolve.reader_model or config.llm.model_name
    verifier_config.model_name = config.evolve.reader_model or config.llm.model_name
    fixer_config.model_name = config.evolve.fixer_model or config.llm.model_name
    # Evolve experiments are intentionally single-model. A preset-wide
    # fallback must not change the treatment halfway through a run.
    scout_config.fallback_models = ""
    verifier_config.fallback_models = ""
    fixer_config.fallback_models = ""
    # max_tokens also pays for hidden reasoning on providers such as OpenRouter.
    # Honor an explicit preset budget; silently capping Scout at 3000 exhausted
    # that budget before the model could emit even a small structured action.
    scout_config.completion_params = dict(scout_config.completion_params)
    verifier_config.completion_params = dict(verifier_config.completion_params)
    fixer_config.completion_params = dict(fixer_config.completion_params)
    scout_config.completion_params.setdefault("max_tokens", 3000)
    verifier_config.completion_params.setdefault("max_tokens", 4000)
    fixer_config.completion_params.setdefault("max_tokens", 8000)
    return (
        AIInference(scout_config),
        AIInference(verifier_config),
        AIInference(fixer_config),
    )


def _task_ids(raw: str) -> tuple[str, ...] | None:
    values = tuple(
        dict.fromkeys(part.strip() for part in raw.split(",") if part.strip())
    )
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


def _decision_exit_code(decision) -> int:
    if decision.infrastructure_error:
        return 3
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


def main(argv: list[str] | None = None) -> int:
    from fedotllm.agents.evolve.storage.findings import default_findings_path

    parser = argparse.ArgumentParser(
        description=(
            "EvolveAgent: find verified FEDOT bug fixes or patches that improve "
            "full Fedot(best_quality, 1h) quality jobs."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval-contract", help="Frozen scorer contract, no LLM")
    p_eval.add_argument("--fedot", type=Path, default=None)

    p_quality = sub.add_parser(
        "quality-job",
        help="Stock vs patch Fedot(best_quality, timeout=3600s) on registered full OpenML tasks",
    )
    p_quality.add_argument("--fedot", type=Path, default=None)
    p_quality.add_argument("--patch-checkout", type=Path, default=None)
    p_quality.add_argument(
        "--tasks",
        default="",
        help="Comma-separated quality task ids; default is the full frozen registry",
    )
    p_quality.add_argument("--n-jobs", type=int, default=None)
    p_quality.add_argument("--cpu-quota", type=int, default=None)
    p_quality.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/evolve-fedot-quality"),
    )
    p_quality.add_argument(
        "--stock-only",
        action="store_true",
        help="Run only stock Fedot(1h) baselines on the registered tasks",
    )

    p_final = sub.add_parser("metric-finalize", help="Resume only an already frozen metric FINAL batch; no LLM")
    p_final.add_argument("--fedot", type=Path, default=None)
    p_final.add_argument("--metric-study", type=Path, required=True)

    p_doctor = sub.add_parser(
        "doctor", help="Validate frozen FEDOT, datasets, runner and evaluator"
    )
    p_doctor.add_argument("--fedot", type=Path, default=None)
    p_doctor.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-doctor")
    )
    p_doctor.add_argument("--task", default="catboost")
    p_doctor.add_argument("--skip-evaluator", action="store_true")

    p_benchmark = sub.add_parser(
        "benchmark", help="Run a deterministic Evolve component benchmark"
    )
    p_benchmark.add_argument(
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
    p_benchmark.add_argument("--fedot", type=Path, default=None)
    p_benchmark.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-benchmark")
    )
    p_benchmark.add_argument("--presets", default="fedotllm:openrouter")
    p_benchmark.add_argument(
        "--architecture", choices=("monolith", "staged", "committee"), default="staged"
    )
    p_benchmark.add_argument(
        "--cases", default="", help="Comma-separated private micro case ids"
    )
    p_benchmark.add_argument("--committee-size", type=int, default=3)
    p_benchmark.add_argument("--llm-max-queries", type=int, default=40)
    p_benchmark.add_argument("--llm-max-cost-usd", type=float, default=0.25)
    p_benchmark.add_argument(
        "--unlimited-llm-budget",
        action="store_true",
        help="Record benchmark usage without enforcing aggregate query or cost limits",
    )
    p_benchmark.add_argument("--llm-reserve-fixer-queries", type=int, default=12)
    p_benchmark.add_argument(
        "--allow-llm",
        action="store_true",
        help="Explicitly authorize paid LLM calls for fixer/e2e components",
    )

    p_leads = sub.add_parser("leads", help="Walk FEDOT source via repo map. No tests.")
    p_leads.add_argument("--fedot", type=Path, default=None)
    p_leads.add_argument("--presets", default="fedotllm:openrouter")

    p_run = sub.add_parser(
        "run", help="Discover a site, patch, KEEP/DROP on the quality suite"
    )
    p_run.add_argument("--workspace", type=Path, default=Path("/tmp/evolve-agent-run"))
    p_run.add_argument("--presets", default="fedotllm:openrouter")
    p_run.add_argument("--fedot", type=Path, default=None)
    p_run.add_argument("--max-hypotheses", type=int, default=5)
    p_run.add_argument("--metric-only", action="store_true", help="Preregistered metric-uplift study; correctness-only findings do not count")
    p_run.add_argument("--metric-study", type=Path, help="Persistent study directory shared across campaigns; one frozen FINAL batch")
    p_run.add_argument("--max-revisions", type=int, default=2)
    p_run.add_argument("--max-edits", type=int, default=4)
    p_run.add_argument("--max-actions", type=int, default=30)
    p_run.add_argument(
        "--max-measurement-pairs", type=int, default=160,
        help="Hard cap on stock/patched benchmark pairs; -1 disables this cap",
    )
    p_run.add_argument(
        "--reserve-final-measurement-pairs", type=int, default=54,
        help="Pairs reserved for the frozen FINAL batch",
    )
    p_run.add_argument(
        "--max-measurement-seconds", type=float, default=None,
        help="Hard wall-clock ceiling for paired measurements; omitted means no time ceiling",
    )
    p_run.add_argument(
        "--reserve-final-measurement-seconds", type=float, default=0.0,
        help="Wall-clock time reserved for the frozen FINAL batch",
    )
    p_run.add_argument(
        "--transfer-screen-sources", type=int, default=3,
        help="Pre-registered dataset sources in the cheap transfer screen",
    )
    p_run.add_argument("--llm-max-queries", type=int, default=30)
    p_run.add_argument("--llm-max-cost-usd", type=float, default=0.25)
    p_run.add_argument(
        "--unlimited-llm-budget",
        action="store_true",
        help=(
            "Record LLM usage without enforcing aggregate query or cost limits; "
            "per-request timeout and campaign action limits still apply"
        ),
    )
    p_run.add_argument("--llm-reserve-fixer-queries", type=int, default=12)
    p_run.add_argument(
        "--site-cooldown-campaigns",
        type=int,
        default=1,
        help="Temporarily skip exact sites from this many completed campaigns",
    )
    p_run.add_argument(
        "--lift-tasks",
        default="",
        help="Comma-separated frozen DEV workloads allowed to establish improvement",
    )
    p_run.add_argument(
        "--protect-tasks",
        default="",
        help="Comma-separated frozen workloads that must not regress; default is the full suite",
    )
    p_run.add_argument(
        "--coverage-tasks",
        type=int,
        default=None,
        help="Collect execution evidence for the first N selected workloads (default: 12 diverse workloads)",
    )
    p_run.add_argument(
        "--findings",
        type=Path,
        default=default_findings_path(),
        help="Append-only cross-run findings dataset",
    )

    p_continue = sub.add_parser(
        "continue",
        help="Resume an agent-owned hypothesis with corrected measured feedback",
    )
    p_continue.add_argument("--from-workspace", type=Path, required=True)
    p_continue.add_argument(
        "--candidate",
        default=None,
        help="Saved candidate id; omit to resume the latest checkpointed patch or Scout lead",
    )
    p_continue.add_argument("--workspace", type=Path, required=True)
    p_continue.add_argument("--presets", default="fedotllm:openrouter")
    p_continue.add_argument("--fedot", type=Path, default=None)
    p_continue.add_argument("--max-hypotheses", type=int, default=1)
    p_continue.add_argument("--metric-only", action="store_true")
    p_continue.add_argument("--metric-study", type=Path)
    p_continue.add_argument("--max-revisions", type=int, default=2)
    p_continue.add_argument("--max-edits", type=int, default=4)
    p_continue.add_argument("--max-actions", type=int, default=30)
    p_continue.add_argument("--max-measurement-pairs", type=int, default=160)
    p_continue.add_argument("--reserve-final-measurement-pairs", type=int, default=54)
    p_continue.add_argument("--max-measurement-seconds", type=float, default=None)
    p_continue.add_argument("--reserve-final-measurement-seconds", type=float, default=0.0)
    p_continue.add_argument("--transfer-screen-sources", type=int, default=3)
    p_continue.add_argument("--llm-max-queries", type=int, default=30)
    p_continue.add_argument("--llm-max-cost-usd", type=float, default=0.25)
    p_continue.add_argument(
        "--unlimited-llm-budget",
        action="store_true",
        help=(
            "Record LLM usage without enforcing aggregate query or cost limits; "
            "per-request timeout and campaign action limits still apply"
        ),
    )
    p_continue.add_argument("--llm-reserve-fixer-queries", type=int, default=12)
    p_continue.add_argument("--site-cooldown-campaigns", type=int, default=1)
    p_continue.add_argument("--lift-tasks", default="")
    p_continue.add_argument("--protect-tasks", default="")
    p_continue.add_argument("--coverage-tasks", type=int, default=None)
    p_continue.add_argument(
        "--findings",
        type=Path,
        default=default_findings_path(),
    )

    p_findings = sub.add_parser(
        "findings", help="Summarize or import the cross-run findings dataset"
    )
    p_findings.add_argument("--dataset", type=Path, default=default_findings_path())
    p_findings.add_argument(
        "--import-workspace",
        type=Path,
        action="append",
        default=[],
        help="Import a completed or interrupted campaign workspace (repeatable)",
    )

    p_board = sub.add_parser(
        "scoreboard", help="DEV getting_better vs FINAL getting_better_final"
    )
    p_board.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-run")
    )

    p_replay = sub.add_parser(
        "replay", help="Dump cmd/log_tail/diff from journal. Harness-only."
    )
    p_replay.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-run")
    )
    p_replay.add_argument("--candidate", default=None)

    p_noise = sub.add_parser(
        "noise", help="Phase 0: stock vs stock on DEV tasks. No LLM."
    )
    p_noise.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-noise")
    )
    p_noise.add_argument("--fedot", type=Path, default=None)
    p_noise.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    p_noise.add_argument("--tasks", default="catboost,lgbm,rf")

    p_recall = sub.add_parser(
        "recall", help="Phase 1: offline file Recall@k. LLM pick optional."
    )
    p_recall.add_argument("--fedot", type=Path, default=None)
    p_recall.add_argument("--presets", default="fedotllm:openrouter")
    p_recall.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-recall")
    )
    p_recall.add_argument("--no-llm", action="store_true")

    p_repair = sub.add_parser("repair", help="Phase 2: oracle-location fixer.")
    p_repair.add_argument("--fedot", type=Path, default=None)
    p_repair.add_argument("--presets", default="fedotllm:openrouter")
    p_repair.add_argument(
        "--workspace", type=Path, default=Path("/tmp/evolve-agent-oracle")
    )
    p_repair.add_argument("--split", default="dev", choices=("dev", "test"))
    p_repair.add_argument("--limit", type=int, default=5)
    p_repair.add_argument(
        "--replay-tests",
        action="store_true",
        help="Gate saved patches with FEDOT unit tests",
    )
    p_repair.add_argument(
        "--replay-holdout",
        action="store_true",
        help="DEV metric on saved patches. Not the e2e scoreboard.",
    )
    p_repair.add_argument(
        "--files", default="", help="Comma-separated FEDOT paths; overrides split/limit"
    )

    args = parser.parse_args(argv)
    _bind_fedot(getattr(args, "fedot", None))

    if args.cmd == "quality-job":
        from fedotllm.agents.evolve.controller.quality_executor import measure_fedot_quality
        from fedotllm.agents.evolve.evaluation.fedot_quality import run_quality_stock_jobs
        from fedotllm.agents.evolve.evaluation.quality_registry import list_quality_task_ids
        from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
        from fedotllm.agents.evolve.storage.journal import append_journal

        source = resolve_fedot_src()
        args.workspace.mkdir(parents=True, exist_ok=True)
        journal = args.workspace / "quality_jobs.jsonl"
        task_ids = tuple(
            part.strip()
            for part in args.tasks.split(",")
            if part.strip()
        ) or list_quality_task_ids()
        if args.stock_only:
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
            payload = {
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
            print(json.dumps(payload, indent=2, default=str))
            return 0 if all(row["search_ran"] for row in rows) else 3
        if args.patch_checkout is None:
            parser.error("quality-job needs --patch-checkout or --stock-only")
        decision = measure_fedot_quality(
            source,
            args.patch_checkout.resolve(),
            journal=journal,
            task_ids=task_ids,
            n_jobs=args.n_jobs,
            cpu_quota=args.cpu_quota,
        )
        payload = {
            "keep": decision.keep,
            "reason": decision.reason,
            "target_delta": decision.target_delta,
            "regression_deltas": decision.regression_deltas,
            "infrastructure_error": decision.infrastructure_error,
            "task_ids": list(task_ids),
            "journal": str(journal),
        }
        print(json.dumps(payload, indent=2, default=str))
        if decision.infrastructure_error:
            return 3
        return 0 if decision.keep else 1

    if args.cmd == "metric-finalize":
        from fedotllm.agents.evolve.controller.metric_study import finalize_batch
        from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
        if not (args.metric_study / "final-batch.json").exists():
            parser.error("metric-finalize only resumes an already sealed FINAL batch")
        payload = finalize_batch(resolve_fedot_src(), args.metric_study)
        print(json.dumps(payload, indent=2))
        return 0 if payload["metric_confirmed_count"] == 3 else 2

    if args.cmd == "eval-contract":
        from fedotllm.agents.evolve.controller.campaign import eval_contract

        payload = eval_contract()
        print(json.dumps(payload, indent=2, default=str))
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

    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

    if args.cmd == "doctor":
        from fedotllm.agents.evolve.commands.doctor import run_doctor

        payload = run_doctor(
            resolve_fedot_src(),
            args.workspace,
            evaluator_task=args.task,
            evaluator=not args.skip_evaluator,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["ok"] else 3

    if args.cmd == "benchmark":
        from fedotllm.agents.evolve.benchmark.runner import run_component
        from fedotllm.agents.evolve.storage.run_budget import EvolveRunBudget

        if args.allow_llm:
            args.workspace.mkdir(parents=True, exist_ok=True)
            os.environ["EVOLVE_AGENT_LLM_AUDIT"] = str(
                (args.workspace / "llm_calls.jsonl").resolve()
            )
        scout_inference, verifier_inference, fixer_inference = (
            _stage_inferences(args.presets) if args.allow_llm else (None, None, None)
        )
        clients = {
            role: client
            for role, client in (
                ("scout", scout_inference),
                ("verifier", verifier_inference),
                ("fixer", fixer_inference),
            )
            if client is not None
        }
        if clients:
            benchmark_max_queries = (
                None if args.unlimited_llm_budget else args.llm_max_queries
            )
            benchmark_max_cost = (
                None if args.unlimited_llm_budget else args.llm_max_cost_usd
            )
            fixer_reserve = (
                0
                if args.unlimited_llm_budget
                else _bounded_fixer_reserve(
                    args.llm_max_queries, args.llm_reserve_fixer_queries
                )
            )
            with EvolveRunBudget(
                clients,
                max_queries=benchmark_max_queries,
                max_cost_usd=benchmark_max_cost,
                reserved_queries={"fixer": fixer_reserve}
                if fixer_inference is not None
                else None,
            ) as budget:
                payload = run_component(
                    args.component,
                    resolve_fedot_src(),
                    args.workspace,
                    inference=fixer_inference,
                    scout_inference=scout_inference,
                    verifier_inference=verifier_inference,
                    fixer_inference=fixer_inference,
                    architecture=args.architecture,
                    case_ids=tuple(
                        part.strip() for part in args.cases.split(",") if part.strip()
                    )
                    or None,
                    committee_size=args.committee_size,
                )
            budget_payload = budget.snapshot().to_dict()
            payload["run_budget"] = budget_payload
            (args.workspace / "run_budget.json").write_text(
                json.dumps(budget_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            payload = run_component(
                args.component,
                resolve_fedot_src(),
                args.workspace,
                architecture=args.architecture,
                case_ids=tuple(
                    part.strip() for part in args.cases.split(",") if part.strip()
                )
                or None,
                committee_size=args.committee_size,
            )
        # Component functions may write their detailed artifact before the CLI
        # budget context closes. Always persist the final payload as the
        # authoritative benchmark-level summary so cost and execution status
        # cannot disappear from the saved result.
        (args.workspace / "benchmark_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload.get("ok") else 1

    if args.cmd == "leads":
        from fedotllm.agents.evolve.discovery.discover import discover_leads

        checkout = resolve_fedot_src()
        scout_inference, _, _ = _stage_inferences(args.presets)
        leads = discover_leads(checkout, inference=scout_inference)
        print(json.dumps([asdict(lead) for lead in leads], indent=2))
        return 0 if leads else 1

    if args.cmd == "scoreboard":
        from fedotllm.agents.evolve.storage.scoreboard import summarize

        print(json.dumps(summarize(args.workspace), indent=2))
        return 0

    if args.cmd == "findings":
        from fedotllm.agents.evolve.storage.findings import import_workspace, summarize

        imported = [
            import_workspace(workspace, args.dataset)
            for workspace in args.import_workspace
        ]
        print(json.dumps({"imports": imported, **summarize(args.dataset)}, indent=2))
        return 0

    if args.cmd == "replay":
        from fedotllm.agents.evolve.storage.replay import load_replay

        payload = load_replay(args.workspace, candidate=args.candidate)
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload else 1

    if args.cmd == "noise":
        from fedotllm.agents.evolve.commands.calibrate import calibrate_stock
        from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src

        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
        task_ids = tuple(part.strip() for part in args.tasks.split(",") if part.strip())
        checkout = resolve_fedot_src()
        payload = calibrate_stock(
            checkout=checkout,
            task_ids=task_ids,
            seeds=seeds,
            workspace=args.workspace,
        )
        print(json.dumps(payload["by_task"], indent=2, default=str))
        return 0

    if args.cmd == "recall":
        from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
        from fedotllm.agents.evolve.commands.recall import measure_localization

        checkout = resolve_fedot_src()
        inference = None if args.no_llm else _stage_inferences(args.presets)[0]
        payload = measure_localization(
            checkout,
            inference=inference,
            workspace=args.workspace,
        )
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.cmd == "repair":
        from fedotllm.agents.evolve.commands.repair import (
            gate_saved_repairs,
            holdout_saved_repairs,
            measure_repair,
        )

        if args.replay_holdout:
            payload = holdout_saved_repairs(
                workspace=args.workspace,
            )
        elif args.replay_tests:
            payload = gate_saved_repairs(
                workspace=args.workspace,
            )
        else:
            inference = _stage_inferences(args.presets)[2]
            extra = tuple(
                part.strip() for part in args.files.split(",") if part.strip()
            )
            payload = measure_repair(
                inference=inference,
                workspace=args.workspace,
                split=args.split,
                limit=args.limit,
                files=extra or None,
            )
        print(json.dumps(payload, indent=2, default=str))
        return 0

    from fedotllm.agents.evolve.controller.campaign import run_once

    try:
        lift_ids = _task_ids(args.lift_tasks)
        protect_ids = _task_ids(args.protect_tasks)
    except ValueError as exc:
        parser.error(str(exc))
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
    if args.coverage_tasks is not None:
        if args.coverage_tasks < 1:
            parser.error("--coverage-tasks must be positive")
        os.environ["EVOLVE_AGENT_COVERAGE_TASKS"] = str(args.coverage_tasks)
    scout_inference, verifier_inference, fixer_inference = _stage_inferences(args.presets)
    if scout_inference is None or fixer_inference is None:
        print(
            "FEDOTLLM_LLM_API_KEY is not set; propose_patch will no-op", file=sys.stderr
        )
    checkout = resolve_fedot_src()
    resume_branch = None
    resume_feedback = ""
    if args.cmd == "continue":
        from fedotllm.agents.evolve.execution.checkout import source_fingerprint
        from fedotllm.agents.evolve.storage.checkpoint import (
            checkpoint_leads,
            load_checkpoint,
        )
        from fedotllm.agents.evolve.storage.journal import resolve_run_workspace
        from fedotllm.agents.evolve.storage.replay import (
            load_resume_branch,
            patch_feedback_from_findings,
        )
        from fedotllm.agents.evolve.protocol import acceptance_protocol_fingerprint

        interrupted_workspace = resolve_run_workspace(args.from_workspace)
        checkpoint = load_checkpoint(interrupted_workspace)
        candidate_id = args.candidate or checkpoint.get("candidate_id")
        resume_branch = (
            load_resume_branch(
                interrupted_workspace,
                candidate_id=str(candidate_id),
            )
            if candidate_id
            else None
        )
        if resume_branch is None:
            selected = checkpoint_leads(interrupted_workspace)
            if selected:
                resume_branch = {
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
        if resume_branch is None:
            parser.error(
                "cannot recover a candidate or Scout lead from --from-workspace"
            )
        if resume_branch["patch_hash"]:
            resume_feedback = patch_feedback_from_findings(
                args.findings,
                source_hash=source_fingerprint(checkout),
                patch_hash=resume_branch["patch_hash"],
                evaluation_protocol_hash=acceptance_protocol_fingerprint(),
            )
        if not resume_feedback:
            resume_feedback = str(resume_branch.get("resume_diagnostic") or "")
        if not resume_feedback:
            resume_feedback = (
                "Resume the exact saved source candidate after interruption and "
                "complete its independent probe, test, and metric checks."
            )
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    workspace_root = args.workspace.resolve()
    run_workspace = workspace_root / "runs" / run_id
    run_workspace.mkdir(parents=True, exist_ok=False)
    (workspace_root / "latest_run.json").write_text(
        json.dumps(
            {"run_id": run_id, "workspace": str(run_workspace)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    from contextlib import nullcontext

    from fedotllm.agents.evolve.storage.run_budget import EvolveRunBudget

    budget_clients = {
        role: client
        for role, client in (
            ("scout", scout_inference),
            ("verifier", verifier_inference),
            ("fixer", fixer_inference),
        )
        if client is not None
    }
    if args.unlimited_llm_budget:
        budget_max_queries = None
        budget_max_cost_usd = None
        reserves = {}
    else:
        budget_max_queries = args.llm_max_queries
        budget_max_cost_usd = args.llm_max_cost_usd
        try:
            reserves = _campaign_reserves(
                budget_max_queries,
                args.llm_reserve_fixer_queries,
                has_verifier=verifier_inference is not None,
                has_fixer=fixer_inference is not None,
            )
        except ValueError as exc:
            parser.error(str(exc))
    budget_context = (
        EvolveRunBudget(
            budget_clients,
            max_queries=budget_max_queries,
            max_cost_usd=budget_max_cost_usd,
            reserved_queries=reserves or None,
        )
        if budget_clients
        else nullcontext(None)
    )
    with budget_context as budget:
        from fedotllm.agents.evolve.types import EvolveAgentConfig
        if args.metric_only and args.metric_study is None:
            parser.error("--metric-only requires --metric-study")
        decision = run_once(
            checkout=checkout,
            scout_inference=scout_inference,
            # Only correctness leads use the bounded three-call Verifier path.
            # Quality leads continue directly to DEV/SHADOW/FINAL evaluation.
            verifier_inference=verifier_inference,
            fixer_inference=fixer_inference,
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
            resume_verification=(
                resume_branch["verification"] if resume_branch else None
            ),
            resume_candidate=(resume_branch.get("candidate") if resume_branch else None),
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
    (run_workspace / "run_budget.json").write_text(
        json.dumps(budget_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path = run_workspace / "campaign_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            summary = None
        if isinstance(summary, dict):
            summary["run_budget"] = budget_payload
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(
        json.dumps(
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
            },
            indent=2,
        )
    )
    return _decision_exit_code(decision)


if __name__ == "__main__":
    raise SystemExit(main())
