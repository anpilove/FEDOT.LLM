from __future__ import annotations

import argparse
import json
import os
import sys
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
    if not os.environ.get("FEDOTLLM_LLM_API_KEY"):
        return None
    from fedotllm.configs.loader import load_config
    from fedotllm.llm import AIInference

    config = load_config(presets=presets)
    return AIInference(config.llm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Metric-improvement agent B: find a FEDOT site, patch, keep iff hidden holdout rises."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval-contract", help="Our hidden exam of the scorer, no LLM")
    p_eval.add_argument("--fedot", type=Path, default=None)

    p_leads = sub.add_parser("leads", help="Walk FEDOT source via repo map. No tests.")
    p_leads.add_argument("--fedot", type=Path, default=None)
    p_leads.add_argument("--presets", default="fedotllm:openrouter")
    p_leads.add_argument("--checkout", type=Path, default=None)

    p_run = sub.add_parser("run", help="Discover a site, patch, score hidden exam")
    p_run.add_argument("--workspace", type=Path, default=Path("/tmp/metric-agent-run"))
    p_run.add_argument("--presets", default="fedotllm:openrouter")
    p_run.add_argument("--fedot", type=Path, default=None)
    p_run.add_argument("--checkout", type=Path, default=None)

    p_board = sub.add_parser("scoreboard", help="Did hidden holdout get better?")
    p_board.add_argument("--workspace", type=Path, default=Path("/tmp/metric-agent-run"))

    p_replay = sub.add_parser("replay", help="Dump cmd/log_tail/diff from journal. Harness-only.")
    p_replay.add_argument("--workspace", type=Path, default=Path("/tmp/metric-agent-run"))
    p_replay.add_argument("--candidate", default=None)

    p_oracle = sub.add_parser("oracle-fixer", help="Oracle locations into Fixer. Research harness, not e2e.")
    p_oracle.add_argument("--workspace", type=Path, default=Path("/tmp/metric-agent-oracle-fixer"))
    p_oracle.add_argument("--presets", default="fedotllm:openrouter")
    p_oracle.add_argument("--fedot", type=Path, default=None)
    p_oracle.add_argument("--checkout", type=Path, default=None)
    p_oracle.add_argument("--cases", default="pca,knn,imputation")
    p_oracle.add_argument("--mode", default=None, help="If set, one arm instead of the compact matrix")
    p_oracle.add_argument("--samples", type=int, default=1)
    p_oracle.add_argument("--max-edits", type=int, default=1)
    p_oracle.add_argument("--prompt-arm", default="stock", choices=("stock", "hypothesis"))
    p_oracle.add_argument("--dev", action="store_true")
    p_oracle.add_argument("--no-dev", action="store_true")
    p_oracle.add_argument("--matrix", action="store_true", help="Run the compact 1-vs-3 / context / edits matrix")

    args = parser.parse_args(argv)
    _bind_fedot(getattr(args, "fedot", None))

    if args.cmd == "eval-contract":
        from research.evolve.metric_agent.loop import eval_contract

        payload = eval_contract()
        print(json.dumps(payload, indent=2, default=str))
        scores = payload["scores"]
        ok = (
            payload["guard_cases"] == "deny"
            and payload["guard_scorer"] == "deny"
            and scores.get("pca->catboost", {}).get("status") == "crash"
            and scores.get("catboost", {}).get("status") == "ok"
            and scores.get("fast_ica->lgbm", {}).get("status") == "ok"
        )
        return 0 if ok else 1

    from research.evolve.metric_agent.checkout import make_disposable_checkout, resolve_fedot_src

    if args.cmd == "leads":
        from research.evolve.metric_agent.discover import discover_leads

        checkout = args.checkout or resolve_fedot_src()
        leads = discover_leads(checkout, inference=_inference(args.presets))
        print(json.dumps([asdict(lead) for lead in leads], indent=2))
        return 0 if leads else 1

    if args.cmd == "scoreboard":
        from research.evolve.metric_agent.scoreboard import summarize

        print(json.dumps(summarize(args.workspace), indent=2))
        return 0

    if args.cmd == "replay":
        from research.evolve.metric_agent.replay import load_replay

        payload = load_replay(args.workspace, candidate=args.candidate)
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload else 1

    if args.cmd == "oracle-fixer":
        os.environ.setdefault("METRIC_AGENT_SEED", "42")
        from research.evolve.metric_agent.oracle_bench import (
            CASES,
            compact_matrix,
            dump_context_probe,
            run_arm,
            summarize,
        )

        inference = _inference(args.presets)
        if inference is None:
            print("KEY_MISSING: FEDOTLLM_LLM_API_KEY is not set; context probe still runs", file=sys.stderr)
        checkout = args.checkout or make_disposable_checkout()
        workspace = args.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        probe = dump_context_probe(checkout, workspace)
        case_ids = tuple(part.strip() for part in args.cases.split(",") if part.strip())
        run_dev = False if args.no_dev else (True if args.dev else None)
        if args.matrix or args.mode is None:
            records = compact_matrix(
                checkout=checkout,
                inference=inference,
                workspace=workspace,
                cases=case_ids,
                run_dev=False if args.no_dev else (True if args.dev else None),
            )
        else:
            records = []
            for case_id in case_ids:
                records.extend(
                    run_arm(
                        checkout=checkout,
                        case=CASES[case_id],
                        inference=inference,
                        workspace=workspace,
                        mode=args.mode,
                        samples=args.samples,
                        max_edits=args.max_edits,
                        prompt_arm=args.prompt_arm,
                        run_dev=bool(run_dev) if run_dev is not None else CASES[case_id].run_dev,
                    )
                )
        payload = summarize(records)
        payload["context_probe"] = probe
        payload["fedot"] = str(checkout)
        print(json.dumps(payload, indent=2, default=str))
        (workspace / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return 0 if payload.get("semantic_any") else 1

    from research.evolve.metric_agent.loop import run_once

    inference = _inference(args.presets)
    if inference is None:
        print("FEDOTLLM_LLM_API_KEY is not set; propose_patch will no-op", file=sys.stderr)
    checkout = args.checkout or make_disposable_checkout()
    decision = run_once(
        checkout=checkout,
        inference=inference,
        workspace=args.workspace,
    )
    print(
        json.dumps(
            {
                "keep": decision.keep,
                "reason": decision.reason,
                "target_delta": decision.target_delta,
                "regression_deltas": decision.regression_deltas,
            },
            indent=2,
        )
    )
    return 0 if decision.keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
