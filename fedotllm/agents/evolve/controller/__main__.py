"""Independent CLI: python -m fedotllm.agents.evolve.controller."""

import argparse
import json
import sys
from pathlib import Path


def add_service_arguments(parser):
    parser.add_argument("--stock-cache", type=Path, default=None)
    for name, default in (("duration-seconds", 604800), ("poll-seconds", 30),
                          ("controller-timeout-seconds", 43200), ("hunt-timeout-seconds", 3600),
                          ("llm-max-cost-usd", .25), ("hunt-total-cost-usd", 10)):
        parser.add_argument(f"--{name}", type=float, default=default)
    for name, default in (("max-controller-restarts", 3), ("target-proposals", 100),
                          ("max-hunt-campaigns", 200), ("llm-max-queries", 30)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--hunt", action="store_true")
    parser.add_argument("--presets", default="fedotllm:openrouter")


def serve(args, parser):
    from fedotllm.agents.evolve.controller.supervisor import QualityServiceConfig, run_quality_service
    from fedotllm.agents.evolve.execution.checkout import resolve_fedot_src
    config = QualityServiceConfig(**{
        name: getattr(args, name) for name in QualityServiceConfig.__dataclass_fields__
    })
    try:
        payload = run_quality_service(resolve_fedot_src(), args.workspace,
                                      config=config, stock_cache=args.stock_cache)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return 0 if payload.get("completion") == "all_measured" else 3


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "launch":
        from fedotllm.agents.evolve.controller.launch import main as launch
        return launch(argv[1:])
    parser = argparse.ArgumentParser(description="FEDOT.LLM Evolve Controller")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show module location and release hash")
    status = sub.add_parser("status", help="Read one run's saved status without launching work")
    status.add_argument("--workspace", type=Path, required=True)
    diagnosis = sub.add_parser("diagnose", help="Explain saved progress and stale stages without changing the run")
    diagnosis.add_argument("--workspace", type=Path, required=True)
    diagnosis.add_argument("--stale-supervisor-seconds", type=float, default=900)
    diagnosis.add_argument("--stale-worker-seconds", type=float, default=3600)
    sub.add_parser("launch", help="Check or launch a new run; default is read-only")
    for name, help_text in (("serve", "Serve a quality campaign"), ("drain", "Run queued improvement checks")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--fedot", type=Path, required=True)
        command.add_argument("--workspace", type=Path, required=True)
        if name == "serve":
            add_service_arguments(command)
        else:
            command.add_argument("--stock-cache", type=Path, default=None)
            command.add_argument("--tasks", default="")
            command.add_argument("--n-jobs", type=int, default=4)
            command.add_argument("--cpu-quota", type=int, default=28)
            command.add_argument("--max-pair-slots", type=int, default=7)
            command.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "info":
        from fedotllm.agents.evolve.controller.release import controller_version
        print(json.dumps({"name": "FEDOT.LLM Evolve Controller", "module": __package__,
                          "path": str(Path(__file__).parent), "controller_sha256": controller_version()}, indent=2))
        return 0
    if args.command == "status":
        path = args.workspace / "logs/service_status.json"
        if not path.is_file():
            path = args.workspace / "service_status.json"
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            parser.error(f"invalid status object: {path}")
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "diagnose":
        from fedotllm.agents.evolve.controller.diagnose import diagnose_run
        try:
            report = diagnose_run(
                args.workspace,
                stale_supervisor_seconds=args.stale_supervisor_seconds,
                stale_worker_seconds=args.stale_worker_seconds,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot diagnose saved run: {type(exc).__name__}: {exc}")
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0 if report.get("state") in {"progressing", "terminal"} else 2
    import os
    os.environ["FEDOTLLM_REPO_PATH"] = str(args.fedot.resolve())
    if args.command == "serve":
        return serve(args, parser)
    os.environ["EVOLVE_AGENT_TMP"] = str(args.workspace.resolve() / "worker_artifacts")
    from fedotllm.agents.evolve.controller.quality_executor import drain_quality_queue
    try:
        payload = drain_quality_queue(args.fedot.resolve(), args.workspace,
                                      task_ids=tuple(filter(None, args.tasks.split(","))) or None,
                                      n_jobs=args.n_jobs, cpu_quota=args.cpu_quota,
                                      max_pair_slots=args.max_pair_slots,
                                      stock_cache=args.stock_cache, continue_on_error=args.continue_on_error)
    except OSError as exc:
        if exc.errno not in {28, 122}:
            raise
        print(json.dumps({"state":"storage_error", "completion":"incomplete",
                          "error":str(exc), "notes":getattr(exc, "__notes__", [])}), flush=True)
        return 75
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return 3 if any(job.get("status") not in {"keep", "drop"} for job in payload["jobs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
