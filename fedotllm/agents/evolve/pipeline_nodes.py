from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage

from fedotllm.agents.evolve.audit import reset_journal_path, use_journal_path
from fedotllm.agents.evolve.leads import (
    CLASS_RANK,
    fixer_attempt_queue,
    fixer_candidates,
    is_run_success,
    union_reader_passes,
    verifier_candidates,
)
from fedotllm.agents.evolve.lint_scan import group_by_file, scan_repository
from fedotllm.agents.evolve.loop import (
    require_clean_repo,
    reset_evolution_changes,
    resolve_repo_python,
    run_evolution_loop,
    untracked_repo_files,
)
from fedotllm.agents.evolve.reader import run_reader_pass
from fedotllm.agents.evolve.results import FixerResult, ReaderResult, ScanResult, VerifierResult
from fedotllm.agents.evolve.semantic_scan import scan_repository as scan_semantics
from fedotllm.agents.evolve.state import EvolveAgentState
from fedotllm.agents.evolve.verifier import verify_leads
from fedotllm.configs.schema import EvolveConfig
from fedotllm.llm import AIInference
from fedotllm.log import logger


def _write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(path)


def _output_dir(state: EvolveAgentState) -> Path:
    root = Path(state["workspace"])
    path = root / "pipeline"
    path.mkdir(parents=True, exist_ok=True)
    state["pipeline_output"] = str(path)
    return path


def pipeline_config(state: EvolveAgentState) -> EvolveConfig:
    raw = state.get("evolve_config")
    if isinstance(raw, EvolveConfig):
        return raw
    if isinstance(raw, dict) and "reader_passes" in raw:
        return EvolveConfig.model_validate(raw)
    return EvolveConfig.from_env()


def _stage_config(
    state: EvolveAgentState,
    evolve_cfg: EvolveConfig | dict | None = None,
) -> EvolveConfig:
    """LangGraph also passes RunnableConfig as `config`; ignore that dict."""
    if isinstance(evolve_cfg, EvolveConfig):
        return evolve_cfg
    if isinstance(evolve_cfg, dict) and "reader_passes" in evolve_cfg:
        return EvolveConfig.model_validate(evolve_cfg)
    return pipeline_config(state)


def scan_stage(
    state: EvolveAgentState,
    evolve_cfg: EvolveConfig | None = None,
    **_: object,
) -> EvolveAgentState:
    """Stage 1: deterministic whole-repository lint seed."""
    cfg = _stage_config(state, evolve_cfg)
    state["pipeline_stages"] = [*state.get("pipeline_stages", []), "scan"]
    repo = Path(state["repo_path"])
    require_clean_repo(repo)
    output = _output_dir(state)
    state["pipeline_usage"] = {"scan": {"cost_usd": 0.0}}
    verified_path = cfg.verified_path
    if verified_path and Path(verified_path).is_file():
        rows = json.loads(Path(verified_path).read_text(encoding="utf-8"))
        scan = ScanResult(external_verified=True, empty=not rows)
        state["verified"] = rows
        state["fixer_queue"] = fixer_candidates(rows)
        state["pipeline_external_verified"] = True
        state["scan_empty"] = scan.empty
        return state
    findings = scan_repository(repo)
    semantic_findings = scan_semantics(
        repo,
        only_files=cfg.file_filter(),
        mode=cfg.semantic_mode,
    )
    scan = ScanResult(
        lint_findings=findings,
        semantic_findings=semantic_findings,
        empty=not findings and not semantic_findings,
    )
    state["lint_findings"] = findings
    state["semantic_findings"] = semantic_findings
    state["lint_path"] = _write_json(output / "lint.json", findings)
    state["semantic_path"] = _write_json(output / "semantic.json", semantic_findings)
    state["pipeline_external_verified"] = False
    state["scan_empty"] = scan.empty
    return state


def reader_stage(
    state: EvolveAgentState,
    inference: AIInference,
    evolve_cfg: EvolveConfig | None = None,
    inferences: list[AIInference] | None = None,
    **_: object,
) -> EvolveAgentState:
    """Stage 2: one full read of each file; union still works if extra passes are set."""
    cfg = _stage_config(state, evolve_cfg)
    state["pipeline_stages"] = [*state.get("pipeline_stages", []), "reader"]
    if state.get("pipeline_external_verified"):
        return state
    repo = Path(state["repo_path"])
    output = _output_dir(state)
    grouped = group_by_file(
        state.get("lint_findings", []),
        repo=repo,
        include_all=cfg.include_all_files,
        only_files=cfg.file_filter(),
    )
    python = resolve_repo_python(repo)

    # A pass may have its own model. Passes beyond the list of models reuse it
    # in order, so `reader_passes=4` with two models gives each model two runs.
    readers = list(inferences or [inference]) or [inference]
    passes_wanted = max(cfg.reader_passes, len(readers))

    def run_pass(index: int) -> list[dict]:
        return run_reader_pass(
            readers[index % len(readers)],
            repo,
            grouped,
            workers=cfg.reader_workers,
            limit_files=cfg.limit_files,
            python=python,
            workdir=output / "reader_runs",
        )

    with ThreadPoolExecutor(max_workers=passes_wanted) as pool:
        passes = list(pool.map(run_pass, range(passes_wanted)))
    semantic_findings = state.get("semantic_findings", [])
    lead_passes = [*passes]
    if semantic_findings:
        lead_passes.append(semantic_findings)
    leads = union_reader_passes(lead_passes)
    result = ReaderResult.from_passes(
        passes,
        scanned_files=bool(grouped),
        leads=leads,
    )
    if result.failed:
        raise RuntimeError(result.reason)
    state["reader_passes"] = result.passes
    state["leads"] = result.leads
    state["reader_empty"] = result.empty
    state["reader_reason"] = result.reason
    state["reader_paths"] = [
        _write_json(output / f"reader{index}.json", rows)
        for index, rows in enumerate(passes, 1)
    ]
    state["leads_path"] = _write_json(output / "leads.json", leads)
    state["pipeline_usage"] = {
        **state.get("pipeline_usage", {}),
        "reader": dict(inference.usage),
    }
    return state


def verifier_stage(
    state: EvolveAgentState,
    inference: AIInference,
    evolve_cfg: EvolveConfig | None = None,
    **_: object,
) -> EvolveAgentState:
    """Stage 3: model proposes proofs; the interpreter and route gates decide."""
    cfg = _stage_config(state, evolve_cfg)
    state["pipeline_stages"] = [*state.get("pipeline_stages", []), "verifier"]
    if state.get("pipeline_external_verified"):
        return state
    repo = Path(state["repo_path"])
    output = _output_dir(state)
    python = resolve_repo_python(repo)
    verified = verify_leads(
        inference,
        repo,
        verifier_candidates(state.get("leads", [])),
        python,
        output / "verify",
        workers=cfg.verifier_workers,
        limit=cfg.limit_leads,
        config=cfg,
    )
    queue = fixer_candidates(verified)
    result = VerifierResult(
        verified=verified,
        fixer_queue=queue,
        empty=not queue,
        failed=bool(state.get("leads")) and bool(verified) and all(
            row.get("status") == "error" for row in verified
        ),
        reason=(
            "verifier stage failed for every lead; refusing false abstention"
            if verified and state.get("leads") and all(row.get("status") == "error" for row in verified)
            else ""
        ),
    )
    if result.failed:
        raise RuntimeError(result.reason)
    queue.sort(
        key=lambda row: (
            CLASS_RANK.get(row.get("defect_class"), 9),
            not row.get("agreed", False),
            row.get("file", ""),
            int(row.get("line", 0)),
        )
    )
    state["verified"] = verified
    state["fixer_queue"] = queue
    state["verifier_empty"] = result.empty
    state["verified_path"] = _write_json(output / "verified.json", verified)
    state["pipeline_usage"] = {
        **state.get("pipeline_usage", {}),
        "verifier": dict(inference.usage),
    }
    return state


def fixer_stage(
    state: EvolveAgentState,
    inference: AIInference,
    evolve_cfg: EvolveConfig | None = None,
    **_: object,
) -> EvolveAgentState:
    """Stage 4: attempt every queued public defect; keep going after a green patch."""
    cfg = _stage_config(state, evolve_cfg)
    state["pipeline_stages"] = [*state.get("pipeline_stages", []), "fixer"]
    queue = state.get("fixer_queue", [])
    attempt = fixer_attempt_queue(
        queue,
        max_unproven=cfg.max_unproven_fixes,
        max_total=cfg.max_fixer_candidates,
    )
    if not attempt:
        reason = (
            "no confirmed defect for the fixer"
            if queue
            else (
                "no reader suspicion left for the fixer "
                "(verifier refuted every lead or none were raised)"
            )
        )
        if state.get("reader_empty"):
            reason = "reader found no defects"
        outcome = FixerResult(success=False, abstained=True, abstain_reason=reason)
        state["evolve_success"] = outcome.success
        state["evolve_abstained"] = outcome.abstained
        state["evolve_abstain_reason"] = outcome.abstain_reason
        state["pipeline_usage"] = {
            **state.get("pipeline_usage", {}),
            "fixer": {"cost_usd": 0.0},
        }
        state["messages"] = state["messages"] + [
            AIMessage(content="EvolveAgent abstained: no confirmed defect to patch.")
        ]
        return state

    repo = Path(state["repo_path"])
    output = _output_dir(state)
    result = None
    selected = attempt[0] if attempt else {}
    won = False
    journal_token = None
    attempts: list[dict] = []
    try:
        if cfg.probe_gate:
            from fedotllm.agents.evolve.probe import run_probe_cached

            run_probe_cached(repo, resolve_repo_python(repo))
        journal_token = use_journal_path(output / "evolution_journal.jsonl")
        won_result = None
        won_selected = None
        for selected in attempt:
            require_clean_repo(repo)
            slug = re.sub(
                r"[^a-z0-9]+",
                "-",
                f"{selected.get('file', 'unknown')}-{selected.get('line', 0)}".lower(),
            ).strip("-")
            evidence_path = output / "fixer" / slug / "verified.json"
            _write_json(evidence_path, [selected])
            result = run_evolution_loop(
                inference=inference,
                repo=repo,
                workspace=output / "fixer" / slug,
                config=cfg,
                verified_path=evidence_path,
                journal_file=output / "evolution_journal.jsonl",
            )
            accepted = is_run_success(selected, result)
            attempts.append(
                {
                    "file": selected.get("file"),
                    "line": selected.get("line"),
                    "route": selected.get("route"),
                    "success": accepted,
                    "audit_path": result.audit_path,
                    "reason": result.abstain_reason,
                }
            )
            if accepted:
                won = True
                won_result = result
                won_selected = selected
            reset_evolution_changes(repo, untracked_repo_files(repo))
        _write_json(output / "fixer" / "attempts.json", attempts)
        if won_result is not None:
            result = won_result
            selected = won_selected
        if not won:
            require_clean_repo(repo)
        if result is None:
            raise RuntimeError("fixer produced no result")
    except Exception as exc:
        logger.exception("fixer failed")
        reason = f"{type(exc).__name__}: {exc}"
        if len(reason) > 800:
            reason = reason[:800] + "…"
        state["evolve_success"] = False
        state["evolve_abstained"] = True
        state["evolve_abstain_reason"] = reason
        state["pipeline_usage"] = {
            **state.get("pipeline_usage", {}),
            "fixer": dict(getattr(inference, "usage", {}) or {"cost_usd": 0.0}),
        }
        state["messages"] = state["messages"] + [
            AIMessage(content=f"EvolveAgent fixer failed: {reason}")
        ]
        return state
    finally:
        if journal_token is not None:
            reset_journal_path(journal_token)
    n_ok = sum(1 for row in attempts if row.get("success"))
    outcome = FixerResult(
        success=won,
        abstained=(not won) or result.abstained,
        abstain_reason=(
            result.abstain_reason
            if result.abstain_reason
            else (
                ""
                if won
                else "no confirmed patch passed the gates"
            )
        ),
        selected_defect=selected,
    )
    state["selected_defect"] = selected
    state["scout_pick"] = result.pick
    state["scout_why"] = result.why
    state["proposal"] = asdict(result.proposal)
    state["evolve_success"] = outcome.success
    state["evolve_severity"] = result.severity
    state["probe_resolved"] = result.probe_resolved
    state["evolve_abstained"] = outcome.abstained
    state["evolve_abstain_reason"] = outcome.abstain_reason
    state["evolve_evidence"] = result.evidence
    state["audit_path"] = result.audit_path
    state["audit_markdown"] = result.audit_markdown
    state["pipeline_usage"] = {
        **state.get("pipeline_usage", {}),
        "fixer": dict(inference.usage),
    }
    state["messages"] = state["messages"] + [
        AIMessage(
            content=(
                f"EvolveAgent {'SUCCESS' if won else 'INCOMPLETE'}: "
                f"{n_ok}/{len(attempts) or 1} patches. "
                f"`{result.pick}`. Audit: `{result.audit_path}`"
            )
        )
    ]
    return state
