"""LLM catalog walk, causal pick validation and localization accounting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from fedotllm.log import logger
from fedotllm.agents.evolve.agents.failures import classify_model_failure
from fedotllm.agents.evolve.storage.llm_audit import capture_structured_create
from fedotllm.agents.evolve.storage.run_budget import EvolveBudgetExhausted
from fedotllm.agents.evolve.discovery.context import (
    _callee_sources,
    open_runtime,
    scout_source_context,
    show_file,
    show_source,
)
from fedotllm.agents.evolve.execution.guard import deny_write
from fedotllm.agents.evolve.discovery.repo_map import in_metric_scan
from fedotllm.agents.evolve.discovery.navigation import architecture_cards
from fedotllm.agents.evolve.storage.replay import semantic_site_id
from fedotllm.agents.evolve.types import PatchSite, ToolAction

_RAW_PICK_CHARS = 32_000

_SCOUT_SOURCE_CHARS = 12_000

_SCOUT_EXTRA_CHARS = 6_000

_SCOUT_TOOL_HISTORY_CHARS = 6_000

_SCOUT_OUTPUT_CHARS = 4_000

_FILE_COVERAGE_PREFIX = "executed metric-bearing lines in file:"

_METADATA_REGISTRY_FILES = {
    "fedot/core/repository/data/model_repository.json",
    "fedot/core/repository/data/data_operation_repository.json",
    "fedot/core/repository/data/gpu_models_repository.json",
}

_LOCATE = """You inspect one FEDOT runtime source file from the catalog.

Your primary goal is one source change with a plausible causal effect on general
predictive behavior or correctness. It may repair a bug/silent contract mismatch, or
improve valid stock behavior with a more suitable algorithm, representation, adaptive
rule, numerical method, or use of information that is currently discarded. Do not
classify the lead in advance. The exact benchmark dataset, score and metric identity
are intentionally withheld during discovery: do not tune a proposal to one named
metric. Explain how data, fitted state or predictions would change across the affected
task family; the controller will choose suitable metrics after the hypothesis exists.

"Use something newer" is not a mechanism. Compare the current behavior with a
concrete alternative implementable with existing dependencies. Preserve unrelated
task families and sibling operations. DEV decides whether it is better. If DEV is
neutral, the change is useful only when an independent stock failure is reproduced
and the same probe passes after patching. You do not have to pick this file.

status=pick — record this file (or an opened neighbor) as a patch site. The catalog
walk continues with the next file; pick does not stop the scan. Fill `mechanism`,
`proposed_change`, `expected_metric_effect`, `hypothesis_kind`, and `change_line`.
Use hypothesis_kind=correctness only for a concrete violation of a stable data,
parameter, lifecycle, or fit/predict contract which a public runtime probe can
reproduce without consulting a metric. Use quality for valid behavior where an
alternative must win a metric comparison. `line` is the source
line you inspected; `change_line` is the exact first source line your proposed
change would alter. They may differ. A structured pick without `change_line` is
invalid. For `default_operation_params.json`, also fill `operation_id` with exactly
one operation named by an `executed operation:` evidence line. The controller
checks that an existing JSON block belongs to that operation, or that a newly added
block explicitly names it. Do not select an unexecuted sibling default merely
because it is visible in the same shared JSON file. Do not return pick while saying
the site is low-value, a passthrough, has
no causal mechanism, or should be skipped.
status=skip — nothing useful here; the next catalog file will be shown.
status=run — put Python in run_code. It runs in the FEDOT checkout (`import fedot`
works). Output comes back, then you pick, skip, read, or run again.
status=read — put another fedot/ runtime path in file_path. That file is appended
so you can use it as context. You may then pick a line in this file or the neighbor.
status=search — put literal source text in query to search FEDOT Python and
frozen repository JSON metadata.
status=symbol — put a class/function/method name in query to retrieve its body.
status=callers — put a function/method name in symbol to retrieve its call sites.
status=docs — put a FEDOT concept, class, or operation name in query to retrieve
relevant frozen FEDOT documentation, docstrings, and operation metadata.
Use these navigation tools when the shown function references code you cannot see;
do not guess an unseen producer, consumer, helper, or caller.
FEDOT's model/data operation repository JSON files describe registration metadata
such as implementation, tags and presets. They do not supply estimator defaults.
Runtime defaults belong in `default_operation_params.json`; do not invent a
`default_params` field in another registry unless source retrieval shows a real
consumer for that field.

Runtime evidence below comes from the real frozen workload. A traceback leaf can
be only the downstream symptom of a transformation that produced invalid data.
Inspect operation paths named in the evidence before patching the leaf. A `run`
experiment must preserve the observed workload invariant; do not manufacture the
same exception with deliberately invalid indexes or impossible input state.
`executed lines in this symbol` is exact coverage, not an approximate file hit.
Prefer mechanisms on those lines or their inputs/conditions. A branch absent from
that list may still contain a library defect, but changing only that branch cannot
explain a metric change on the measured workload.
The train/evaluation boundary is fixed. Never improve a score by making validation,
DEV, or FINAL target rows available to fit; that is leakage, not a library improvement.

Do not name a path outside fedot/ runtime source.
Do not import the scoring harness, datasets, or case catalogs.
Do not choose plotting, visualization, logging-only, cache-only, remote, tests,
or research-harness code."""


class SiteProposal(BaseModel):
    """LLM verdict on the file just shown: pick, skip, or run a snippet."""

    status: ToolAction = Field(
        default=ToolAction.PICK,
        description="pick, skip, run, read, search, symbol, or callers",
    )
    file_path: str = Field(
        default="",
        description="Path relative to FEDOT checkout, e.g. fedot/core/foo.py",
    )
    line: int = Field(default=1, ge=1)
    change_line: int = Field(
        default=0,
        ge=0,
        description=(
            "Exact first source line that proposed_change would alter; required "
            "for a structured pick"
        ),
    )
    why: str = ""
    run_code: str = Field(default="", description="Python to execute when status=run")
    query: str = Field(
        default="", description="Literal source query for search or symbol"
    )
    symbol: str = Field(default="", description="Function/method name for callers")
    mechanism: str = Field(
        default="",
        description="Causal path from this executed source behavior to the measured metric",
    )
    proposed_change: str = Field(
        default="",
        description="Concrete source-level alternative worth independent verification",
    )
    expected_metric_effect: str = Field(
        default="",
        description="Why the alternative may improve the named frozen metric",
    )
    hypothesis_kind: Literal["correctness", "quality"] = Field(
        default="quality",
        description=(
            "correctness for a reproducible contract violation; quality for a "
            "valid alternative whose value depends on metric evaluation"
        ),
    )
    operation_id: str = Field(
        default="",
        description=(
            "Executed operation affected by a default_operation_params.json pick; "
            "empty for ordinary source files"
        ),
    )


class CatalogPlan(BaseModel):
    """Cheap first-pass ranking after seeing the complete source catalog."""

    selected_indices: list[int] = Field(default_factory=list, max_length=3)
    rationale: str = ""


_SELF_REJECTING_PICK = re.compile(
    r"(?:\bno\s+(?:meaningful|plausible|credible|causal)\b|"
    r"\blow[- ]value\b|\bbetter\s+to\s+skip\b|\bshould\s+skip\b|"
    r"\buse\s+skip\b|\breal\s+lead\s+is\b|\bno\s+code\s+change\b)",
    re.IGNORECASE,
)


def _recent_tool_context(
    history: list[str], limit: int = _SCOUT_TOOL_HISTORY_CHARS
) -> str:
    """Keep the latest tool's identity and source start when evicting old output."""

    if not history:
        return ""
    latest = history[-1]
    if len(latest) > limit:
        marker = "\n...[middle truncated]...\n"
        head = (limit - len(marker)) // 2
        return latest[:head] + marker + latest[-(limit - len(marker) - head) :]
    selected = [latest]
    remaining = limit - len(latest)
    for previous in reversed(history[:-1]):
        if len(previous) + 2 > remaining:
            break
        selected.insert(0, previous)
        remaining -= len(previous) + 2
    return "\n\n".join(selected)


def _pick_claim_is_positive(parsed: SiteProposal) -> bool:
    """Accept only a complete causal claim that does not semantically abstain."""

    explanation = " ".join(
        filter(
            None,
            (
                parsed.why,
                parsed.mechanism,
                parsed.proposed_change,
                parsed.expected_metric_effect,
            ),
        )
    )
    if _SELF_REJECTING_PICK.search(explanation):
        return False
    structured = (
        parsed.mechanism.strip(),
        parsed.proposed_change.strip(),
        parsed.expected_metric_effect.strip(),
    )
    return all(structured)


def _pick_change_line(parsed: SiteProposal, fallback: int) -> int | None:
    """Resolve the actual edit location, rejecting incomplete causal claims.

    The model must distinguish the inspected anchor from the first line it
    proposes to modify.  This prevents an executed nearby line from laundering a
    proposed change to an unexecuted sibling branch.
    """

    structured = all(
        (
            parsed.mechanism.strip(),
            parsed.proposed_change.strip(),
            parsed.expected_metric_effect.strip(),
        )
    )
    if not structured or parsed.change_line <= 0:
        return None
    return max(1, int(parsed.change_line))


def _impact(lead: PatchSite) -> int:
    """Compose bonuses. Call graph is not a hard gate; package is not a hard gate."""

    score = 0
    sigs = set(lead.signals)
    if "row_identity_contract" in sigs:
        # This is a high-confidence silent-corruption pattern: independently
        # filtered parent matrices are later joined by position. Put it before
        # broad registry/default opportunities so Scout can inspect it within
        # a small action and cost budget.
        score += 6
    if "registry" in sigs or "defaults" in sigs:
        score += 3
    if "data_plane" in sigs or "pipeline" in sigs:
        score += 2
    if "reachable" in sigs:
        score += 3
    if "impl_class" in sigs:
        score += 2
    if "same_class" in sigs or "same_module" in sigs:
        score += 1
    if "core_scan" in sigs:
        score += 1
    path = lead.file_path
    if "/operation_implementations/" in path or path.startswith(
        ("fedot/preprocessing/", "fedot/core/data")
    ):
        score += 2
    elif path.startswith(
        (
            "fedot/core/operations/",
            "fedot/core/pipelines",
            "fedot/core/composer",
            "fedot/core/optimisers",
            "fedot/core/repository",
            "fedot/api/",
        )
    ):
        score += 1
    return score


def _metric_path(path: str) -> bool:
    """False for plots, cache, remote, explainability."""

    # The frozen quality protocol executes already-built pipelines. Operation
    # registry metadata controls discovery/composition, so editing it cannot
    # change any fitted model or prediction in this protocol. It remains
    # available through Scout read/search/docs as navigation evidence.
    return path not in _METADATA_REGISTRY_FILES and in_metric_scan(path)


def _execution_causal_priority(lead: PatchSite) -> int:
    """Rank measured data-plane code above high-hit registry/dispatch plumbing."""

    path = lead.file_path.lower()
    symbol = lead.why.removeprefix("executed symbol ").split(".", 1)[0]

    def class_key(name: str) -> str:
        token = re.sub(r"[^a-z0-9]", "", name.lower())
        for suffix in (
            "transformationimplementation",
            "implementation",
            "transformation",
        ):
            if token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    token = class_key(symbol)
    implementations: list[str] = []
    for item in lead.evidence:
        if not item.startswith("runtime operation instances:"):
            continue
        for instance in item.partition(":")[2].split("|"):
            operation_and_implementation = instance.strip().split(" ", 1)[0]
            if "/" in operation_and_implementation:
                implementations.append(operation_and_implementation.split("/", 1)[1])
    runtime_classes = {class_key(name) for name in implementations}
    # A dispatcher named Fedot is not the concrete FedotLightGBM... class.
    # Substring matching promoted generic API plumbing above real dataflow.
    if token and token in runtime_classes:
        return 0
    if "definition" in lead.signals:
        return 5
    # coverage.py records class declaration lines while importing operation
    # modules. Without an executed method/function or a concrete dataflow-class
    # match, this is import reachability rather than a metric-bearing call path.
    if "class" in lead.signals:
        return 5
    # Registry-backed strategies, interfaces and factories are reachable
    # plumbing.  A high line count there is weaker causal evidence than an
    # executed concrete implementation or data transformation.
    if "registry" in lead.signals or any(
        marker in path
        for marker in (
            "interface.py",
            "interfaces.py",
            "/factory.py",
        )
    ):
        return 4
    if "/operation_implementations/" in path:
        return 1
    if any(
        marker in path
        for marker in (
            "/core/data/",
            "/preprocessing/",
            "/operations/evaluation/",
        )
    ):
        return 2
    if any(
        marker in path
        for marker in (
            "/repository/",
            "/api/",
            "/caching/",
            "/pipelines/",
            "/optimisers/",
            "/composer/",
        )
    ) or path.endswith("/operations/operation.py"):
        return 4
    return 3


def _unique(leads: list[PatchSite]) -> list[PatchSite]:
    by_key: dict[tuple[str, int], PatchSite] = {}
    order: list[tuple[str, int]] = []
    for lead in leads:
        key = (lead.file_path, lead.line)
        old = by_key.get(key)
        if old is None:
            by_key[key] = lead
            order.append(key)
            continue
        sigs = tuple(dict.fromkeys((*old.signals, *lead.signals)))
        evidence = old.evidence or lead.evidence
        if old.channel == "llm" or lead.channel == "llm":
            channel = "llm"
            why = (old.why if old.channel == "llm" else lead.why) or old.why or lead.why
        else:
            channel = old.channel
            why = old.why or lead.why
        by_key[key] = PatchSite(
            channel=channel,
            file_path=old.file_path,
            line=old.line,
            why=why,
            evidence=evidence,
            signals=sigs,
            mechanism=(old.mechanism if old.channel == "llm" else lead.mechanism)
            or old.mechanism
            or lead.mechanism,
            proposed_change=(
                old.proposed_change if old.channel == "llm" else lead.proposed_change
            )
            or old.proposed_change
            or lead.proposed_change,
            expected_metric_effect=(
                old.expected_metric_effect
                if old.channel == "llm"
                else lead.expected_metric_effect
            )
            or old.expected_metric_effect
            or lead.expected_metric_effect,
        )
    return [by_key[key] for key in order]


def pool_rows(leads: list[PatchSite], *, picked: PatchSite | None = None) -> list[dict]:
    _ = picked
    rows = []
    for rank, lead in enumerate(leads, start=1):
        rows.append(
            {
                "rank": rank,
                "file_path": lead.file_path,
                "line": lead.line,
                "symbol": lead.why,
                "channel": lead.channel,
                "signals": list(lead.signals),
                "impact": _impact(lead),
                "llm_picked": lead.channel == "llm",
            }
        )
    return rows


def rank_of(rows: list[dict], file_path: str, line: int) -> int | None:
    for row in rows:
        if row.get("file_path") == file_path and int(row.get("line") or 0) == int(line):
            return int(row["rank"])
    return None


def localization(
    file_path: str,
    line: int,
    *,
    static_rows: list[dict],
    final_rows: list[dict],
    llm_pick: dict | None,
) -> dict:
    """static_rank vs final_rank. rank_delta > 0 means LLM moved the site up."""

    static_rank = rank_of(static_rows, file_path, line)
    final_rank = rank_of(final_rows, file_path, line)
    delta = None
    if static_rank is not None and final_rank is not None:
        delta = static_rank - final_rank
    picked = False
    if llm_pick:
        picked = llm_pick.get("file_path") == file_path and int(
            llm_pick.get("line") or 0
        ) == int(line)
    return {
        "static_rank": static_rank,
        "final_rank": final_rank,
        "rank_delta": delta,
        "picked_by_llm": picked,
    }


def annotate_pool_rows(
    final_rows: list[dict],
    *,
    static_rows: list[dict],
    llm_pick: dict | None,
) -> list[dict]:
    return [
        {
            **row,
            **localization(
                row["file_path"],
                int(row["line"]),
                static_rows=static_rows,
                final_rows=final_rows,
                llm_pick=llm_pick,
            ),
        }
        for row in final_rows
    ]


def _clip_raw(raw: str | None) -> str | None:
    if not raw:
        return None
    if len(raw) <= _RAW_PICK_CHARS:
        return raw
    return raw[:_RAW_PICK_CHARS] + "\n…[truncated]"


def _create_pick(inference, prompt, *, metadata: dict | None = None):
    return capture_structured_create(
        inference,
        prompt,
        SiteProposal,
        stage="scout",
        metadata=metadata,
    )


def _pick_files(leads: list[PatchSite]) -> list[PatchSite]:
    """One catalog site per file, in ranked order."""

    seen: set[str] = set()
    out: list[PatchSite] = []
    for lead in leads:
        if lead.file_path in seen:
            continue
        seen.add(lead.file_path)
        out.append(lead)
    return out


def _compact_line_numbers(lines: set[int]) -> str:
    ordered = sorted(int(line) for line in lines if int(line) > 0)
    if not ordered:
        return ""
    ranges: list[str] = []
    first = previous = ordered[0]
    for line in ordered[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = line
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def _metric_line_is_executed(line: int, evidence: tuple[str, ...]) -> bool | None:
    """True/False for measured files, None when no exact file coverage exists."""

    payload = next(
        (
            item.split(":", 1)[1].strip()
            for item in evidence
            if item.startswith(_FILE_COVERAGE_PREFIX)
        ),
        "",
    )
    if not payload:
        return None
    for token in payload.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                first, last = (int(part) for part in token.split("-", 1))
            else:
                first = last = int(token)
        except ValueError:
            continue
        if first <= int(line) <= last:
            return True
    return False


def _default_pick_matches_executed_operation(
    parsed: SiteProposal,
    picked: PatchSite,
    checkout: Path,
) -> bool:
    """Keep a shared defaults-file pick tied to one measured operation."""

    operation = parsed.operation_id.strip()
    executed = {
        item.split(":", 1)[1].strip()
        for item in picked.evidence
        if item.startswith("executed operation:")
    }
    if not operation or operation not in executed:
        return False
    path = checkout / picked.file_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if operation not in payload:
        return bool(
            re.search(
                rf'["\']{re.escape(operation)}["\']\s*:',
                parsed.proposed_change,
            )
        )
    owner = ""
    top_level_key = re.compile(r'^\s{2}"([^"]+)"\s*:')
    for text in lines[: max(1, picked.line)]:
        match = top_level_key.match(text)
        if match:
            owner = match.group(1)
    return owner == operation


def _configuration_assignment_signature(text: str) -> frozenset[tuple[str, str]]:
    """Extract concrete JSON-like scalar assignments from a proposal."""

    pairs = re.findall(
        r'["\']([A-Za-z_][\w-]*)["\']\s*:\s*'
        r'(true|false|null|-?\d+(?:\.\d+)?|["\'][^"\']+["\'])',
        text,
        flags=re.IGNORECASE,
    )
    return frozenset((key.lower(), value.strip("\"'").lower()) for key, value in pairs)


def _json_after_marker(text: str, marker: str) -> dict:
    position = text.find(marker)
    if position < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(
            text[position + len(marker) :].lstrip()
        )
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_scalar(value) -> str:
    if isinstance(value, str):
        return value.lower()
    return json.dumps(value, sort_keys=True).lower()


def _default_pick_changes_effective_value(
    parsed: SiteProposal,
    picked: PatchSite,
) -> bool:
    """Reject explicit defaults that exactly restate current runtime values."""

    proposed = _configuration_assignment_signature(parsed.proposed_change)
    if not proposed:
        return True
    current: dict = {}
    for item in picked.evidence:
        current.update(_json_after_marker(item, "estimator defaults:"))
        current.update(_json_after_marker(item, "current FEDOT defaults:"))
        current.update(_json_after_marker(item, "effective params:"))
    if not current:
        return True
    return any(
        key not in current or _canonical_scalar(current[key]) != value
        for key, value in proposed
    )


def _matches_recent_concrete_proposal(
    parsed: SiteProposal,
    file_path: str,
    prior_hypotheses: list[dict],
) -> bool:
    """Reject only the same concrete defaults assignment, not an operation family."""

    if not file_path.endswith("/default_operation_params.json"):
        return False
    signature = _configuration_assignment_signature(parsed.proposed_change)
    if not signature:
        return False
    return any(
        str(row.get("file_path") or "").lstrip("/") == file_path
        and _configuration_assignment_signature(str(row.get("proposed_change") or ""))
        == signature
        for row in prior_hypotheses
    )


_DEFECT_MEMORY_STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "both",
    "could",
    "data",
    "does",
    "from",
    "have",
    "into",
    "only",
    "same",
    "that",
    "their",
    "then",
    "this",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def _defect_terms(*parts: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z_][a-z0-9_+-]{2,}", " ".join(parts).lower())
        if token not in _DEFECT_MEMORY_STOPWORDS
    }


def _matches_confirmed_defect_proposal(
    parsed: SiteProposal,
    file_path: str,
    prior_hypotheses: list[dict],
) -> bool:
    """Detect the same defect mechanism while keeping the source site eligible."""

    proposed = _defect_terms(parsed.why, parsed.mechanism, parsed.proposed_change)
    if len(proposed) < 8:
        return False
    for row in prior_hypotheses:
        if row.get("history_kind") != "confirmed_defect":
            continue
        if str(row.get("file_path") or "").lstrip("/") != file_path:
            continue
        known = _defect_terms(
            str(row.get("mechanism") or ""),
            str(row.get("proposed_change") or ""),
        )
        shared = len(proposed & known)
        if shared >= 8 and shared / min(len(proposed), len(known)) >= 0.4:
            return True
    return False


def _configuration_pick_matches_executed_operation(
    parsed: SiteProposal,
    picked: PatchSite,
    checkout: Path,
) -> bool:
    """Keep a shared defaults-file pick tied to one measured operation."""

    if not picked.file_path.endswith("/default_operation_params.json"):
        return True
    executed = {
        item.split(":", 1)[1].strip()
        for item in picked.evidence
        if item.startswith("executed operation:")
    }
    operation = parsed.operation_id.strip()
    if not operation or operation not in executed:
        return False
    path = checkout / picked.file_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if operation not in payload:
        claim = " ".join((parsed.proposed_change, parsed.mechanism))
        return bool(re.search(rf'["\']{re.escape(operation)}["\']', claim))
    owner = ""
    for text in lines[: max(1, picked.line)]:
        match = re.match(r'^  "([^"]+)"\s*:', text)
        if match:
            owner = match.group(1)
    return owner == operation


def _accept_pick(
    parsed: SiteProposal,
    shown: set[str],
    checkout: Path,
    *,
    default_file: str = "",
    default_line: int = 1,
) -> PatchSite | None:
    if (parsed.status or "pick").strip().lower() == "skip":
        return None
    if not _pick_claim_is_positive(parsed):
        return None
    # A complete claim may omit only the current catalog path; causal fields and
    # change_line remain mandatory. Explicit neighboring paths still have to be
    # present in `shown`.
    used_default = not (parsed.file_path or "").strip()
    rel = (parsed.file_path or default_file or "").lstrip("/")
    if "fedot/" in rel and not rel.startswith("fedot/"):
        rel = rel[rel.index("fedot/") :]
    if rel not in shown:
        return None
    target = (checkout / rel).resolve()
    if deny_write(target, checkout=checkout) or not target.is_file():
        return None
    if not _metric_path(rel):
        return None
    inspected_line = max(1, default_line if used_default else int(parsed.line))
    line = _pick_change_line(parsed, inspected_line)
    if line is None:
        return None
    if not show_source(target, checkout=checkout, around=line):
        return None
    return PatchSite(
        channel="llm",
        file_path=rel,
        line=line,
        why=(
            parsed.why
            or "; ".join(
                filter(
                    None,
                    (
                        parsed.mechanism,
                        parsed.proposed_change,
                        parsed.expected_metric_effect,
                    ),
                )
            )
        ),
        mechanism=parsed.mechanism.strip(),
        proposed_change=parsed.proposed_change.strip(),
        expected_metric_effect=parsed.expected_metric_effect.strip(),
        hypothesis_kind=parsed.hypothesis_kind,
    )


def _open_requested_runtime(
    checkout: Path,
    requested_path: str,
    *,
    line: int = 1,
) -> tuple[str, str, bool]:
    """Open a requested FEDOT file, repairing one unambiguous path mistake.

    Small models often identify the right implementation class and basename but
    hallucinate an old/intermediate package directory.  Treating that as an empty
    read wastes the remaining tool turns and hides useful source.  A unique
    basename below ``fedot/`` is safe to recover; ambiguous names remain a miss
    instead of silently opening an arbitrary file.
    """

    rel = (requested_path or "").lstrip("/")
    if "fedot/" in rel and not rel.startswith("fedot/"):
        rel = rel[rel.index("fedot/") :]

    opened = show_source(rel, checkout=checkout, around=max(1, line), radius=20)
    if not opened:
        opened = open_runtime(checkout, rel, line=line)
    if opened:
        return rel, opened, False

    basename = Path(rel).name
    if not basename.endswith(".py"):
        return rel, "", False
    candidates: list[str] = []
    for candidate in sorted((checkout / "fedot").rglob(basename)):
        try:
            candidate_rel = (
                candidate.resolve().relative_to(checkout.resolve()).as_posix()
            )
        except (OSError, ValueError):
            continue
        if (
            candidate.is_file()
            and _metric_path(candidate_rel)
            and not deny_write(candidate, checkout=checkout)
        ):
            candidates.append(candidate_rel)
    if len(candidates) != 1:
        return rel, "", False

    corrected = candidates[0]
    opened = show_source(
        corrected,
        checkout=checkout,
        around=max(1, line),
        radius=20,
    )
    if not opened:
        opened = open_runtime(checkout, corrected, line=line)
    return corrected, opened, bool(opened)


def _llm_pick(
    inference,
    checkout: Path,
    leads: list[PatchSite],
    *,
    max_picks: int = 3,
    trace: dict | None = None,
    max_actions: int | None = None,
    max_runs_per_file: int = 2,
    excluded_sites: set[tuple[str, int]] | None = None,
    prior_hypotheses: list[dict] | None = None,
    excluded_semantic_sites: set[str] | None = None,
    present_full_catalog: bool = False,
    on_pick: Callable[[list[PatchSite]], None] | None = None,
) -> list[PatchSite]:
    """Walk catalog files. A pick is recorded; the walk continues until max_picks or the catalog ends."""

    from fedotllm.agents.evolve.execution.run_code import MAX_STEPS, run_fedot_snippet

    catalog = _pick_files(leads)
    n_files = len(catalog)
    want = max(1, max_picks)
    logger.info("evolve pick 0/%s files, want %s sites", n_files, want)
    raws: list[str] = []
    rounds: list[dict] = []
    found: list[PatchSite] = []
    # A final skip is meaningful evidence for this bounded catalog walk.  The
    # file may still be opened as dependency context, but selecting it again
    # from a later generic/dispatcher entry would recreate the same hypothesis
    # and starve the portfolio of independent metric paths.  This memory is
    # deliberately local to one Scout call; future campaigns reconsider files.
    declined_files: set[str] = set()
    evidence_by_file: dict[str, PatchSite] = {}
    for catalog_lead in leads:
        current = evidence_by_file.get(catalog_lead.file_path)
        has_crash = any(
            item.startswith("stock runtime crash:") for item in catalog_lead.evidence
        )
        current_has_crash = bool(
            current
            and any(
                item.startswith("stock runtime crash:") for item in current.evidence
            )
        )
        current_rank = (
            current_has_crash,
            bool(current and "upstream_of_crash" in current.signals),
            bool(current and current.evidence),
            bool(current and "executed" in current.signals),
        )
        candidate_rank = (
            has_crash,
            "upstream_of_crash" in catalog_lead.signals,
            bool(catalog_lead.evidence),
            "executed" in catalog_lead.signals,
        )
        winner = (
            catalog_lead
            if current is None or candidate_rank > current_rank
            else current
        )
        if current is None:
            evidence_by_file[catalog_lead.file_path] = catalog_lead
            continue
        evidence_by_file[catalog_lead.file_path] = PatchSite(
            channel=winner.channel,
            file_path=winner.file_path,
            line=winner.line,
            why=winner.why,
            evidence=tuple(dict.fromkeys((*current.evidence, *catalog_lead.evidence))),
            signals=tuple(
                dict.fromkeys(
                    (
                        *winner.signals,
                        *(
                            ("upstream_of_crash",)
                            if "upstream_of_crash"
                            in (
                                *current.signals,
                                *catalog_lead.signals,
                            )
                            else ()
                        ),
                    )
                )
            ),
            mechanism=winner.mechanism,
            proposed_change=winner.proposed_change,
            expected_metric_effect=winner.expected_metric_effect,
            hypothesis_kind=winner.hypothesis_kind,
        )
    seen: set[str] = set()
    reviewed_catalog_files: list[str] = []
    walked = 0
    actions = 0
    action_limit = max(1, max_actions) if max_actions is not None else 20
    budget_exhausted = False
    external_stop = ""
    catalog_plan: dict = {"presented": False, "selected_indices": []}
    if present_full_catalog and catalog and action_limit > 1:
        paths = tuple(item.file_path for item in catalog)
        cards = architecture_cards(checkout, paths, max_chars_per_file=500)
        annotations = "\n".join(
            f"[{index}] {item.file_path}:{item.line} | {item.why[:180]} | "
            f"signals={','.join(item.signals)}"
            for index, item in enumerate(catalog)
        )
        prompt = (
            "Review the complete FEDOT runtime catalog below before deep source "
            "inspection. Select up to three zero-based indices with the strongest "
            "independent correctness or predictive-quality mechanisms. Do not infer "
            "a solution from a filename alone; this is only a search order. Every "
            "catalog entry remains eligible later.\n\nArchitecture cards:\n"
            + cards
            + "\n\nRuntime ranking annotations:\n"
            + annotations
        )
        try:
            planned, raw = capture_structured_create(
                inference,
                prompt,
                CatalogPlan,
                stage="scout_catalog_plan",
                metadata={"catalog_size": n_files},
            )
            actions += 1
            raws.append(raw)
            indices = list(
                dict.fromkeys(
                    index
                    for index in planned.selected_indices
                    if 0 <= index < len(catalog)
                )
            )
            preferred = [catalog[index] for index in indices]
            preferred_ids = {id(item) for item in preferred}
            catalog = preferred + [item for item in catalog if id(item) not in preferred_ids]
            catalog_plan = {
                "presented": True,
                "selected_indices": indices,
                "rationale": planned.rationale[:2_000],
            }
        except Exception as exc:
            actions += 1
            if isinstance(exc, EvolveBudgetExhausted):
                budget_exhausted = True
                external_stop = "llm_budget_exhausted"
                rounds.append(
                    {
                        "file_path": "<catalog-plan>",
                        "status": "budget_exhausted",
                        "error_type": type(exc).__name__,
                    }
                )
                catalog_plan = {
                    "presented": True,
                    "selected_indices": [],
                    "error": str(exc)[:1_000],
                }
            else:
                failure = classify_model_failure(exc)
                if failure.infrastructure:
                    raise failure from exc
                catalog_plan = {
                    "presented": True,
                    "selected_indices": [],
                    "error": str(failure)[:1_000],
                }
    run_limit = max(0, max_runs_per_file)
    # Four turns may be spent gathering source/runtime evidence.  Reserve a
    # separate fifth turn for the actual pick/skip decision; previously a model
    # that used `run` on step 4 lost the file without ever being asked to decide.
    research_steps = min(4, max(1, MAX_STEPS - 1))
    per_file_steps = research_steps + 1
    excluded_exact = set(excluded_sites or ())
    excluded_semantic = set(excluded_semantic_sites or ())
    for lead in (() if external_stop else catalog):
        if len(found) >= want or actions >= action_limit:
            budget_exhausted = actions >= action_limit
            break
        if lead.file_path in seen:
            continue
        walked += 1
        # Scout ranks a symbol, not an entire module.  Whole 300–1000 line files
        # made inexpensive models spend their complete latency budget merely
        # reading 25–50k-character prompts.  The model can explicitly request a
        # neighboring symbol with status=read when the causal chain requires it.
        text = scout_source_context(
            lead,
            checkout=checkout,
            max_chars=_SCOUT_SOURCE_CHARS,
        )
        if not text:
            text = show_file(
                lead.file_path,
                checkout=checkout,
                max_chars=_SCOUT_SOURCE_CHARS,
            )
        text = text[:_SCOUT_SOURCE_CHARS]
        if not text:
            logger.info("evolve pick %s/%s empty %s", walked, n_files, lead.file_path)
            continue
        reviewed_catalog_files.append(lead.file_path)
        base_extra = ""
        if lead.evidence:
            base_extra += "\n\nRuntime evidence:\n" + "\n".join(
                f"- {item}" for item in lead.evidence
            )
        callees = _callee_sources(checkout, lead, skip_file=lead.file_path, limit=2)
        if callees:
            base_extra += f"\n\nCalled from this function:\n{callees}"
        from fedotllm.agents.evolve.discovery.knowledge import knowledge_for_lead

        architecture = knowledge_for_lead(checkout, lead, max_chars=4_000)
        # The measured evidence and original lead must remain visible even after
        # several tool calls.  Only the rolling tool history is evicted.
        base_extra = base_extra[:_SCOUT_EXTRA_CHARS]
        tool_history: list[str] = []
        shown = {lead.file_path}
        picked: PatchSite | None = None
        runs = 0
        for step in range(1, per_file_steps + 1):
            if actions >= action_limit:
                budget_exhausted = True
                rounds.append({"file_path": lead.file_path, "status": "action_limit"})
                break
            recent_tools = _recent_tool_context(tool_history)
            architecture_block = f"\n\n{architecture}" if architecture else ""
            prompt = (
                f"{_LOCATE}\n\n"
                f"Step {step}/{per_file_steps}; campaign actions "
                f"{actions}/{action_limit}.\n"
                f"Catalog site: {lead.file_path}:{lead.line} {lead.why}\n\n"
                f"File:\n{text}"
                f"{base_extra}"
                f"{architecture_block}"
                f"\n\nPrevious tool results:\n{recent_tools or '(none)'}"
            )
            if found:
                prompt += (
                    "\n\nAlready selected files (do not pick them again):\n- "
                    + "\n- ".join(item.file_path for item in found)
                    + "\nAlready selected causal hypotheses (search history):\n"
                    + json.dumps(
                        [
                            {
                                "file": item.file_path,
                                "line": item.line,
                                "mechanism": item.mechanism[:600],
                                "proposed_change": item.proposed_change[:400],
                            }
                            for item in found[-6:]
                        ],
                        ensure_ascii=False,
                    )
                    + "\nPrefer an independent predictive mechanism. A different "
                    "file handling the same failure or transformed state is an "
                    "alternative repair of an existing hypothesis, not a new "
                    "search direction. Skip such repeats in this catalog walk."
                )
            if declined_files:
                prompt += (
                    "\n\nFiles explicitly declined earlier in this catalog walk "
                    "(may be read as context, but do not pick them again):\n- "
                    + "\n- ".join(sorted(declined_files))
                )
            if prior_hypotheses:
                prompt += (
                    "\n\nPrior source proposals. Rows with history_kind=confirmed_defect "
                    "are already established defect mechanisms and may come from an "
                    "older FEDOT version; other rows are recent hypotheses:\n"
                    + json.dumps(prior_hypotheses, ensure_ascii=False)
                    + "\nPrefer a different causal mechanism. A changed line number or "
                    "equivalent sanitizer does not make a confirmed defect new; moving "
                    "it to the adjacent line does not make it new either. The "
                    "same file and method remain available for a substantively different "
                    "bug because FEDOT source and behavior can change between versions."
                )
            if excluded_exact:
                prompt += (
                    "\n\nExact locations selected in recent completed campaigns "
                    "(do not return any of these as change_line; adjacent executed "
                    "lines remain eligible only for a genuinely different change):\n- "
                    + "\n- ".join(
                        f"{file_path}:{line}"
                        for file_path, line in sorted(excluded_exact)
                    )
                )
            if step == per_file_steps:
                prompt += (
                    "\n\nFinal decision turn for this catalog file. "
                    "Return status=pick (the current file_path may be omitted) "
                    "or status=skip. Do not call read/search/symbol/callers/docs/run."
                )
            try:
                parsed, raw = _create_pick(
                    inference,
                    prompt,
                    metadata={
                        "catalog_file": lead.file_path,
                        "catalog_line": lead.line,
                        "catalog_index": walked,
                        "catalog_size": n_files,
                        "tool_step": step,
                    },
                )
                actions += 1
            except Exception as exc:
                actions += 1
                if isinstance(exc, EvolveBudgetExhausted):
                    external_stop = "llm_budget_exhausted"
                    budget_exhausted = True
                    rounds.append(
                        {
                            "file_path": lead.file_path,
                            "status": "budget_exhausted",
                            "error_type": type(exc).__name__,
                        }
                    )
                    logger.info(
                        "evolve pick stopped by LLM budget at %s/%s with %s sites",
                        walked,
                        n_files,
                        len(found),
                    )
                    picked = None
                    break
                if "experiment_budget_exhausted" in str(exc):
                    external_stop = "experiment_budget_exhausted"
                    budget_exhausted = True
                    rounds.append(
                        {
                            "file_path": lead.file_path,
                            "status": "budget_exhausted",
                            "error_type": type(exc).__name__,
                        }
                    )
                    logger.info(
                        "evolve pick stopped by experiment budget at %s/%s",
                        walked,
                        n_files,
                    )
                    picked = None
                    break
                failure = classify_model_failure(exc)
                if failure.infrastructure:
                    raise failure from exc
                rounds.append(
                    {
                        "file_path": lead.file_path,
                        "status": "error",
                        "error_type": failure.category,
                    }
                )
                logger.info(
                    "evolve pick %s/%s error %s", walked, n_files, lead.file_path
                )
                picked = None
                break
            if raw:
                raws.append(raw)
            status = (parsed.status or "pick").strip().lower()
            if step == per_file_steps and status not in {"pick", "skip"}:
                rounds.append(
                    {
                        "file_path": lead.file_path,
                        "status": "invalid_final_tool_action",
                        "action": status,
                    }
                )
                logger.info(
                    "evolve pick %s/%s invalid final action %s %s",
                    walked,
                    n_files,
                    status,
                    lead.file_path,
                )
                break
            if status == "run" and (parsed.run_code or "").strip():
                if runs >= run_limit:
                    rounds.append({"file_path": lead.file_path, "status": "run_limit"})
                    logger.info(
                        "evolve pick %s/%s run limit %s",
                        walked,
                        n_files,
                        lead.file_path,
                    )
                    break
                # Scout probes are independent diagnostics. Replaying a previous
                # caught-error probe polluted later stdout and made a successful
                # experiment look as if the real crash persisted.
                run_result = run_fedot_snippet(checkout, parsed.run_code)
                runs += 1
                from fedotllm.agents.evolve.discovery.research_tools import (
                    format_snippet_feedback,
                )

                output = format_snippet_feedback(
                    checkout,
                    run_result,
                    max_chars=_SCOUT_OUTPUT_CHARS,
                )
                shown_output = output
                tool_history.append(
                    f"Step {step} action=run Output:\n```\n{shown_output}\n```"
                )
                if runs >= run_limit:
                    tool_history.append(
                        "Run budget exhausted for this file. "
                        "The next action must be pick, skip, or read; do not run more code.\n"
                    )
                rounds.append({"file_path": lead.file_path, "status": "run"})
                logger.info("evolve pick %s/%s run %s", walked, n_files, lead.file_path)
                continue
            if status == "read":
                read_line = int(parsed.line or 1)
                rel, opened, corrected = _open_requested_runtime(
                    checkout,
                    parsed.file_path,
                    line=read_line,
                )
                opened = opened[:_SCOUT_SOURCE_CHARS]
                if not opened:
                    tool_history.append(
                        f"Step {step} read: cannot open {parsed.file_path}"
                    )
                    rounds.append(
                        {"file_path": parsed.file_path, "status": "read_miss"}
                    )
                    continue
                shown.add(rel)
                correction = (
                    f" (resolved from requested {parsed.file_path})"
                    if corrected
                    else ""
                )
                tool_history.append(
                    f"Step {step} action=read Opened {rel}{correction}:\n{opened}"
                )
                rounds.append(
                    {
                        "file_path": rel,
                        "requested_file_path": parsed.file_path,
                        "status": "read_resolved" if corrected else "read",
                        "line": read_line,
                    }
                )
                continue
            if status in {"search", "symbol", "callers", "docs"}:
                from fedotllm.agents.evolve.discovery.research_tools import (
                    callers_runtime,
                    docs_runtime,
                    search_runtime,
                    symbol_runtime,
                )

                if status == "search":
                    output = search_runtime(checkout, parsed.query)
                    tool_query = parsed.query
                elif status == "symbol":
                    output = symbol_runtime(checkout, parsed.query or parsed.symbol)
                    tool_query = parsed.query or parsed.symbol
                elif status == "docs":
                    output = docs_runtime(checkout, parsed.query)
                    tool_query = parsed.query
                else:
                    tool_query = parsed.symbol or parsed.query
                    output = callers_runtime(checkout, tool_query)
                shown_output = output[:_SCOUT_OUTPUT_CHARS]
                tool_history.append(
                    f"Step {step} action={status} query={tool_query!r}:\n{shown_output}"
                )
                # A path returned by a first-class repository tool is source the
                # model has actually seen, so it may pick that site directly.
                for match in re.finditer(
                    r"(?:^|\n)(?:# )?(fedot/[^:\n]+\.py)(?::\d+)?", output
                ):
                    shown.add(match.group(1))
                rounds.append(
                    {
                        "file_path": lead.file_path,
                        "status": status,
                        "query": tool_query,
                    }
                )
                continue
            complete_claim = all(
                (
                    parsed.mechanism.strip(),
                    parsed.proposed_change.strip(),
                    parsed.expected_metric_effect.strip(),
                )
            )
            invalid_pick = status == "pick" and (
                not complete_claim or parsed.change_line <= 0
            )
            if invalid_pick:
                rejection = (
                    "incomplete_pick_claim"
                    if not complete_claim
                    else "missing_change_line"
                )
                rounds.append(
                    {
                        "file_path": parsed.file_path or lead.file_path,
                        "status": rejection,
                        "catalog_file": lead.file_path,
                    }
                )
                if not complete_claim:
                    feedback = (
                        "pick rejected: fill mechanism, proposed_change, and "
                        "expected_metric_effect with one concrete causal claim"
                    )
                else:
                    feedback = (
                        "pick rejected: set change_line to the exact first source "
                        "line proposed_change would alter"
                    )
                tool_history.append(f"Step {step} {feedback}.")
                logger.info(
                    "evolve pick %s/%s reject %s %s",
                    walked,
                    n_files,
                    rejection,
                    parsed.file_path or lead.file_path,
                )
                if step < per_file_steps:
                    continue
                break
            requested_target = parsed.file_path or lead.file_path
            if (
                status == "pick"
                and requested_target not in shown
                and step < per_file_steps
                and actions < action_limit
            ):
                rel, opened, _ = _open_requested_runtime(
                    checkout,
                    requested_target,
                    line=int(parsed.change_line or parsed.line or 1),
                )
                if opened:
                    shown.add(rel)
                    tool_history.append(
                        f"Step {step}: proposed target was not read, so the pick "
                        "has NOT been accepted. Inspect this source and return "
                        "pick again only if the mechanism still holds; otherwise "
                        "revise it or skip. Pending unverified proposal:\n"
                        + json.dumps(
                            {
                                "file_path": rel,
                                "change_line": parsed.change_line,
                                "mechanism": parsed.mechanism[:800],
                                "proposed_change": parsed.proposed_change[:1_000],
                                "expected_metric_effect": parsed.expected_metric_effect[
                                    :400
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + f"\nOpened {rel}:\n"
                        + opened[:_SCOUT_SOURCE_CHARS]
                    )
                    rounds.append(
                        {
                            "file_path": rel,
                            "status": "pick_target_read",
                            "catalog_file": lead.file_path,
                        }
                    )
                    continue
            picked = _accept_pick(
                parsed,
                shown,
                checkout,
                default_file=lead.file_path,
                default_line=lead.line,
            )
            if picked is not None:
                runtime_lead = evidence_by_file.get(picked.file_path)
                if runtime_lead is not None:
                    evidence = tuple(
                        dict.fromkeys(
                            (
                                f"catalog semantic site: {semantic_site_id(runtime_lead)}",
                                *runtime_lead.evidence,
                            )
                        )
                    )
                    if (
                        picked.file_path.endswith("/default_operation_params.json")
                        and parsed.operation_id.strip()
                    ):
                        evidence = tuple(
                            dict.fromkeys(
                                (
                                    f"executed operation: {parsed.operation_id.strip()}",
                                    *evidence,
                                )
                            )
                        )
                    picked = PatchSite(
                        channel=picked.channel,
                        file_path=picked.file_path,
                        line=picked.line,
                        why=picked.why,
                        evidence=evidence,
                        signals=runtime_lead.signals,
                        mechanism=picked.mechanism,
                        proposed_change=picked.proposed_change,
                        expected_metric_effect=picked.expected_metric_effect,
                        hypothesis_kind=picked.hypothesis_kind,
                    )
            if (
                picked is not None
                and picked.file_path.endswith("/default_operation_params.json")
                and not _default_pick_matches_executed_operation(
                    parsed, picked, checkout
                )
            ):
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "unexecuted_operation_default",
                        "line": picked.line,
                        "operation_id": parsed.operation_id,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: operation_id must name an operation from the "
                    "runtime evidence and the proposed JSON block must belong to it"
                )
                logger.info(
                    "evolve pick %s/%s reject unexecuted default %s:%s (%s)",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                    parsed.operation_id,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if (
                picked is not None
                and picked.file_path.endswith("/default_operation_params.json")
                and not _default_pick_changes_effective_value(parsed, picked)
            ):
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "inert_operation_default",
                        "line": picked.line,
                        "operation_id": parsed.operation_id,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: every proposed parameter/value already equals "
                    "the current effective or estimator default; choose a value "
                    "that changes runtime behavior or skip"
                )
                logger.info(
                    "evolve pick %s/%s reject inert default %s:%s (%s)",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                    parsed.operation_id,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if picked is not None and _matches_recent_concrete_proposal(
                parsed, picked.file_path, prior_hypotheses or []
            ):
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "recent_concrete_proposal_duplicate",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: the same concrete parameter/value assignment "
                    "already appears in recent source-proposal memory; choose a "
                    "different causal change or skip"
                )
                logger.info(
                    "evolve pick %s/%s reject repeated concrete proposal %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if picked is not None and _matches_confirmed_defect_proposal(
                parsed, picked.file_path, prior_hypotheses or []
            ):
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "confirmed_defect_duplicate",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: its defect mechanism matches a previously "
                    "confirmed finding; inspect the current version for a different "
                    "mechanism or skip"
                )
                logger.info(
                    "evolve pick %s/%s reject known defect mechanism %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if picked is not None and (picked.file_path, picked.line) in excluded_exact:
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "recent_site_cooldown",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: that exact change_line is cooling down after "
                    "a recent completed campaign; choose a different executed line "
                    "with a distinct causal change or skip this file"
                )
                logger.info(
                    "evolve pick %s/%s reject cooldown %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if picked is not None and semantic_site_id(picked) in excluded_semantic:
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "recent_semantic_site_cooldown",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                tool_history.append(
                    "pick rejected: that logical source site is cooling down after "
                    "a recent completed campaign; choose a distinct operation or "
                    "runtime mechanism"
                )
                logger.info(
                    "evolve pick %s/%s reject semantic cooldown %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
                if step < per_file_steps:
                    continue
                break
            if (
                picked is not None
                and _metric_line_is_executed(picked.line, picked.evidence) is False
            ):
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "unexecuted_pick",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                logger.info(
                    "evolve pick %s/%s reject unexecuted %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
            declined_repick = picked is not None and picked.file_path in declined_files
            duplicate_pick = picked is not None and picked.file_path in {
                item.file_path for item in found
            }
            if declined_repick:
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "declined_repick",
                        "line": picked.line,
                        "catalog_file": lead.file_path,
                    }
                )
                logger.info(
                    "evolve pick %s/%s declined repick %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
            elif duplicate_pick:
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "duplicate_pick",
                        "line": picked.line,
                    }
                )
                logger.info(
                    "evolve pick %s/%s duplicate %s:%s",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                )
                picked = None
            elif picked is None:
                rounds.append({"file_path": lead.file_path, "status": "skip"})
                logger.info(
                    "evolve pick %s/%s skip %s", walked, n_files, lead.file_path
                )
            else:
                rounds.append(
                    {
                        "file_path": picked.file_path,
                        "status": "pick",
                        "line": picked.line,
                    }
                )
                logger.info(
                    "evolve pick %s/%s pick %s:%s (%s/%s sites)",
                    walked,
                    n_files,
                    picked.file_path,
                    picked.line,
                    len(found) + 1,
                    want,
                )
            if status == "skip":
                declined_files.add(lead.file_path)
            break
        if picked is not None and picked.file_path not in {
            item.file_path for item in found
        }:
            found.append(picked)
            seen.add(picked.file_path)
            seen.add(lead.file_path)
            if on_pick is not None:
                on_pick(list(found))
        if external_stop:
            break
    logger.info(
        "evolve pick done %s/%s files, %s/%s sites", walked, n_files, len(found), want
    )
    if trace is not None:
        trace["llm_pick_raw"] = _clip_raw("\n---\n".join(raws)) if raws else None
        trace["llm_pick_rounds"] = rounds
        trace["scout_actions"] = actions
        trace["scout_action_limit"] = action_limit
        trace["scout_budget_exhausted"] = budget_exhausted
        trace["scout_external_stop"] = external_stop or None
        trace["scout_declined_files"] = sorted(declined_files)
        trace["catalog_files_total"] = n_files
        trace["catalog_files_reviewed"] = len(reviewed_catalog_files)
        trace["catalog_review_fraction"] = (
            len(reviewed_catalog_files) / n_files if n_files else 0.0
        )
        trace["catalog_reviewed_paths"] = reviewed_catalog_files
        trace["catalog_plan"] = catalog_plan
    return found
