# Documented path to registry A/A (no KEEP claim without data)

This is the production-shaped path to a full EvolveAgent quality A/A on the
40-task registry. It does **not** claim a quality improvement by itself.

## Preconditions (doctor must be green)

1. Cloud Environment prepared by the owner (boot from a successful build; not `build: null`).
2. `FEDOTLLM_REPO_PATH` → git checkout of FEDOT at manifest pin `8990f28b…` **or** a research worktree that passes `return_best` preflight (`EVOLVE_FEDOT_RETURN_BEST_POLICY=research_worktree`, `EVOLVE_FEDOT_RETURN_BEST_ROOT=…`).
3. Checksum assets: `examples/real_cases/data` match `fedotllm/agents/evolve/benchmark_manifest.json`.
4. Runtime deps: Python 3.11, `dask` importable, EvolveAgent `uv sync --group dev`.
5. Offline OpenML: `EVOLVE_QUALITY_DATA_OFFLINE=1` and `EVOLVE_OPENML_CACHE` pointing at a prepared cache covering the 32 OpenML tasks in `quality_registry.json`.
6. Doctor:

```bash
export FEDOTLLM_REPO_PATH=/path/to/FEDOT
python -m fedotllm.agents.evolve doctor --source "$FEDOTLLM_REPO_PATH" --workspace /tmp/evolve-doctor
# expect ok=true only when pin+assets+dask+path all pass
```

## A/A command shape (stock vs stock)

Use the controller quality drain / worker with identical stock checkouts on both sides, short budget for smoke, full registry budgets only on the prepared environment:

```bash
# identity pinned in quality_registry.json
# fedot_timeout_policy=return_best, side 5400s, search 4500s, reserve 1500s
python -m fedotllm.agents.evolve controller …   # campaign quality pair path
```

Keep FINAL metrics out of LLM context; Judge-only.

## Incompleteness statuses (do not invent KEEP)

| Outcome | Meaning |
|---|---|
| `invalid` + `fedot_pin_incompatible` | pin/API/deps refused before fit |
| `timeout` + `fedot_budget_timeout` | budget exhausted, no metric |
| `crash` + other `failure_kind` | unexpected worker failure |
| missing OpenML cache | stop; list missing task ids; no quality claim |

## Evidence already collected offline

- Synthetic A/A: `evidence/aa-smoke.json`
- Local TS temperature A/A on research worktree: `evidence/aa-local-ts.json`
- Doctor refuse on stock pin: `evidence/doctor-stock-pin.json`
