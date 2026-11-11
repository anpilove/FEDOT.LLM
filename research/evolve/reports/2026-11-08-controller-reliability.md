# EvolveAgent controller reliability

Scope: controller and worker reliability on `research/evolve-agent`.
No model change, no FEDOT automerge, no quality KEEP.

## Doctor

Doctor preflight fails a stock pin that does not name return_best parameters, requires dask only when the checkout imports it, and rejects a fedot path outside the checkout. Evaluator is skipped when preflight fails.

## Worker invalid

Caught worker failures other than a FEDOT budget timeout publish status=invalid, outcome=invalid, and exit 2. Budget timeout stays status=timeout.

## Checkout contract

Smoke A/A and the microbenchmark workspace path run in an experiment checkout. Runtime temp and GOLEM directories stay under the workspace and outside the frozen source. The checkout is discarded after the probe.

## return_best policy

return_best requires explicit Fedot.__init__ parameters. Installed pin compatible=False missing=['timeout_policy', 'final_fit_reserve_seconds', 'cv_folds', 'max_pipeline_fit_time', 'max_pipeline_evaluation_time']. The worker publishes return_best_incompatible and does not call Fedot(). legacy keeps the current kwargs and does not claim return_best.

## Candidate invalid

`fedot_import_error`, `fedot_fit_error`, `scoring_error`, and `worker_error` are a broken patch in compare, affected-metric, metric_study, transfer, feedback, and confirmation. `return_best_incompatible` and timeout stay infrastructure. A traced invalid baseline stays in a transfer plan. An untraced invalid baseline stays incomplete.

## DEV and FINAL

`frozen_split_outcomes` on 80 alternating labels, problem classification: disjoint=True, keep=None, quality_claim=False. DEV outcome=reserved count=16. FINAL outcome=reserved count=16. No model was fit.

## Interrupted recovery

An unfinished started side stays status=crash. The journal result now also has outcome=interrupted and failure_kind=controller_interrupted.

## Smoke A/A

purpose: smoke_aa_measurement_noise
quality_claim: False
values: [16, 16]
abs_delta: 0
identical: True

The control arm of the fixture smoke records control_score_pairs=3 and control_score_abs_delta=0.

## Missing requested dataset

GOLEM `GraphVerifier.verify` returns false when a rule returns `False` or raises `ValueError`. A rule that returns `None` does not reject the graph. Any other exception leaves `verify`. `compare_pack` used to raise `KeyError` when a requested lift or protect id was absent from the stock or patch map, which stopped the campaign before the next hypothesis. That gap is now `keep=False`, `infrastructure_error=True`, reason `missing_measurement <task_id>`. A measured gain on another dataset in the same pack does not KEEP.

## Doctor import diagnosis

A doctor run on pinned commit `8990f28b0664c99f7eb2fddcee3931151fe16e79` failed while importing the checkout. The snippet worker imported `fedot` before the preflight script, so the probe died as `process exited 1` and the dask check was reported before dask had been measured. The worker now emits `EVOLVE_PREFLIGHT` with `dask` and `import_error` when that import fails. Doctor keeps the record and puts the exception on the path check. The same run now reports `ModuleNotFoundError: No module named 'dask'` and `dask.importable=false`.

## Configuration search without a measurement

`benchmark --component configuration-search` on pinned FEDOT `8990f28` measured `rf`, `cancer`, and `kc2` as `invalid` because the checkout import failed. The search then returned `no_configuration_keep` with `infrastructure_error=false` and no trials. That is now `configuration_stock_unmeasured rf` with `infrastructure_error=true`. A measured stock pack that simply has no tunable surface still ends as `no_configuration_keep`.

## Behavior probe before import

The configuration benchmark's behavior probe on pinned FEDOT `8990f28` exited on both sides with `EVOLVE_PREFLIGHT` and `ModuleNotFoundError: No module named 'dask'`. That pair was labeled `invalid` / `inconclusive_invalid_probe`, so the campaign would spend the next revision rewriting a probe that never ran. The same pair is now `infrastructure` / `environment_import_failed`. The campaign records `behavior_probe_infrastructure` and stops instead of asking for a new probe. A one-sided probe failure stays `invalid`.

## Configuration search proposes without a model key

`dask-ml==2024.4.4` is a project dependency so `uv run` can import pinned FEDOT `8990f28`. That release imports `dask_ml` and still runs with scikit-learn 1.3.2. `dask-ml` 2025.1.0 does not. Doctor on this pin now reports `fedot_import.ok=true` and `dask.importable=true`. `return_best` stays missing, so the evaluator remains skipped.

With no `FEDOTLLM_LLM_API_KEY`, `benchmark --component configuration-search` measured `rf`, `cancer`, and `kc2`, wrote `min_samples_leaf` 1 -> 2 into the experiment copy of `default_operation_params.json`, and saw the probe change to `{"min_samples_leaf": 2, "n_jobs": 1}`. DEV target_delta was 0.0172 (`kc2` +0.0172, `rf` +0.0050, `cancer` 0). FINAL target_delta was 0.0170 (`rf` +0.0170, `kc2` +0.0011, `cancer` -0.0017). The benchmark then discarded the experiment checkout. A DEV keep now also writes `kept_configuration.patch` in the workspace, outside `experiments/`, so the applied edit remains after that discard. The benchmark summary names the change, both target deltas, the per-task deltas, `final_keep`, and that patch path.

The search used to return the first passing default. With a two-trial budget it now keeps the later default only when its DEV target_delta is strictly larger. On pinned FEDOT `8990f28` the second trial, `min_samples_leaf` 4, beat 2: DEV target_delta 0.0232 (`kc2` +0.0232, `rf` +0.0105, `cancer` -0.0017) and FINAL target_delta 0.0249 (`rf` +0.0249, `kc2` +0.0071, `cancer` 0). The surviving patch sets `min_samples_leaf` to 4. That file is a diff of the whole defaults document, with a trailing newline, so `git apply` accepts it. A diff of only the `rf` object did not: `git apply` reported a corrupt patch, and `patch` needed fuzz.

The benchmark now scores three trials and lists each one in `summary.measured`, including the drop reason. The third trial is `min_samples_split` 2 -> 4. Its best lift was 0.0055 on `kc2`, and `rf` fell by 0.0115, so the quick gate returned `regression rf delta -0.0115`. The kept patch stays `min_samples_leaf` 4. This is not a 40-dataset quality KEEP, and `run` / `hunt` still do not invent a source patch without `FEDOTLLM_LLM_API_KEY`.

## Not claimed

Full OpenML registry KEEP was not run. Identical smoke repeats and reserved split counts are measurement bookkeeping only. The three-task configuration keep above is not that registry KEEP.

## Regression command

```
uv run pytest -q tests/unit/agents/test_evolve_runtime_contract.py tests/unit/agents/test_evolve_return_best_policy.py tests/unit/agents/test_evolve_worker_invocation_protocol.py tests/unit/agents/test_evolve_quality_worker_diagnostics.py tests/unit/agents/test_evolve_controller_smoke.py tests/unit/agents/test_evolve_doctor_preflight.py tests/unit/agents/test_evolve_quality_cache_policy.py tests/unit/agents/test_evolve_mvp.py::test_dev_final_indices_are_disjoint_and_repeatable tests/unit/agents/test_evolve_quality_reliability.py tests/unit/agents/test_evolve_budget_retry_review_fixes.py tests/unit/agents/test_evolve_invalid_alignment.py tests/unit/agents/test_evolve_compare_invalid.py tests/unit/agents/test_evolve_transfer.py tests/unit/agents/test_evolve_agent.py::test_compare_pack_keep_and_protect tests/unit/agents/test_evolve_agent.py::test_compare_pack_uses_per_task_rmse_threshold tests/unit/agents/test_evolve_agent.py::test_compare_pack_distinguishes_small_metric_signal_from_practical_keep tests/unit/agents/test_evolve_audit.py::test_nonfinite_protect_score_cannot_pass tests/unit/agents/test_evolve_audit.py::test_scoreboard_preserves_small_relative_keep_and_last_drop
```

Result: 263 passed in 12.91s.
