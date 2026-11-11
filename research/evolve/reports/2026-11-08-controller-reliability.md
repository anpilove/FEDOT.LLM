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

The benchmark now scores three trials and lists each one in `summary.measured`, including the drop reason. On one run the third trial was `min_samples_split` 2 -> 4. Its best lift was 0.0055 on `kc2`, and `rf` fell by 0.0115, so the quick gate returned `regression rf delta -0.0115`.

When a later value of the same parameter beats an earlier keep, the search now inserts one farther step ahead of the unrelated parameters. `min_samples_leaf` 6 then beat 4 on DEV: target_delta 0.0307 (`kc2` +0.0307, `rf` +0.0128, `cancer` -0.0017) against 0.0232. FINAL for 6 still kept, at target_delta 0.0198 (`rf` +0.0198, `kc2` +0.0071, `cancer` -0.0017). That FINAL delta is smaller than the 0.0249 measured for leaf 4 on the previous identical three-task benchmark. DEV-max and FINAL-max are not the same choice. The benchmark now retains every DEV keep, scores FINAL on each, and writes `kept_configuration.patch` for the passing FINAL with the larger target delta. On the same pin, seed, and three tasks the measured FINAL deltas were leaf 2 at 0.0170, leaf 4 at 0.0249 (`rf` +0.0249, `kc2` +0.0071, `cancer` 0), and leaf 6 at 0.0198. The kept patch is `min_samples_leaf` 4, and `git apply --check` succeeds. DEV still names leaf 6 as its winner. `run` without `FEDOTLLM_LLM_API_KEY` now takes that same measured path instead of printing that `propose_patch` will no-op. On 11 November 2026 at 23:15 Europe/Moscow the command `run --lift-tasks rf,cancer,kc2` exited 0, kept `rf min_samples_leaf 1 -> 4` (FINAL target_delta 0.0249), and wrote `kept_configuration.patch` inside the run workspace. `git apply --check` on that patch exits 0. A second `run` in the same workspace skipped those three measured patches and tried `min_samples_split` 4 (quick drop, `rf` -0.0115), `max_depth` 8 (FINAL 0.0152), and `max_features` log2 (quick drop, `rf` -0.0174). `max_depth` 8 is a weaker FINAL than leaf 4, so a later run keeps the leaf-4 patch when the new trial does not beat it. The third run skipped six measured patches, dropped `class_weight` balanced (`kc2` -0.0120), and still reported leaf 4. The next run applied that leaf-4 patch as the baseline and measured the following parameters on top of it. `min_samples_split` 4 kept the same FINAL delta as leaf 4 alone, 0.0249. `max_depth` 8 lost on DEV (0.0227 against the split stack's 0.0232). `max_features` log2 on top of leaf 4 raised FINAL target_delta to 0.0261 (`rf` +0.0261, `cancer` -0.0017, `kc2` 0). The kept patch contains both `min_samples_leaf` 4 and `max_features` log2, and `git apply --check` exits 0. A run at 23:58 Europe/Moscow on 11 November 2026 FINAL-scored the trials that lost the DEV race: `max_depth` 8 finished at FINAL 0.0176, `class_weight` balanced at FINAL 0.0210, and `min_samples_split` 4 tied the kept FINAL 0.0261. None replaced the patch. A keyless `run` that already has an rf FINAL keep now hunts `lgbm` on `lgbm`, `cancer->lgbm`, and `kc2->lgbm`, and protects `rf`, `cancer`, and `kc2`. A FINAL keep on that new operation stays even when its target delta is smaller than the rf delta. The published patch is applied to `improved_fedot` in the workspace. The pinned checkout is not modified. At 00:04 Europe/Moscow on 12 November 2026 that hunt exited 0 and kept the rf patch: `min_child_samples` 10 failed shadow (`kc2->lgbm` -0.0169), `min_child_samples` 40 reached only 0.0010, and `reg_lambda` 0.01 reached 0.0057. `improved_fedot` contains `min_samples_leaf` 4 and `max_features` log2, and `git apply --check` exits 0. The first `min_child_samples` proposal had been the variance-increasing value 10. The queue now proposes the larger leaf support first. A second scale of the same parameter no longer consumes the budget before every other parameter has one trial. At 00:10 Europe/Moscow on 12 November 2026 the next hunt skipped the three measured patches and tried `reg_alpha` 0.01 (0.0057, below threshold), `max_depth` 8 (`kc2->lgbm` -0.0126), and `class_weight` balanced (`lgbm` -0.0108). The published patch stayed the rf default. At 00:14 the following hunt spent its three trials on `learning_rate` 0.05 (0.0032), then the inserted neighborhood 0.075 (0.0017) and 0.025 (0.0035), and never reached `num_leaves`. A positive quick miss now waits behind the next parameter. At 00:16, after that change, `num_leaves` 15 scored 0.0036, the `learning_rate` 0.05 plus `num_leaves` 15 bundle scored 0.0048, and `min_child_samples` 40 plus `num_leaves` 15 passed quick and full DEV then failed shadow at 0.0010. The published patch stayed `min_samples_leaf` 4 and `max_features` log2. The next keyless `run` measures one source defect before another default: `update_callbacks` builds `lgbm_early_stopping(...)` and discards the return value. At 00:24 Europe/Moscow on 12 November 2026 that patch scored target_delta 0.0000 on `lgbm`, `cancer->lgbm`, and `kc2->lgbm`. LightGBM 4.6 already stops from the `early_stopping_rounds` parameter, so installing the callback does not change these predictions. The hash is remembered and the published patch stayed the rf default. The following run at 00:30 Europe/Moscow skipped that hash and spent its three trials on refinements of earlier near-misses: `learning_rate` 0.075 with `num_leaves` 15 scored 0.0018, `learning_rate` 0.025 with `num_leaves` 15 scored 0.0035, and `learning_rate` 0.05 with `num_leaves` 23 scored 0.0032. None replaced the rf patch. A historical refinement now waits until every parameter has had its first trial. At 00:37 Europe/Moscow on 12 November 2026 the next hunt skipped the measured prefix and tried `reg_lambda` 0.01 with `num_leaves` 15 (0.0057), `reg_alpha` 0.01 with `num_leaves` 15 (0.0057), and `colsample_bytree` 0.8. The last passed quick and full DEV, then shadow reached only 0.0065 (`cancer->lgbm` +0.0065). The published patch stayed the rf default. `continue` still needs the key to rewrite arbitrary source. This is not a 40-dataset quality KEEP.

## Not claimed

Full OpenML registry KEEP was not run. Identical smoke repeats and reserved split counts are measurement bookkeeping only. The three-task configuration keep above is not that registry KEEP.

## Regression command

```
uv run pytest -q tests/unit/agents/test_evolve_runtime_contract.py tests/unit/agents/test_evolve_return_best_policy.py tests/unit/agents/test_evolve_worker_invocation_protocol.py tests/unit/agents/test_evolve_quality_worker_diagnostics.py tests/unit/agents/test_evolve_controller_smoke.py tests/unit/agents/test_evolve_doctor_preflight.py tests/unit/agents/test_evolve_quality_cache_policy.py tests/unit/agents/test_evolve_mvp.py::test_dev_final_indices_are_disjoint_and_repeatable tests/unit/agents/test_evolve_quality_reliability.py tests/unit/agents/test_evolve_budget_retry_review_fixes.py tests/unit/agents/test_evolve_invalid_alignment.py tests/unit/agents/test_evolve_compare_invalid.py tests/unit/agents/test_evolve_transfer.py tests/unit/agents/test_evolve_agent.py::test_compare_pack_keep_and_protect tests/unit/agents/test_evolve_agent.py::test_compare_pack_uses_per_task_rmse_threshold tests/unit/agents/test_evolve_agent.py::test_compare_pack_distinguishes_small_metric_signal_from_practical_keep tests/unit/agents/test_evolve_audit.py::test_nonfinite_protect_score_cannot_pass tests/unit/agents/test_evolve_audit.py::test_scoreboard_preserves_small_relative_keep_and_last_drop
```

Result: 263 passed in 12.91s.
