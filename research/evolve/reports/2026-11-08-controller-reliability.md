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

## DEV != FINAL, budget, recovery

tests/unit/agents/test_evolve_mvp.py::test_dev_final_indices_are_disjoint_and_repeatable passed. SHADOW, DEV and FINAL index sets are disjoint.

Worker tests reject a search budget that consumes the whole side deadline and keep fedot_budget_timeout as timeout, not invalid. test_evolve_budget_retry_review_fixes.py passed.

test_evolve_quality_reliability.py passed. An unfinished started side is still recovered as status=crash. That journal status is not the worker exception payload.

## Smoke A/A

purpose: smoke_aa_measurement_noise
quality_claim: False
values: [16, 16]
abs_delta: 0
identical: True

## Not claimed

Full OpenML registry KEEP was not run. Identical smoke repeats are
measurement noise only.

## Regression command

```
uv run pytest -q tests/unit/agents/test_evolve_runtime_contract.py tests/unit/agents/test_evolve_return_best_policy.py tests/unit/agents/test_evolve_worker_invocation_protocol.py tests/unit/agents/test_evolve_quality_worker_diagnostics.py tests/unit/agents/test_evolve_controller_smoke.py tests/unit/agents/test_evolve_doctor_preflight.py tests/unit/agents/test_evolve_quality_cache_policy.py tests/unit/agents/test_evolve_mvp.py::test_dev_final_indices_are_disjoint_and_repeatable tests/unit/agents/test_evolve_quality_reliability.py tests/unit/agents/test_evolve_budget_retry_review_fixes.py
```

Result: 197 passed in 11.12s.
