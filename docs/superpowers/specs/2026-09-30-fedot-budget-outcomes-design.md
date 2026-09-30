# FEDOT Budget Completion and Quality Outcomes

## Goal

Make a one-hour FEDOT evaluation produce a useful, attributable outcome instead of letting the controller kill a worker and label the candidate `infrastructure_error`. FEDOT must manage its own search budget, preserve a fitted incumbent or fallback, and return a model that the worker can score. The controller records the measured quality or a specific non-quality outcome.

The one-hour budget applies independently to each stock or patched run for one registered dataset, using the same dataset, split, seed, FEDOT settings, and resource profile. Existing completed runs remain immutable historical records.

## Evidence and current behavior

- The worker passes `timeout=60` to `Fedot`, then calls `fit`, prediction, and metric calculation.
- FEDOT's current timer allocates time across composition and tuning, but initial-assumption fit and final full-data fit are not bounded by one end-to-end deadline. Per-graph fit time defaults to `None`.
- The controller separately kills the worker at `timeout_seconds + 300` (3900 seconds for a 3600-second registry budget). A killed worker cannot return its incumbent or a metric.
- The completed retry queue still contains five `infrastructure_error` outcomes. The saved local retry journal is empty/stale, so this specification does not assign those five a cause without a fresh candidate/task/log match.
- The local controller code has newer failure classification than the old server runs. Updating code must not rewrite statuses already stored in old queues.

## Desired FEDOT behavior

Add an explicit return-best timeout policy for the quality-evaluation path, keeping legacy timeout behavior available to other callers until compatibility is verified. In this mode, `Fedot.fit` uses one monotonic deadline shared by initial fit, composition, tuning, and final fit.

1. Fit a simple initial incumbent early. Keep it available if later search candidates fail or consume the remaining budget.
2. Apply a finite limit to each blocking candidate fit based on remaining time; a slow candidate is skipped so the search can continue or stop cleanly.
3. Stop composition and tuning early enough to reserve time for final fitting and worker prediction/metric calculation.
4. Promote only successfully fitted candidates. At the search cutoff, return the best usable fitted candidate. If refitting the winner on the full training data cannot finish in the reserve, retain the already fitted incumbent rather than blocking past the deadline.
5. If neither a composed candidate nor the initial incumbent can produce a valid prediction, return a structured invalid/budget-exhausted result with diagnostics. Do not report an invented metric or claim a quality regression.

The quality worker passes the policy explicitly. The deadline covers the FEDOT fit and leaves a documented reserve for prediction and metric extraction. Dataset loading remains separately timed and recorded, since it occurs before `Fedot.fit` today.

## Controller outcome model

Separate the proposal's quality outcome from the worker attempt's operational state:

- `keep` / `drop`: a valid stock/patch pair produced metrics for the registered tasks, and the frozen decision rule was applied.
- `timeout`: the internal FEDOT budget or emergency outer watchdog expired without a valid metric. Record which deadline fired and the last FEDOT phase.
- `invalid`: FEDOT or the worker completed but could not provide a valid model/metric. Record stock versus patched side and the structured reason.
- `infrastructure_error`: reserve this operational label for evidence-backed external failures such as host loss, filesystem failure, or worker transport failure. Do not use it as a synonym for `search_ran=false`, candidate exceptions, invalid output, or normal budget exhaustion.

Keep retries finite: a timeout or confirmed transient infrastructure failure receives at most one same-parameter retry. A candidate-side invalid result is not retried automatically. When a retry also fails, preserve both attempts and pause before starting another proposal; do not loop.

The outer subprocess timeout remains a last-resort safety guard with a margin over the internal deadline. If it fires, it records `timeout` plus `watchdog_fired=true`; it is not interpreted as evidence that the host infrastructure failed. True host/transport errors retain a distinct operational reason. Existing queue records are not reclassified in place.

## Event and report requirements

For each dataset-side attempt, persist candidate ID, task ID, stock/patch side, attempt number, seed, FEDOT commit, resource settings, internal budget, watchdog budget, final FEDOT phase, incumbent/fallback identity, score validity, metric, and structured error detail. Write each side's result before moving to the next task. Human-readable reports must distinguish:

- measured quality (`keep`/`drop`),
- exhausted budget with a usable model and score,
- timeout without a score,
- invalid model/metric,
- confirmed external infrastructure failure.

Do not collapse these into one `infrastructure_error` count. Historical reports may retain old labels but must identify their controller version and avoid presenting them as new failures.

## Compatibility and rollout

- Do not alter or clear the old 49-proposal queue, the 14-proposal retry queue, or the existing eight-series run.
- Add the policy in the FEDOT source repository, not only in the ignored `.repo_cache/FEDOT` checkout. Pin the tested FEDOT commit in new quality-run records.
- Update the controller worker and result schema after the FEDOT API is available. If the active server has only the old FEDOT commit, do not claim the new behavior is deployed.
- Validate the current five retry errors by candidate ID, task ID, paired stock/patched status, and worker log before deciding whether the allowed retry was transient or candidate-side. Do not rerun them solely because their aggregate status says `infrastructure_error`.
- Start a new timestamped run only after local tests, FEDOT tests, loader checks, and server checkout/version checks pass. Use an isolated workspace and preserve the old runs.

## Acceptance criteria

1. A deterministic slow-candidate test shows that the return-best policy stops search within the configured budget, retains a previously fitted incumbent, returns from `Fedot.fit`, and yields a finite holdout metric within the budget plus the documented scoring reserve.
2. If a candidate fit exceeds its per-candidate allowance, that candidate is skipped and the incumbent remains usable; it does not terminate the entire evaluation as infrastructure failure.
3. A test with no usable model returns a structured invalid/timeout result with no score, never `keep` or `drop`.
4. Worker tests show that internal budget exhaustion with a valid metric is evaluated normally and does not set `infrastructure_error` or trigger a retry.
5. Tests distinguish internal timeout, watchdog timeout, candidate-side invalid/crash, and confirmed external infrastructure errors; only a timeout without a valid metric and confirmed transient infrastructure failures get one retry.
6. Old queue files and journals remain unchanged; new records contain stable candidate/task/attempt/log linkage.
7. A new run on the server reports its FEDOT commit and policy in the journal, and a time-series benchmark produces valid scores without reaching the watchdog under the configured internal budget.
