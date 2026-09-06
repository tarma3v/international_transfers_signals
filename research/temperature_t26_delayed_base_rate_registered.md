# T26 preregistration: delayed mature-only base-rate correction

Registered 2026-09-06 before computing T26 outputs. T24/T25 established that a
history-only h20 rank transfers to 2025--2026, while its absolute event rate
drifts. T26 freezes the T25 `residual_a040` rank and tests whether a slow,
causal intercept update can repair probability level. The opened 2025--2026
period remains diagnostic and cannot choose a window, penalty or model.

## Frozen state and unique feedback

- Rebuild the T25 00:15 history-only query and assert that its pre-2025 selector
  still returns `residual_a040`.
- A feedback observation is included once, on its actual CBR publication date;
  calendar holds/weekends never duplicate the same outcome.
- At query date `t`, a feedback row is eligible only when its complete h20
  maturity date is strictly before `t - 2 calendar days`.
- Every query on the same date receives the same global intercept state; a
  hierarchical candidate may add a shrunken currency residual.
- The update never sees current or future query targets, announced tomorrow
  CBR, intraday market data or a manually labelled SVO/post-2022 regime.

## Frozen candidates

For eligible rows, solve an intercept-only penalized logistic recalibration
around the frozen T25 logit. The global ridge penalty is 20 and the correction
is clipped to `[-1.5, 1.5]`. Until at least 30 unique mature publication dates
exist, return T25 unchanged.

1. `delayed_global_w30`: last 30 eligible publication dates.
2. `delayed_global_w60`: last 60 dates.
3. `delayed_global_w125`: last 125 dates.
4. `delayed_global_w250`: last 250 dates.
5. `delayed_global_expanding`: every eligible date.
6. `delayed_hier_w125`: global 125-date correction plus currency residuals
   fitted on the same rows with ridge 40 and clipped to `[-0.75, 0.75]`.

No half-life, window, penalty, clipping threshold or minimum-history rule may
be altered after results are opened.

## Selection and evaluation

Select on mature 2024-H2 calendar queries only. A candidate is feasible against
frozen T25 when Brier and log-loss are lower, ECE is no more than 0.005 worse,
and AUC is no more than 0.005 lower. Select minimum Brier with fixed priority in
the order above. If none passes, retain T25.

Evaluate once on 2025--2026. A strict diagnostic pass requires:

- aggregate Brier and log-loss below T25, ECE no more than 0.005 worse and AUC
  no more than 0.005 lower;
- 20- and 50-date paired moving-block 95% Brier-delta upper bounds below zero
  against T25;
- Brier no worse than T25 in either 2025 or 2026;
- no currency Brier deterioration larger than 0.005;
- all probabilities finite/bounded and every update trace causal.

Even a pass is only a prospective shadow challenger because 2025--2026 is an
opened retrospective. It does not change AP37, push thresholds or receipt-time
routing.

## Required audits

- source hashes, row/date counts and T25 model selection rebuilt;
- publication/source timestamps no later than query;
- every feedback maturity strictly before the query-specific embargo cutoff;
- no repeated calendar hold contributes feedback;
- selected candidate unchanged after evaluation target corruption; for any
  audit origin, corrupting targets whose maturity is not strictly before that
  origin's embargo cutoff leaves the prediction prefix through the origin
  unchanged (older evaluation outcomes may legitimately update later dates);
- exact selected-column reconstruction, bounded probabilities and complete
  paired 20/50-date intervals.
