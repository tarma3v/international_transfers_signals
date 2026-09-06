# T45 registration: quality-gated direct-pair residual benefit

Registered 2026-09-07 before any T45 fit, prediction, gate decision, or target
metric was computed. All 2025-2026 outcomes and the older round-7 direct-pair
experiments are already open. T45 therefore uses a new disjoint 2023 screen and
2024 validation and can become only a prospective shadow, never a
production-proven model from this replay.

## Question

Does a corridor's own MOEX/RUB pair add stable information to the existing
15:30 CNY-anchored expected future-only CBR benefit when, and only when, that
pair is observably fresh enough?

## Frozen baseline and target

- Baseline: T6 `cutoff_1530` expected future-only CBR benefit for each
  `h=1/3/5/10/20`.
- Target: the mean of the next `h` effective CBR rates versus the currently
  effective CBR rate, in sender-positive basis points, exactly
  `benefit_forward_only`.
- This experiment does not change probability, temperature, sparse push,
  phase routing, receipt logic, or any bank-savings claim.

## Frozen data and observable quality gate

Use the saved round-7 ten-minute CETS TOM/TOD panel at the fixed 15:30 Moscow
cutoff. TOM is preferred and TOD is used only when TOM has no completed candle.
The direct correction is eligible only when the selected pair has at least six
completed candles since 10:00 and its last completed candle is no more than 60
minutes old. Otherwise T45 must equal T6 exactly.

Within eligible rows define the already frozen round-7 soft quality

`q = min(completed_candles / 24, 1) * exp(-age_minutes / 120)`.

This is a candle-density/freshness proxy, not turnover or executable
liquidity. Null volume is never interpreted as zero.

## Frozen model

For each horizon fit one global quarterly residual model. The response is
`actual_future_bps - T6_expected_future_bps`. Training rows must:

- start on or after 2022-02-24;
- precede the quarter origin;
- have the full horizon mature strictly before origin minus a two-day embargo;
- have finite target and T6 baseline;
- pass the same observable hard-quality gate as query rows.

Features are fixed to T6 baseline, direct mean/last/previous CBR basis,
intraday return, session range, completed-candle count, age, soft quality,
TOM/TOD counts and the five corridor one-hot indicators. Basis features are
clipped to +/-5000 bps and age to 720 minutes before fitting.

The estimator is `StandardScaler + Ridge(alpha=100)`. Its target is clipped at
the training-only 2.5th/97.5th percentiles. Training weights equal a fixed
730-day half-life times `q`. There is no parameter grid and no local refit.
For an eligible query:

`candidate = baseline + q * predicted_residual`.

For an ineligible query or a fit with fewer than 200 eligible rows:

`candidate = baseline`.

## Frozen evaluation and gates

Evaluate screen-2023 first, then validation-2024 only. For each horizon report
MAE, RMSE, bias and Spearman against T6 on all rows, the hard-direct subset and
each currency. Bootstrap the all-row daily mean absolute-error delta with
circular moving blocks of 20 and 50 dates, 2000 draws, fixed seed.

A horizon passes a stage only if all conditions hold:

1. all-row MAE delta is strictly below zero;
2. all-row RMSE delta is at most zero;
3. all-row Spearman delta is at least -0.01;
4. hard-direct MAE delta is strictly below zero;
5. KZT and AMD hard-direct MAE deltas are each at most +2 bps and at least one
   is strictly below zero;
6. the upper 95% bootstrap bound for all-row MAE delta is at most zero for both
   20- and 50-date blocks.

A horizon's 2025-2026 candidate metrics may be computed and saved only if that
horizon passes both 2023 and 2024. Other horizons keep their open predictions
masked and remain exact T6 fallbacks. `production_promoted=false` regardless
of the result because the design follows already-open related research.

## Required audit

- source hashes and exact quarterly reconstruction;
- exact T6 equality on every ineligible row;
- hard gate, q formula, training-only clipping, recency weights, maturity and
  embargo reconstructed;
- complete h/currency/year metric grid for the historical stages;
- paired bootstrap and stage decisions independently rebuilt;
- corrupting direct features, targets and maturity at/after a future boundary
  cannot change any earlier raw candidate prediction;
- sparse push, probabilities and runtime routing remain unchanged.

