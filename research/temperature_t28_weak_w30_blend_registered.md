# T28 preregistration: weak delayed-w30 blend with Q3/Q4 separation

Registered 2026-09-06 before computing T28 blend outputs. T26/T27 found one
repeatable pre-2025 direction: full-strength w30 improves Brier, log-loss and
AUC, but worsens ECE beyond the frozen gate. T28 tests small fixed doses of the
same causal correction. No 2025--2026 metric may choose a dose.

## Frozen inputs

- Rebuild the T27 combined stream: 247 unique OOS publication dates from 2023
  plus the unchanged T25/T26 calendar query from 2024 onward.
- Recompute the exact T26 `delayed_global_w30` using only unique h20 outcomes
  whose maturity is strictly before query date minus the two-day embargo.
- Keep T25 `residual_a040` as probability anchor. No new feature, window,
  penalty, SVO/year flag, currency switch or intraday input is introduced.

## Frozen blend family

For beta in `0.10, 0.20, 0.30, 0.40, 0.50`, define:

`blend_logit = T25_logit + beta * (w30_logit - T25_logit)`.

The full w30 candidate is a reported control, not selectable. Candidate order
for ties is beta small-to-large.

## Nested pre-2025 decision

1. Weight screen: query dates in 2024-Q3, but only outcomes that matured before
   2024-10-01 minus embargo. A beta is feasible if Brier and log-loss are below
   T25, ECE delta <= +0.005 and AUC delta >= -0.005. Choose minimum Brier.
2. Frozen validation: query dates in 2024-Q4 with outcomes matured before
   2025-01-01 minus embargo. The selected beta must independently meet the same
   four gates. Otherwise the final model is T25.
3. Open diagnostic: apply the Q3-selected/Q4-validated rule once to
   2025--2026. This period cannot change beta or validation outcome.

The 2024 split is not a fresh untouched holdout because prior packets disclosed
aggregate 2024-H2 behaviour. It is a stricter retrospective nesting check, not
new independent evidence.

## Open evaluation gate

Any retrospective pass additionally requires, against T25 on 2025--2026:

- aggregate Brier/log-loss lower, ECE delta <= +0.005 and AUC delta >= -0.005;
- 20/50-date paired Brier 95% CI upper bounds below zero;
- Brier no worse in either year and no currency worse by more than 0.005;
- causal trace, bounded probabilities and exact selected-column rebuild.

Even a pass is a shadow challenger only. AP37, push cadence and after-receipt
T22 remain unchanged; prospective outcomes are still required for promotion.

## Required audits

- T27 historical quarterly fits and unique feedback rebuilt;
- Q3/Q4 masks disjoint, mature before their decision cutoffs and both pre-2025;
- selected beta rebuilt from Q3 only and validation rebuilt from Q4 only;
- corrupting every 2025--2026 target leaves Q3 choice and Q4 decision unchanged;
- future-prefix, source/publication timestamps, probability bounds, source
  hashes and complete paired interval grid.
