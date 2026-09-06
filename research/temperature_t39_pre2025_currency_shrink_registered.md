# T39 registration: pre-2025 currency shrink for the h20 history phase

Registered after T38 identified currency-local calibration gaps on the already
opened 2025-2026 retrospective, but before any T39 candidate weight or metric
was computed. T38 local results motivate the question only. No 2025-2026
target, metric, prediction error, or T38 failing group may select a T39 weight.

## Question

Can a currency-specific h20 history-phase shrinkage map learned entirely from
pre-2025 chronological OOS predictions improve T37 local stability without
changing its source-driven causal route?

## Frozen data and split

- source predictions: T34 `development_predictions.csv.gz` only;
- screen: mature 2023 rows, using the same h20 maturity cutoff and embargo as
  T34;
- validation: mature 2024 rows, never used to choose a candidate weight;
- open diagnostic: unchanged T37 2025-2026 snapshots, evaluated exactly once
  after the complete currency map is frozen;
- five currencies are handled independently, with no use of their open-period
  T38 error ranking.

The after-receipt and intraday phases have no matching disjoint pre-2025 replay
in the current packet. T39 therefore changes only rows whose observed
`snapshot_source_kind == cbr_history`. Every market, bridge, perpetual, hold,
and receipt-dependent row must remain bitwise equal to T37.

## Frozen candidate map

For each currency evaluate the single grid
`alpha in {0.00, 0.25, 0.50, 0.75, 1.00}` on 2023 only, where

`p(alpha) = sigmoid((1-alpha) * logit(identity) + alpha * logit(T34))`.

Choose the alpha with minimum 2023 Brier. Exact ties prefer the value closest
to 0.50, then the smaller alpha. Require at least 200 mature rows for both the
screen and validation currency cells; otherwise freeze alpha=0.50.

The selected 2023 alpha is accepted on 2024 only when, relative to the fixed
T37 alpha=0.50 history baseline:

- Brier delta <= +0.001;
- log-loss delta <= +0.003;
- ECE delta <= +0.01;
- AUC delta >= -0.005.

If any gate fails, that currency falls back to alpha=0.50. Validation never
chooses a different non-default alpha. The resulting five-value map is frozen
before the open diagnostic.

## Open diagnostic and success rule

Apply the frozen map only to T37 `cbr_history` rows. Evaluate both receipt
scenarios, all 20 clocks, pooled metrics, source components, and the exact T38
currency/year/currency-year local audit with 20/50-date paired block bootstrap.

T39 is a successful retrospective repair only if all of the following hold:

- at least one currency has a validated alpha different from 0.50;
- all 40 `scenario x clock` rows are non-inferior to T37 using the existing
  +0.001 Brier, +0.003 log-loss, +0.01 ECE, and -0.005 AUC allowances;
- both pooled scenarios are non-inferior to T37 under the same point gates;
- all four pooled year groups continue to pass strict T38 evidence;
- clock-local pass count is greater than 619/680;
- pooled local pass count is greater than 25/34;
- no output uses a source timestamp later than its query;
- all non-history rows are bitwise equal to T37.

The open period may reject this frozen map, but may not alter a weight, gate,
fallback, or routing rule.

## Interpretation

T39 cannot promote a production model even if it passes. The motivating T38
diagnostic and the 2025-2026 evaluation are already open. A pass creates a
stronger frozen prospective challenger; a failure preserves T37 exactly and
becomes evidence that two pre-2025 years do not justify currency-specific
repair. Sparse push, expected future-only basis points, receipt policy, client
copy, and runtime promotion remain unchanged. `production_promoted=false`.

## Required audit

- source hashes for registration, code, audit, T34 development predictions,
  T37 predictions/metadata, and T38 local evidence;
- exact reconstruction of the 2023 grid, selected weights, 2024 gates, frozen
  map, open predictions, metrics, local bootstrap, and final decision;
- explicit proof that no 2025-2026 target enters weight selection;
- bitwise preservation of every non-history T37 prediction;
- corruption check showing future/open targets cannot change the frozen map.
