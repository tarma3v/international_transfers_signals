# T38 registration: h20 local stability audit

Registered after T37 was completed and after its already saved pointwise
currency/year metrics were inspected. The observed point counts are therefore
open: year slices pass the existing non-inferiority rule, while some currency
and currency-year slices exceed the local ECE allowance. T38 does not select,
fit, calibrate, blend, or change a model. It freezes a stricter evidence audit
before computing any local paired-bootstrap interval.

## Question

Does T37's pooled 40/40 state result remain supported when the same h20 output
is examined by currency, year, and currency-year, keeping whole dates and all
their clocks together?

## Frozen inputs and outputs

- use the saved T37 predictions without modification;
- preserve both receipt scenarios and all 20 clocks;
- report the existing clock-local rows for `currency`, `year`, and
  `currency_year`;
- additionally pool all clocks inside each
  `scenario x {currency|year|currency_year}` group;
- compute paired circular moving-block bootstrap with block lengths 20 and 50
  dates, 500 deterministic draws, sampling the complete date payload together;
- report Brier, log-loss, ECE, AUC, AP, point deltas, and Brier/log-loss/AUC
  paired intervals.

## Frozen gates

A clock-local slice is non-inferior only if:

- Brier delta <= +0.001;
- log-loss delta <= +0.003;
- ECE delta <= +0.01;
- AUC delta >= -0.005.

A pooled local group passes strict evidence only if:

- Brier, log-loss, and AUC improve at point estimate;
- ECE delta <= +0.01;
- the maximum Brier CI upper bound across 20/50-date blocks is below zero;
- the minimum AUC CI lower bound across 20/50-date blocks is above zero.

`local_stability_passed` requires every clock-local slice and every pooled
local group to pass. No threshold may be relaxed after the result.

## Interpretation

T38 cannot promote T37 regardless of outcome. The period is the already opened
2025-2026 retrospective, and T38 was motivated by inspected local point
metrics. A pass would support freezing T37 for prospective shadow; a failure
must identify exact currencies/years and becomes the next prospective repair
specification. `production_promoted=false`, sparse AP37 push, runtime routing,
future-only bps, texts, and receipt policy remain unchanged.

## Required audit

- immutable source hashes including the saved T37 predictions and metrics;
- exact reconstruction of every local point metric and bootstrap interval;
- complete expected grids and unique keys;
- date-grouped sampling with both block lengths;
- saved decision exactly recomputed from the frozen gates;
- no model, runtime, or push mutation.
