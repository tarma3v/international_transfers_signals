# T21 preregistration: cross-horizon h20 discrimination head

Registered 2026-09-06 before computing T21 results. T19 showed that no h20
state passes the strict probability gate. T20 showed that a one-shot calibration
map cannot repair this. T21 therefore changes the rank model, not merely the
probability mapping. It does not change AP37, push cadence, expected-benefit
heads or T17/T18 availability semantics.

## Frozen target and split

- Target: `target_now_favourable` at h20, anchored to the CBR rate effective on
  the query date.
- Scenarios: `calendar_assumed_replay` and `no_same_day_receipt`, separate.
- Clocks: the same 20 T19 clocks.
- Base fit: query date before 2024-09-01 and target maturity strictly before
  2024-09-01 minus the two-day embargo.
- Calibration: query date from 2024-09-01, target maturity strictly before
  2025-01-01 minus the embargo. This is later than the base fit and remains
  disjoint from it.
- Evaluation: 2025-01-01 through the frozen T19 end. It is already open and
  diagnostic only.

## Causal features

At each query use the frozen snapshot's probability curve h1/h3/h5/h10/h20,
expected future-bps curve, short-vs-long contrasts, within-query relative
position across the five currencies, source age, currency and the exact causal
state `phase|source_kind|confidence|freshness`. Every horizon-specific
probability and benefit source timestamp must be no later than `as_of`.

No raw future rate, target, later clock or end-of-day candle is a feature.

## Frozen candidates

1. `identity_h20`: the frozen T17 h20 probability.
2. `curve_logit`: L2 logistic regression on the joint curve, fit on every
   mature pre-2025 row; fixed `C=0.1`.
3. `curve_hgb_platt`: primary. HistGradientBoosting with learning rate 0.04,
   max depth 2, 120 iterations, min leaf 60 and L2=10, fit on the base period.
   A Platt map with C=1 is fit only on the disjoint calibration period.
4. `curve_extra_platt`: ExtraTrees control with 300 trees, max depth 5,
   min leaf 40 and max features 0.8, followed by the same disjoint Platt map.

Models fit separately for `scenario × clock`. Categories unseen in training map
to zero. If base or calibration data are insufficient or single-class, the
candidate falls back to identity and records the reason.

## Primary gate

Versus `identity_h20`, a state passes only when:

- mean AUC delta is positive and the lower bounds of both 20-date and 50-date
  paired moving-block 95% intervals are above zero;
- Brier delta is negative and both corresponding upper bounds are below zero;
- log-loss delta is negative;
- ECE is not worse by more than 0.01.

Report point AUC/AP/Brier/log-loss/ECE overall and by currency/year, plus the
number of currency-year rows with ECE > 0.08. No candidate is promoted on the
opened evaluation. A passing primary becomes only a prospective challenger.

## Required audits

- all base/calibration labels mature before their respective cutoffs;
- base and calibration date ranges are disjoint;
- every horizon probability/benefit source is no later than query time;
- changing evaluation labels leaves fit coefficients and evaluation
  predictions unchanged;
- source hashes, feature names, fit/calibration sizes and saved outputs are
  persisted.
