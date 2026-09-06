# T24 preregistration: history-only multiscale h20 anchor

Registered 2026-09-06 before computing T24 outputs. T20--T23 show that mapping
changes cannot repair weak pre-receipt h20 discrimination. T24 therefore tests
new causal daily observables rather than another calibrator: multiscale level,
trend, volatility, common-RUB and own-minus-common trajectory features. It is
intended for overnight/premarket holding until a live market-prefix expert is
available.

## Frozen timeline

- Target: effective-reference `target_now_favourable`, h20.
- Model fit: publication rows before 2024-01-01 whose h20 outcomes mature before
  that origin minus two-day embargo.
- Probability calibration: calendar queries 2024-01-01 through 2024-06-30.
- Candidate selection: queries from 2024-07-01 whose h20 outcomes mature before
  2025-01-01 minus embargo.
- Evaluation: opened 2025--2026 diagnostic.
- Query state: 00:15 history-only. The same saved score may be held at 06:00 and
  09:15 only while no fresher causal source exists; duplicated clocks are not
  counted as independent evidence.

All features end at the latest CBR publication available to the query. Weekend
queries reuse that publication score with older provenance; they do not create
synthetic zero-return observations.

## Frozen candidates

1. `identity_early`: current T17 frozen h20 probability.
2. `compact_logit`: L2 logistic C=0.1 on a fixed compact set of trailing returns,
   range/rank, volatility, moving-average distance, USD/CNY, peer and cyclical
   calendar features plus currency one-hot.
3. `extended_hgb`: all 279 causal extended features; learning rate 0.03,
   depth 2, 200 iterations, min leaf 80, L2=20.
4. `recent_hgb`: same model with exponential sample weights of 730-day half-life
   clipped below at 0.15. This encodes gradual regime relevance rather than a
   hand-picked SVO split.
5. `path_extra`: ExtraTrees with 400 trees, depth 6, min leaf 40 and max-features
   0.5 on the 492 causal own/common/residual trajectory summaries.

Each non-identity raw score receives one Platt map fitted only on the calibration
period. On selection, a candidate is feasible only if AUC is above identity,
Brier and log-loss are no worse, and ECE is no more than 0.01 worse. Among
feasible models select maximum AUC with fixed complexity tie order compact,
extended, recent, path. Otherwise select identity. No refit or hyperparameter
change follows evaluation.

## Evaluation gate

Primary promotion requires positive AUC delta, negative Brier and log-loss
deltas, ECE delta <= 0.01, and both 20- and 50-date moving-block intervals
strictly positive for AUC and strictly negative for Brier. Report currency/year
slices and local ECE. Because 2025--2026 is opened, a pass creates only a frozen
prospective challenger.

## Required audits

- model-fit and calibration/selection outcomes mature before their cutoffs;
- model fit, calibration, selection and evaluation dates are disjoint;
- mapping a calendar query uses the latest publication date no later than it;
- changing every future rate/target leaves the retained prefix unchanged;
- changing evaluation targets leaves fitted models, selection and predictions
  unchanged;
- all probabilities finite/bounded; source hashes and model details persisted.
