# Deep research round two — frozen protocol

Date frozen: 2026-09-04.

## Objective

Find materially different, strictly past-only approaches for `fav_h5`. The
primary success criterion is future-only lift above 1.30 at 1--2 alerts per
corridor per week, accompanied by positive future-only benefit and reasonable
stability across years and currencies. The next effective CBR rate is excluded
from every ordinary-model feature.

The previously discovered anchor rules and the post-publication policy are
comparators only. They must not define the new candidate search.

## Validation status

The 2024--2026 period has already been inspected in round one. No newly invented
round-two policy can call it a genuinely unseen holdout. Model fitting and
selection will remain mechanically causal, but round-two results on this period
will be labelled retrospective. Strong claims require either nested rolling
selection across many outer years or future observations collected after this
protocol date.

## New research families

1. Local probabilistic models fitted separately to each target currency.
2. Global pooled models with currency identity and common-factor features.
3. The user-proposed tower: local per-currency base model, strictly out-of-fold
   base predictions, then a global booster across all currencies that learns an
   offset/residual correction.
4. The reverse tower: global base followed by local residual correction.
5. Local time-series path forecasts (drift/volatility, ETS, autoregressive and
   nearest-neighbour analogue forecasts) converted into the probability that
   the current value survives as the minimum for five publications.
6. Dynamic mixtures of experts whose weights depend only on trailing validation
   performance and observable regimes.
7. Regime models: volatility/trend states, common cross-currency factors,
   unsupervised states fitted inside each training fold, and explicit pre/post
   2022 adaptation.
8. Data enrichment candidates with release-aware joins only: monetary-policy,
   liquidity/interest-rate, oil/global-dollar and event features where an
   authoritative historical source and publication timestamp are available.
9. Calendar and seasonal effects only when they repeat out of sample and add
   value conditionally rather than merely correlating in the full history.
10. Negative controls, label permutation, truncation tests, parameter-neighbour
    plateaus, block bootstrap and multiplicity-aware comparisons.

## Dedicated hypothesis: error-routed mixture of experts

For every sufficiently strong but imperfect model, retain strictly out-of-fold
errors and investigate where they concentrate: currency, trend direction,
volatility, common-factor strength, residual-vs-common movement, calendar gap,
season and distance from a recent extremum. Train a causal gating model on these
older OOF errors to predict which expert should be trusted, or to assign convex
weights. Compare hard routing, soft probability weighting and a simple trailing
loss-weighted ensemble.

The gate may use only information observable at signal time. A date interval or
regime discovered from the same future outcomes on which it is scored is not an
eligible routing feature. The gate itself must receive a second layer of
chronological OOF evaluation; otherwise error analysis would merely move the
leakage up one level.

## Required outputs

- Train-only/regime EDA tables and figures.
- Candidate registry including failures, not only winners.
- Yearly and corridor-level metrics for every finalist.
- A nested walk-forward comparison with thresholds learned from prior data.
- A separate retrospective 2024--2026 audit with explicit contamination label.
- Reproducible code, machine-readable CSV results and a verified PDF report.
