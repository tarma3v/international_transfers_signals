# T22 preregistration: small h20 rank correction around frozen probability

Registered 2026-09-06 before computing T22 outputs. T21 showed that its
cross-horizon logistic control ranks h20 better but is unusable as a probability.
T22 tests whether a small correction can preserve the frozen probability anchor
while importing only part of that rank information. The opened 2025--2026
period remains diagnostic and cannot make this a fresh winner.

## Frozen target, states and split

- Target: `target_now_favourable` at h20, anchored to the CBR rate effective on
  the query date.
- States: the same two receipt scenarios and 20 Moscow clocks as T19--T21.
- Rank fit: query dates before 2024-09-01, labels fully mature strictly before
  2024-09-01 minus the two-day embargo.
- Weight calibration: 2024-09-01 through 2024-12-31, labels fully mature before
  2025-01-01 minus the same embargo.
- Evaluation: 2025-01-01 through the frozen T19 end, already-open diagnostic.

## Frozen correction

For each `scenario × clock`:

1. Fit an L2 logistic ranker with `C=0.1` on the T21 causal feature set and only
   the base period.
2. On base data, regress its decision score on `logit(frozen_h20)` by ordinary
   least squares. Standardize the residual using base-only mean and scale. This
   is the incremental rank direction not already present in the frozen anchor.
3. For `alpha ∈ {0, 0.01, 0.025, 0.05, 0.10, 0.20}`, form
   `sigmoid(logit(frozen_h20) + alpha × residual_z)`.
4. Select alpha on the disjoint calibration period only. An alpha is feasible
   when versus frozen identity it has Brier delta <= 0.001, log-loss delta <=
   0.003 and ECE delta <= 0.02. Among feasible alphas choose the highest AUC;
   ties within 1e-12 choose the smaller alpha. If none is feasible, use alpha 0.

Controls saved on evaluation:

- `identity_h20`;
- `rank_correction_selected` (primary);
- each fixed-alpha probability in the frozen grid, for diagnosis only.

The primary gate versus identity requires positive point AUC delta, lower
paired moving-block AUC bounds above zero for both 20- and 50-date blocks,
negative point Brier and log-loss deltas, Brier upper bounds below zero for both
blocks, and ECE delta <= 0.01. Also report alpha coverage, zero-alpha fallbacks,
AP, local currency-year ECE and fixed-alpha Pareto tables. The opened evaluation
does not select alpha or change the production router.

## Required audits

- all fit/calibration labels mature before their cutoffs;
- base and calibration periods are disjoint;
- every probability/benefit source timestamp is no later than the query;
- standardization and residualization use base only;
- changing evaluation targets leaves selected alpha and predictions unchanged;
- all probabilities are finite and bounded;
- hashes, coefficients, residual scaling, selected alpha and calibration metrics
  are persisted.
