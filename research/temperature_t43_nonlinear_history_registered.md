# T43 registration: nonlinear annual history expert with observable state interactions

Registered after T42 was rejected and before any T43 fit, probability or
outcome metric was computed. T40-T42 outcomes are already open, so T43 is a
mechanism test only and can never be promoted from this replay.

## Question

Did the long-history failures come from forcing one linear feature-to-target
relationship across regimes? Test one nonlinear tree model that can represent
interactions among the already causal momentum, volatility, range, common-move
and seasonal features without a hand-written SVO switch or query-period label.

## Frozen model and information contract

- Use the exact T40 base, target, maturity, causal prior, annual train,
  calibration and query masks, and the exact 41 T24 compact CBR features.
- No market, receipt, target-derived query state, fixed SVO flag, calendar
  regime switch, feature selection or open-period routing is added.
- For each evaluation year `Y`, fit exactly one
  `HistGradientBoostingClassifier` on T40's raw training mask with:
  `loss=log_loss`, `max_iter=160`, `learning_rate=0.05`,
  `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=5`,
  `early_stopping=false`, and seed `20260906`.
- These capacity parameters are inherited from the pre-existing AP1 classical
  model family; there is no T43 grid or alternative candidate.
- Score T40's complete previous-year calibration mask and current-year query
  mask. Fit the same single Platt logistic (`C=1`, L2, lbfgs) on the clipped
  raw calibration logit and apply it unchanged to the query year.
- Final probability is the same fixed 50% log-odds blend with T40's causal
  prior. This isolates nonlinear regime interaction from calibration/blending.
- Generate annual OOS rows for 2019-2026 without selecting years, currencies,
  features, hyperparameters, probability thresholds or blend weights.

Every cutoff, support count, latest maturity, fitted iteration count, Platt
coefficient and publication prediction must be saved. Changing targets or
features after a query year may not change its historical prediction prefix.

## Frozen evidence and gates

- historical screen: annual OOS publication rows in 2019-2022;
- historical validation: annual OOS publication rows in 2023-2024;
- open diagnostic: 2025-2026 only if both historical stages pass;
- paired circular moving-block bootstrap resamples whole publication dates and
  all five currencies together, 1,000 fixed-seed draws, blocks 20 and 50.

Use T40's exact gates. In each historical stage, pooled Brier and log-loss
deltas versus causal prior must be below zero, AUC delta above zero, ECE delta
at most +0.01, every Brier CI upper bound below zero, and every AUC CI lower
bound above zero. Every year and currency must be non-inferior under +0.001
Brier, +0.003 log-loss, +0.01 ECE and -0.005 AUC.

If either stage fails, do not evaluate 2025-2026 model metrics. If both pass,
replace T37 only on observed `cbr_history` rows and apply T40's exact open
gates: 40/40 state non-inferiority, both pooled scenarios, all four pooled
years, more than 619/680 clock-local and more than 25/34 pooled-local passes,
exact non-history preservation and causal timestamps.

## Interpretation

T43 tests one new hypothesis: whether bounded nonlinear feature interactions
repair the regime-dependent mapping that distance-only shrinkage could not.
A pass creates one frozen prospective challenger. A failure rejects this
specific capacity change; it does not prove that all state-aware models fail.

T43 never changes production, T37, sparse push, expected future bps, source
availability, runtime or client copy. `production_promoted=false` in every
outcome.

## Required audit

- source hashes for registration, code, audit, T40 implementation, feature
  data and conditional open-route evidence;
- exact reconstruction of all annual fits, predictions, historical metrics,
  bootstrap and gates;
- complete annual/currency grids and exact T40 maturity/embargo masks;
- fixed estimator parameters and `n_iter=160` for every annual fit;
- future feature/target corruption may not change an earlier prefix;
- conditional open decision and exact non-history preservation if reached.
