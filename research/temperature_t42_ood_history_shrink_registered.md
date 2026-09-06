# T42 registration: observable OOD shrink for the annual history expert

Registered after T41 was rejected and before any T42 probability or outcome
metric was computed. T40/T41 outcomes are already open, so T42 is a mechanism
test only and can never be promoted from this replay.

## Question

Can a label-free measure of current covariate shift reduce trust in the annual
T40 history expert before a new regime damages probability calibration?

## Frozen information and state

- Use the exact T40 annual OOS publication probabilities, causal h20 prior,
  target, maturity and source-row mapping.
- Use the exact 41 T24 compact CBR features. No market, receipt, target-derived
  state, fixed SVO switch or calendar regime flag is added.
- For evaluation year Y, reproduce the exact T40 raw-model training mask:
  publication date before `Y-1-01-01`, finite h20 target, and h20 maturity
  strictly before that origin minus the existing two-day embargo.
- Fit one `StandardScaler` on those training features only. Save the complete
  mean, scale, support and latest training date/maturity for every year.
- For each query row compute squared standardized distance
  `energy = mean(z_j^2)` across all 41 features.
- The sole trust formula is `alpha = min(1, 1 / energy)`, with numerical floor
  `1e-12`. Thus an in-distribution row with energy at most one keeps T40, while
  an outlying row is smoothly shrunk toward the causal prior.
- The candidate is the linear probability pool
  `prior + alpha * (T40 - prior)`, clipped only to `[1e-6, 1-1e-6]`.

No distance family, clipping level, alpha curve, threshold, currency map,
feature subset, model refit or outcome gate is searched. Alpha depends only on
features and the earlier training distribution, never on query/future labels.

## Frozen evidence and gates

- historical screen: annual OOS publication rows in 2019-2022;
- historical validation: annual OOS rows in 2023-2024;
- open diagnostic: 2025-2026 only if both historical stages pass;
- paired circular moving-block bootstrap resamples whole publication dates and
  all five currencies together, 1,000 fixed-seed draws, blocks 20 and 50.

Use T40's exact historical rules. In both screen and validation, pooled Brier
and log-loss deltas versus causal prior must be below zero, AUC delta above
zero, ECE delta no greater than +0.01, every Brier CI upper bound below zero,
and every AUC CI lower bound above zero. Every year and currency must be
non-inferior under +0.001 Brier, +0.003 log-loss, +0.01 ECE and -0.005 AUC.

If either historical stage fails, do not evaluate 2025-2026 model metrics. If
both pass, replace T37 only on `cbr_history` rows and apply the exact T40 open
gates: 40/40 state non-inferiority, both pooled scenarios, all four years, more
than 619/680 clock-local and more than 25/34 pooled-local passes, exact
non-history preservation and causal timestamps.

## Interpretation

T42 never changes production, sparse push, expected future bps, source
availability, runtime or client copy. A pass supports one frozen prospective
OOD challenger. A failure means feature-space novelty alone cannot turn the
unstable long-history expert into a reliable h20 temperature.
`production_promoted=false` in every outcome.

## Required audit

- source hashes for registration, code, audit, T40 evidence and feature data;
- exact reconstruction of annual scaler state, row distances, alphas,
  probabilities, historical metrics, bootstrap and gates;
- equality of the T42 and T40 row keys/source mapping;
- exact reproduction of every T40 training mask and embargo;
- future feature, target and source-probability corruption may not change an
  earlier prefix;
- conditional open decision and exact non-history preservation if reached.
