# T41 registration: mature quarterly shrink of the long-history expert

Registered after T40 was rejected and before any T41 probability or outcome
metric was computed. T40 historical metrics are already open, so T41 can never
be promoted from this replay. It is a mechanism test and, if successful, a
frozen prospective challenger only.

## Question

Can a parameter-free causal trust weight prevent the stationary T40 expert from
damaging probability calibration when the CBR regime changes, while retaining
its useful rank during stable quarters?

## Frozen candidate

- inputs are the exact annual OOS publication probabilities, causal global
  prior, h20 target and maturity saved by T40;
- at each calendar-quarter origin, collect publication-date batches whose h20
  maturity is strictly earlier than origin minus the two-day embargo;
- use the most recent 125 eligible publication dates, inherited from the
  strongest fixed T33 quarterly mechanism; no window grid is allowed;
- if fewer than 20 eligible batches exist, use alpha zero, inherited from T34
  safe cold-start semantics;
- otherwise choose the unique scalar alpha in `[0,1]` that minimizes Brier loss
  of `prior + alpha * (T40 - prior)` on those eligible rows. The solution is
  the closed-form clipped least-squares coefficient, not a searched grid;
- freeze alpha for the whole quarter and apply the same value to all five
  currencies;
- do not add a calendar SVO switch, currency weight, market feature,
  hyperparameter search, or outcome gate.

The candidate is a linear probability pool. Alpha zero is the causal prior;
alpha one is T40. Every state must save origin, cutoff, selected batches,
latest publication and maturity, numerator, denominator, alpha and row count.

## Frozen evidence and gates

- historical screen: 2019-2022;
- historical validation: 2023-2024;
- open diagnostic: 2025-2026 only after both historical stages pass;
- paired circular moving-block bootstrap resamples whole publication dates and
  all currencies together, 1,000 deterministic draws, blocks 20 and 50.

Use T40's exact historical rules. In screen and validation separately, pooled
Brier/log-loss deltas versus the causal prior must be below zero, AUC delta
above zero, ECE delta no greater than +0.01, every Brier CI upper bound below
zero and every AUC CI lower bound above zero. Every year and every currency
must be non-inferior under +0.001 Brier, +0.003 log-loss, +0.01 ECE and -0.005
AUC allowances.

If either historical stage fails, do not evaluate 2025-2026 model metrics and
end the experiment. If both pass, replace T37 only on `cbr_history` rows using
the T40 publication-date join and apply the exact T40 open gates: 40/40 state
non-inferiority, both pooled scenarios, all four year groups, more than 619/680
clock-local and more than 25/34 pooled-local passes, causal timestamps and
bitwise equality outside history.

## Interpretation

T41 never changes sparse push, expected future bps, source availability,
runtime, client copy or production. A pass supports one frozen prospective
history challenger. A failure shows that quarterly mature-feedback shrinkage
alone cannot convert the unstable annual expert into a reliable h20
temperature. `production_promoted=false` in every outcome.

## Required audit

- source hashes for registration, code, audit and T40 evidence;
- exact reconstruction of states, probabilities, historical metrics,
  bootstrap, gates and any conditional open outputs;
- complete quarterly and publication grids;
- maturity/embargo proof for every state;
- future target and prediction corruption after a cutoff may not change the
  earlier prefix;
- exact decision reconstruction and non-history preservation if open runs.
