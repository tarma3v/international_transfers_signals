# T33 preregistration: quarterly frozen stacking of regime experts

Registered before computing T33 outputs. T31/T32 showed that mature-only
feedback is causal, but changing expert weights on almost every publication can
damage cross-date rank. T33 isolates that mechanism by fitting weights only at
calendar-quarter origins and holding them fixed for the complete quarter.

## Frozen inputs

Use exactly the immutable T30 publication-event columns from 2023 onward:

1. T4 `identity_early`;
2. T30 `all_platt_b050`;
3. T30 `recent2y_platt_b100`.

No expert, probability map or T25 anchor is refit. T30 source hashes must pass.
There is deliberately no pre-2023 warm-up after T32 rejected it.

## Quarterly fit

At each quarter origin, admit only earlier publication batches whose maximum
h20 `maturity_ord` is strictly below `(origin - 2 days).toordinal()`. Choose the
last W eligible unique publication dates, pool their currency rows, and solve a
non-negative simplex mixture minimizing mean Bernoulli log-loss plus
`ridge * sum((w - 1/3)^2)`. If fewer than 20 feedback dates are available, use
uniform weights. Weights are fitted independently from uniform at every origin
and remain constant until the next quarter.

Frozen grid:

- W in `{60, 125, 250, expanding}` publication dates;
- ridge in `{0.00, 0.01, 0.10, 1.00}`;
- 16 candidates; priority is short-to-long window, then low-to-high ridge.

## Chronological decision

- Screen: mature 2023 rows. Feasible means AUC delta versus identity >= 0.02,
  lower Brier and log-loss, and ECE delta <= 0.005. Select minimum Brier, then
  frozen priority.
- Validation: only the exact screen winner on mature 2024 rows, same gates. No
  second-best substitution.
- Evaluation: opened 2025--2026, diagnostic only. Continue quarterly fits from
  past mature outcomes, but do not select any hyperparameter on open labels.

If screen or validation fails, `t33_selected` equals identity. Persist the exact
screen candidate even when rejected. Any historical pass remains a prospective
shadow because 2023--2026 have already been inspected.

## Required audits

- source hashes, complete expert columns and unique publication keys;
- one state per quarter/candidate, finite non-negative weights summing to one;
- every training publication earlier than origin and fully mature before its
  embargo cutoff;
- weight constancy for every row in a quarter;
- independent rebuild of all 16 mixtures from saved states;
- exact screen winner, validation verdict and fallback;
- corrupting targets unavailable at an origin cannot change that origin or any
  earlier prediction;
- complete overall/year/currency/currency-year Brier, log-loss, ECE, AUC/AP;
- paired 20/50-date AUC/Brier/log-loss intervals against identity and T25;
- explicit open non-selection and `fresh_independent_holdout=false`.
