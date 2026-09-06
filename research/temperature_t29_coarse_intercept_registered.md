# T29 preregistration: coarse-period delayed intercept

Registered 2026-09-06 before computing T29 outputs. T28 showed that a daily
global intercept can improve probability losses yet damage rank because the
shift changes between neighbouring dates. T29 updates only at a calendar month
or quarter boundary and holds the shift constant until the next boundary.

## Frozen causal input

- Rebuild the T27 stream: 247 unique 2023 OOS publication dates plus the T25
  calendar query from 2024 onward.
- At a period boundary `b`, feedback is eligible only when it is a unique CBR
  publication event before `b` and its full h20 maturity is strictly before
  `b - 2 days`.
- Solve the same T26 global intercept with ridge 20, minimum 30 publication
  dates and clip `[-1.5, 1.5]`; multiply it by a frozen beta and hold it within
  the period.
- No update occurs because a day passes inside the period. No currency/year/SVO
  switch, intraday input or 2025--2026 target is used.

## Frozen candidate grid

1. quarterly w30 with beta 0.25, 0.50 and 1.00;
2. quarterly w60 with beta 0.50 and 1.00;
3. monthly w30 with beta 0.25, 0.50 and 1.00;
4. monthly w60 with beta 0.50 and 1.00.

Candidate priority is the order above; minimum Brier decides before tie order.
Quarterly candidates preserve T25 rank exactly within a quarter, while monthly
candidates trade some cross-month rank for faster level adaptation.

## Nested pre-2025 decision

- Screen: mature 2024-Q3 rows available before 2024-10-01 minus embargo.
- Validation: disjoint mature 2024-Q4 rows available before 2025-01-01 minus
  embargo.
- On each stage, feasibility against T25 requires lower Brier/log-loss, ECE
  delta <= +0.005 and AUC delta >= -0.005.
- Choose minimum-Brier feasible screen candidate. Keep it only if that exact
  candidate independently passes validation; otherwise retain T25.

The aggregate 2024-H2 period has already been inspected, so this is a nested
retrospective test, not a fresh holdout.

## Open evaluation

Apply the frozen decision to 2025--2026 without changing period, window or
beta. A retrospective pass additionally requires aggregate Brier/log-loss
improvement, ECE delta <= +0.005, AUC delta >= -0.005, negative 20/50-date
Brier CI upper bounds, Brier no worse in either year and no currency worse by
more than 0.005. Even a pass remains shadow-only.

## Required audits

- exact 2023 OOS reconstruction and causal quarterly compact fits;
- every period state uses only feedback mature before its boundary cutoff;
- one constant applied delta for every row/currency within a candidate-period;
- disjoint mature Q3/Q4, selector/validation rebuild and invariance to
  2025--2026 target corruption;
- source/publication times, probability bounds, source hashes and paired grid;
- AP37/push and T22 after-receipt branch unchanged.
