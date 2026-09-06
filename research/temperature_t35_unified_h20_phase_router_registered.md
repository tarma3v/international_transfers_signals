# T35 registration: unified h20 information-phase router

Registered before running T35 or inspecting any T35 metric.

## Question

Can the strongest already frozen h20 components be combined by observable
information availability into one any-time shadow without choosing a clock,
weight, or model on the open 2025--2026 outcomes?

## Frozen route

T35 has one candidate, `unified_h20_phase_shadow`:

1. At 00:15, 06:00, 09:15, and 10:15 Moscow time, use the latest available
   T34 `cold_identity_qstack_w125_r100` publication anchor for the currency.
   The join is backward by publication date; a future publication is never
   used.
2. From 10:30 until a new CBR receipt is actually observed, keep the existing
   T19/T17 routed h20 probability. This includes all 10:45--17:45 states and
   all later states in the no-receipt scenario.
3. After a verified same-day CBR receipt, use the already frozen T22
   `rank_correction_selected` h20 probability. Historical evaluation can only
   show the explicit `calendar_assumed_replay` upper-bound scenario; production
   activation still requires caller-supplied `verified_receipt_at` and must not
   rely on 18:00 or 18:30 as a guaranteed event time.
4. The route does not alter h1/h3/h5/h10, expected future-only bps, freshness,
   confidence, or `push_now`. It is an h20 probability shadow only.

The four early clocks come from the product contract that history is used
before the 10:30 market boundary. They are not selected in T35. The six
after-receipt replay clocks are exactly the states already passed by T22. No
new alpha, threshold, blend, clock grid, or currency rule is searched.

## Evaluation

- Inputs are immutable T22 and T34 prediction artifacts.
- Evaluation is the already open 2025--2026 retrospective and therefore cannot
  produce a fresh independent winner.
- Report both scenarios separately: `calendar_assumed_replay` and
  `no_same_day_receipt`.
- At each scenario/clock, report Brier, log-loss, ECE, AUC, average precision,
  currency/year slices, reliability bins, and paired 20/50-date bootstrap.
- Also report pooled all-clock metrics by scenario, with date-block bootstrap
  that keeps all five currencies and all clocks of a date together.

A state passes only if AUC delta is positive with bootstrap lower bound above
zero, Brier delta is negative with bootstrap upper bound below zero, log-loss
is lower, and ECE delta is no larger than +0.01. Pooled results are descriptive
because repeated clocks from one day are correlated.

## Status rule

`retrospective_route_passed` may be true if every modified state passes and no
unchanged state regresses numerically. `production_promoted` is fixed to false:
the component results and 2025--2026 interval were already inspected before
this composition. The output is a frozen prospective shadow and a causal route
specification, not a production replacement.

## Required audit

- immutable source hashes;
- unique scenario/date/clock/currency keys;
- exact backward T34 join and target alignment on publication dates;
- early route only at the four registered clocks;
- T22 route only in after-receipt replay states;
- no-receipt rows never use T22;
- source/query timestamps remain causal in the source packet;
- corruption of future T34 rows and all target columns leaves the historical
  prefix unchanged;
- complete state/slice/reliability/bootstrap grids;
- no changes to runtime or push policy.
