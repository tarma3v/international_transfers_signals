# T23 preregistration: delayed online h20 calibration

Registered 2026-09-06 before computing T23 outputs. T22 showed that a fixed
cross-horizon correction is useful after receipt but changes sign elsewhere.
T23 tests whether a causal delayed calibrator can adapt by information state
using only outcomes that have already matured. The opened 2025--2026 period is
still diagnostic; this experiment designs a prospective mechanism rather than
creating a fresh holdout.

## Frozen rank model and stream

- Target: effective-reference `target_now_favourable`, h20.
- States: two receipt scenarios and the same 20 Moscow clocks.
- Frozen rank fit: query before 2024-09-01, h20 label mature before that origin
  minus two-day embargo.
- Calibration stream begins 2024-09-01.
- Evaluation begins 2025-01-01.
- At the first day of each evaluation month, history includes only rows whose
  h20 maturity precedes that monthly origin minus the embargo. The mapping is
  then frozen for the whole month.

The base ranker and base-only residual score are exactly the T22 construction.
Every monthly mapper uses at most the latest 1,000 eligible rows (200 complete
five-currency dates), never the current month target.

## Frozen monthly selection

Eligible historical dates are split chronologically: first 75% fit, last 25%
screen. Both parts must contain both classes and at least 150/50 rows. Candidate
families use L2 logistic regression with `C=0.05`:

1. `identity`: frozen h20 probability, no fit.
2. `anchor_logit`: recalibrate `logit(frozen_h20)`.
3. `joint_logit`: recalibrate `[logit(frozen_h20), base-only residual_z]`.

A fitted family is feasible on the trailing screen only if Brier and log-loss
do not exceed identity and ECE delta is <= 0.01. Among feasible candidates plus
identity choose the highest AUC; ties within 1e-12 prefer identity, then anchor,
then joint. Refit the selected family on all eligible history and freeze it for
the month. Insufficient/single-class history falls back to identity.

Save primary `delayed_selected` plus `delayed_anchor` and `delayed_joint`
controls, where each control is refit monthly regardless of selection when
history is sufficient.

## Evaluation gate

Versus identity, a state passes only with positive point AUC delta, negative
point Brier and log-loss deltas, ECE delta <= 0.01, and both 20- and 50-date
paired moving-block intervals strictly positive for AUC and strictly negative
for Brier. Report overall, currency, year, currency-year, receipt phase,
monthly choice frequencies and local ECE. No result changes AP37 or production
without prospective confirmation.

## Required audits

- every mapper row has `maturity < monthly_origin - embargo`;
- monthly predictions are invariant to every target from that month onward;
- fit/screen are chronological and disjoint;
- base rank fit, residualization and preprocessing never use stream targets;
- all feature sources are no later than query;
- predictions are finite/bounded, hashes and monthly logs are persisted.
