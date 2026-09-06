# Temperature T15 frozen protocol - evening perpetual FX after spot close

Registered 2026-09-06 before T15 outputs or metrics were computed.

## Question and immutable clocks

Test whether physically completed CNYRUBF and USDRUBF hourly candles improve
the continuous widget after the frozen 18:30 after-publication decision and
after the useful currency-spot window. Query clocks are fixed at 20:00, 21:00,
22:00 and 23:00 Moscow. The control at every clock is the last T7B 20:00
probability and expected-benefit state. This deliberately asks whether futures
add information beyond the already available after-publication and spot state.

At each clock, a futures candle is admissible only when it is from the same
calendar day, its recorded end is strictly before the query clock, and its
nominal hourly end is no later than the query clock. The feature must contain at
least one candle newer than the latest candle admissible at the 18:30 base. The
persisted source timestamp is the maximum recorded end actually used. No
unfinished 23:00-23:59 bar may enter the 23:00 snapshot.

## Frozen candidate family

For h=3/5/10/20 test exactly three probability candidates:

1. `perp_cny_logit`: base logit plus CNY return, absolute return, last-hour
   return, range, volume, bar count, availability and fixed currency dummies.
2. `perp_dual_logit`: base logit plus the full frozen CNY/USD return, range,
   volume and divergence vector, availability and currency dummies.
3. `perp_dual_hgb`: one global shallow histogram gradient boosting model on
   the full vector, base logit and currency dummies, followed by causal
   quarterly Platt calibration.

The only magnitude challenger is `perp_dual_ridge`: AP51 multihorizon features,
the carried 20:00 probability, the full frozen CNY/USD vector and availability.
h=1 is already known from the current and announced CBR rates and is copied
exactly; it is not a forecast or a candidate.

All models refit only at calendar-quarter origins. Training labels must be
mature before origin minus two days, training starts no earlier than the
existing post-2022 minimum date, and observations receive a fixed 730-day
half-life. Missing physical futures data falls back exactly to the control.

## Frozen selection and reporting

Selection uses 2024 only. For each clock and horizon, choose the lowest-Brier
probability candidate on the all-currency screen, breaking ties by candidate
name. Adopt it only if paired circular date-block bootstrap confidence
intervals for candidate-minus-control Brier have upper bound below zero at both
20- and 50-date blocks. Otherwise retain the control.

For magnitude, compare `perp_dual_ridge` with the control by absolute error and
adopt only when both 20- and 50-date 95% interval upper bounds are below zero.
The already open 2025-2026 period is diagnostic only and cannot change the
selection. Report all-currency, currency and year slices plus availability,
Brier/log-loss/ECE/AUC/AP, MAE/rank correlation and paired intervals.

Independently corrupt future candles, outcomes and maturity dates and require
past outputs to remain bitwise or numerically identical. Exact reconstruction,
exact unavailable fallback and exact h=1 copying are mandatory. Historical CBR
receipt timestamps remain calendar-assumed; the experiment makes no claim
about executable bank quotes, commissions or actual customer savings.
