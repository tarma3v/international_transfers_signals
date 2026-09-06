# Temperature T19 frozen protocol — unified anytime quality audit

Registered 2026-09-06 before T19 metrics, bootstrap intervals or weak-spot
tables were computed.

## Purpose

T3--T17 validated individual model phases, while T18 repaired the live receipt
state machine. T19 evaluates the final frozen snapshot stream as one any-time
widget. It is diagnostic only: 2024--2026 is already open and cannot select a
new model, calibration mapping, clock, threshold or confidence rule.

## Frozen query grid

Starting with the first Moscow calendar day for which every corridor already
has a snapshot by 00:15, and then for every day through the final artifact day,
select the latest valid snapshot for every corridor and the following
user-arrival clocks. The partial first artifact day is reported as a boundary,
not backfilled from a future evening row:

`00:15, 06:00, 09:15, 10:15, 10:45, 11:45, 12:45, 13:45, 14:45,
15:10, 15:25, 15:45, 16:45, 17:45, 18:45, 19:15, 20:15, 21:15,
22:15, 23:15` Moscow.

Evaluate two explicitly separated paths:

1. `calendar_assumed_replay`: the historical research convention permits the
   assumed 18:30 receipt and therefore measures frozen after-publication heads;
2. `no_same_day_receipt`: every receipt-dependent row from the query's own day
   is unavailable, so the router must hold the latest pre-receipt or prior-day
   state. This measures T18 fallback quality, not a claim about actual receipt
   times.

No hypothetical verified timestamp is labelled as historical fact.

## Target alignment

At every query, the reference is the CBR rate already effective on that Moscow
calendar day. Map the day to the latest corridor publication date not later
than the day. Targets are:

- `fav_h`: current effective rate is no higher than every next `h` corridor
  publication;
- `future_bps_h`: advantage of the current effective rate against the mean of
  the next `h` publications.

Use all `h=1/3/5/10/20`. Rows whose future horizon has not matured are excluded
from scoring, never imputed.

## Strictly past baseline

For every query/h, compare against a global constant estimated from unique
corridor-publication events whose target maturity date is strictly earlier than
`query_day - 2 calendar days`. The probability baseline is the past favourable
rate; the benefit baseline is the past mean future-only bps. Each underlying
event enters the baseline once regardless of the number of clocks.

## Frozen metrics

Probability: count, positive rate, mean prediction, Brier, log-loss, 10-bin ECE,
ROC AUC and average precision, with the same metrics for the strictly past
baseline. Benefit: count, mean predicted/actual bps, MAE, RMSE, signed bias,
Spearman correlation, and past-baseline MAE/RMSE.

Report:

- overall by scenario, query clock and horizon;
- by currency;
- by year;
- by currency×year;
- reliability bins overall and by currency;
- freshness/source/phase coverage by clock, weekday and scenario.

For each scenario×clock×h, compute paired circular moving-block bootstrap over
Moscow query dates for daily mean Brier-loss and absolute-error differences
against the past baseline, using fixed block lengths 20 and 50 dates, 1,000
draws and deterministic seeds. Identical held daily-loss sequences reuse the
same seed so that a display clock alone cannot change a bootstrap verdict.
Negative difference favours the router.

## Frozen interpretation gates

- Do not promote or recalibrate from this open audit.
- Call a state `probability_supported` only when both block-size upper bounds
  for Brier difference are below zero.
- Call a state `benefit_supported` only when both upper bounds for absolute-error
  difference are below zero.
- Flag ECE above 0.08, AUC below 0.55, a non-negative paired upper bound, fewer
  than 100 scored rows, and any currency/year slice with missing output.
- A held prediction is not relabelled as a fresh update. Source timestamps and
  horizon-specific provenance must remain no later than query time.
- Historical receipts and bank execution prices remain uncertified.

The next experiment may address a weak state only with a new preregistered
earlier-data design; it may not choose a repair from T19's 2024--2026 outcomes.
