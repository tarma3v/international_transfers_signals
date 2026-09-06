# Frozen protocol: paired 15:30 versus after-receipt comparison

Registered 06.09.2026 before the paired bootstrap and market waiting-cost
numbers were computed. This is the numerical fast-versus-slow answer required
by the case, not a new model-selection round.

## Fixed candidates

- **Fast:** packet ED `availability_route`, decision at 15:30 Moscow, fixed
  rolling policy rate 0.22 and memory 20, selected on 2024 without the next CBR
  publication.
- **Slow:** AP37 `ap26_core_mature_precision_calendar_fallback_cap2`, decision
  at the calendar-assumed 18:30 receipt state, using the newly announced CBR
  rate as an input while retaining today-effective CBR as target reference.

No threshold, model, horizon or time is changed after seeing this comparison.

## Like-for-like support

Use only 2025–2026 rows sharing the exact `(currency, calendar decision date)`
between the two already frozen artifacts. Require their `now_favourable`,
symmetric and future-only targets to match exactly. This intersection excludes
dates present in only one event convention; report its size explicitly rather
than filling or shifting them.

For each `h=1/3/5/10/20`, report both candidates on the same valid rows:
hit rate, random-day base, corridor-year adjusted lift, signal count,
symmetric benefit and future-only benefit. Report signal overlap at h5.

Uncertainty uses 1 000 circular 20-date block-bootstrap draws. A sampled date
carries all five currencies together. Report the 2.5/97.5 percentiles of the
paired slow-minus-fast difference in adjusted lift, symmetric benefit and
future-only benefit.

## Price of waiting

The official CBR reference cannot measure the intraday 15:30→18:30 execution
cost: the today-effective CBR rate is unchanged by construction. Therefore the
official-series waiting cost is recorded as exactly zero/inapplicable, not as
proof that waiting is free.

As an explicitly weaker market proxy, use CNYRUB_TOM only. For each common
calendar date with both observations, compare the last completed close strictly
before 15:30 with the last close strictly before 18:10 (the slow decision's
18:30 clock minus the frozen 20-minute feed delay), using only candles beginning
at or after 10:00 Moscow. Positive bps means CNY became more expensive in RUB,
so waiting was worse for a RUB sender. Report all dates, dates with at least one
fast signal and dates with at least one slow signal. This is not a bank quote,
fee-inclusive saving or execution guarantee.
