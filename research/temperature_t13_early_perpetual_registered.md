# Temperature T13 frozen protocol — completed perpetual-FX candles before 10:00

Registered 2026-09-06 before any T13 model output was computed.

## Question

Can completed MOEX CNYRUBF and USDRUBF hourly candles provide a materially
better calibrated widget probability at 09:00, when the modern CNYRUB spot
archive has no same-day candle, and do they add anything to the T10 spot route
at 10:00?

## Frozen information set

Use the immutable hourly perpetual archive already in the repository. A candle
is visible only when its recorded `end` is strictly before the requested clock
and its nominal one-hour interval is complete. Use same-day candles only.
Current effective CBR history and the T4 compact history feature set are
allowed. Tomorrow's CBR value, end-of-day market values and target-neighbour
outcomes are forbidden.

Evaluate clocks 09:00 and 10:00 Moscow. A market candidate is eligible only
when a completed same-day CNYRUBF candle physically exists and its calibrated
probability is finite. Otherwise route exactly to the control. The control is
T4 history-only at 09:00 and the frozen T10 spot-or-history route at 10:00.

## Frozen candidates

For every h=1/3/5/10/20 compare three candidates:

1. causal rank of CNYRUBF mean price versus current effective CBR;
2. causal rank of the CNYRUBF same-day open-to-cutoff return;
3. one global HistGradientBoosting model on the fixed T4 history features plus
   the two perpetual prefixes, volume, range, slope, age and missingness.

The global model uses quarterly walk-forward fits, training rows with a
physically available CNY candle only, exponential half-life 730 days, a
two-day embargo through target maturity, and the already frozen T4 HGB
hyperparameters. Each raw candidate receives quarterly mature-only Platt
calibration on available history only. No candidate or parameter may be added
after seeing 2025–2026.

## Selection and reporting

2024 is the only selection screen. For each clock and horizon, identify the new
candidate with the lowest 2024 Brier loss. Adopt it over the clock-specific
control only if both 20-date and 50-date paired circular-block 95% intervals for
the Brier-loss delta end below zero. Otherwise retain the control. The
2025–2026 period is opened once for diagnosis and must not change this rule.

Report Brier, log loss, ECE, AUC and average precision on the full routed rows,
plus physical availability by year and clock, currency/year slices, reliability
bins, and paired 20/50-date Brier deltas. Negative results remain saved.

## Mandatory audits

Verify source hashes, exact persisted-array rebuild, `source_at < as_of`,
nominal candle completion, future-candle corruption invariance, future target
and maturity prefix invariance, exact fallback on unavailable rows, selection
identity and the absence of tomorrow's CBR. T13 changes no push policy and makes
no bank-execution claim.
