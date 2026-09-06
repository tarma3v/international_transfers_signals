# Product goal: continuous transfer temperature

Registered 2026-09-06 as a persistent addition to the after-publication CBR
research. This document is a product/evaluation contract, not evidence that the
full intraday widget is already validated.

## User experience

For every corridor and every requested `as_of` moment, return the latest score
that could really have been computed by then:

- `temperature_0_100`: calibrated attractiveness of transferring now;
- `probability_now_best_h`: probability that the current effective CBR fixing
  is no higher than every fixing in the next `h` publications;
- `expected_future_bps_h`: separately calibrated expected advantage against
  the mean of the next `h` fixings;
- `horizon_publications`: the horizon the user is viewing;
- `score_as_of`, `last_source_at`, `age_minutes`, and a freshness band;
- `phase`: before-new-CBR, after-new-CBR, weekend/holiday, or stale;
- `confidence`: enough mature history / limited history / unavailable;
- `push_now`: a separate sparse binary decision from the frozen push policy.

The widget may say "момент выглядит выгоднее обычного". It must not say that a
future course is guaranteed, and it must not translate CBR basis points into
rubles saved at the bank until executable customer quotes, fees, limits and
quote validity are present.

## Meaning of temperature

Temperature is not the raw model rank and not the push threshold. For a selected
horizon it starts from a causally calibrated probability. A value near 80 means
that comparable historical model states resolved favourably about 80% of the
time after calibration; it does not mean an 80% guaranteed return. A composite
view may use the fixed geometric mean of calibrated h=3/5/10/20 probabilities,
but the four horizon values must remain inspectable.

Expected future benefit is a second output because two moments with the same
success probability can have different monetary magnitude. It uses the
future-only target, never the symmetric +/-h metric, and stays in CBR basis
points until bank execution data are available.

## Information-time contract

Every feature, peer quote, model, calibration mapping and freshness label must
be a function only of records with `received_at <= as_of`. Model training may
use only labels whose full horizon matured before the fit origin, with the
existing embargo. Corrupting every future feature, target, timestamp and source
must leave the returned historical prefix unchanged.

The current historical CBR replay has calendar-assumed receipt events rather
than certified publication timestamps. Therefore its after-publication widget
is a daily latest-valid-snapshot prototype. Between two valid updates the score
is held constant and becomes progressively stale; time passing alone must not
invent a fresh prediction.

A truly varying intraday temperature requires timestamped information such as
completed MOEX candles or executable bank quotes. It must be replayed at fixed
pre-registered slices or event times and grouped by day in validation. A whole
day's high/low/close/volume cannot be used before that day ends. Missing trading
and weekends produce an explicit stale state, not imputed current prices.

## Model plan

1. Calibrate the four AP49 OOS probabilities quarterly using only mature prior
   rows. Compare logistic calibration with a train-prior baseline; do not select
   a mapper on opened 2024--2026.
2. Fit a separate robust causal regressor for future-only basis-point benefit
   at each horizon. Report error and calibration by predicted-benefit bins.
3. Implement `score_as_of(currency, timestamp, horizon)` that selects the latest
   admissible snapshot and returns freshness/phase metadata.
4. Add intraday market snapshots only after their timestamps and availability
   conventions are verified. Keep CBR-only and market-enhanced scores separate.
5. Preserve the sparse AP37/AP49-family push router. Push firing must not be
   required for the widget to return a score.

The concrete time-of-day routing design and known coverage gaps are frozen in
`research/transfer_temperature_router_design.md`. In particular, a 15:30 model
may be held after 15:30 as a stale-but-causal anchor, but it is not silently
treated as an updated 17:30 estimate; post-window candles need their own causal
correction and validation.

## Acceptance evidence

- chronological quarterly OOS predictions with mature labels and embargo;
- Brier score and log-loss against a train-only constant prior;
- reliability table/plot and expected calibration error overall, by currency,
  horizon, year and phase;
- discrimination (ROC AUC and average precision) reported separately from
  calibration;
- future-only expected-bps MAE and bin calibration;
- timestamp/staleness unit tests and future-prefix corruption audit;
- coverage for weekdays, weekends, missing data and a user arriving hours after
  the push;
- no claim of real bank savings until quote/fee data pass an execution audit.

Push and widget can therefore improve independently: the push takes only a few
top opportunities to maximize lift, while the widget remains defined and
honestly qualified for every `as_of`.
