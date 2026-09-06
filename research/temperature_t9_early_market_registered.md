# Temperature T9 frozen protocol — early-session market expert

Registered 2026-09-06 before T9 metrics were computed.

T9 tests one new information source for the weak premarket part of the widget:
completed same-day MOEX CNY/RUB and USD/RUB ten-minute candles before the
existing 10:30 router boundary. Fixed query clocks are 07:30, 08:00, 08:30,
09:00, 09:30 and 10:00 Moscow time. A candle is admissible only when its real
end is strictly before the query clock and its nominal ten-minute end is no
later than that clock. No end-of-day backfill and no tomorrow-CBR value are
allowed.

At every clock compare three frozen candidates on all h=1/3/5/10/20:

1. the existing T4 history-only probability;
2. a transparent causal percentile of same-day CNY mean price versus the
   current effective CBR CNY rate;
3. one pooled quarterly HistGradientBoosting model per horizon using the frozen
   53 T4 CBR/calendar features plus 32 fixed early-market state features.

The market state includes CNY and USD mean/last basis to current CBR, overnight
gap, open-to-cutoff and recent returns, range, realised volatility, slope,
range position, completed-candle count, age/missingness and four CNY/USD cross
features. Model parameters are inherited unchanged from T4. Fits start at the
externally fixed 2022-02-24 regime boundary, use only targets fully mature before
quarterly origin minus a two-day embargo, and use the same half-life 730 days.
Each candidate is calibrated quarterly with the same post-2022 mature-only
Platt method.

Report Brier, log loss, ECE, AUC and average precision separately on 2024 and
the already-open 2025--2026 period. Do not select a clock or architecture from
2025--2026. Route by the latest clock actually reached; if no same-day completed
candle exists, retain T4 rather than pretending the market expert is fresh.
Availability itself may be inspected before outcomes: the protocol retains all
clocks as a schedule-regime diagnostic even if late-year coverage is zero.

Physically corrupt all candles at and after a future date/clock and verify the
earlier prefix exactly. Rebuild every persisted probability from source, verify
source hashes, and corrupt future targets/maturity to prove earlier predictions
are invariant. Historical exchange availability and the CBR receipt timestamp
remain research assumptions, not certified production events.
