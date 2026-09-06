# Temperature T7B frozen protocol — new market information after 18:30

Registered 2026-09-06 after invalidating T7 and before T7B metrics were computed.

The frozen AP49/AP50/AP51 base is a decision at 18:30 Moscow with a 20-minute
market-feed delay. Historical CBR receipt before that decision remains a calendar
assumption. T7B tests only genuinely later query clocks 19:00 and 20:00.

At the 18:30 base and each later clock, a candle is available only when its
recorded end plus 20 minutes is strictly earlier than the decision and its nominal
ten-minute end plus 20 minutes is no later than the decision. Compute the CNY/RUB
TOM log return from the latest eligible base candle to the latest eligible query
candle. Require the latter to be newer; otherwise set zero delta and a missing
flag. This avoids reusing the market interval already available to the base.

Keep the T7 model forms and fixed post-2022 causal training rules: per-horizon
logistic recalibration for h=3/5/10/20 and a Ridge future-only basis-point model,
with mature labels before quarterly origin minus two days. h=1 remains exactly
known from today's effective and newly announced CBR rates.

Report 2024 and already-open 2025--2026 metrics versus AP50/AP51 without clock or
weight selection. Corrupt future and post-cutoff candles, labels and maturities;
earlier outputs must remain unchanged. T7 is an invalid control and cannot serve
as a comparator. No bank execution claim is allowed.
