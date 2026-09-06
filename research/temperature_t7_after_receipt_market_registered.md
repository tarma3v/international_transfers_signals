# Temperature T7 frozen protocol — market updates after CBR receipt

Registered 2026-09-06 before T7 metrics were computed.

T7 tests whether completed market candles improve the AP50/AP51 after-receipt
widget. Historical receipt remains the explicit calendar assumption 18:00
Moscow, never a certified release timestamp. Fixed query clocks are 18:30 and
20:00. Production must trigger from the actual received CBR record.

At each clock compute the CNY/RUB TOM log return from the last completed candle
strictly before assumed receipt to the last completed candle strictly before the
query time. Admit only same-day candles with nominal ten-minute completion no
later than the relevant boundary. If none arrived, record missing and zero delta.

For h=3/5/10/20, fit a quarterly logistic recalibrator with fixed features:
logit(AP49 raw head), market delta, absolute delta, availability and five
currency one-hots. Fit a quarterly future-only Ridge magnitude model using the
frozen AP49 multihorizon feature vector, AP50 calibrated probability and the same
market fields. Use the fixed 2022-02-24 minimum date, 730-day half-life, mature
labels before origin minus two days, C=1, Ridge alpha=10 and train-only target
winsorisation. Compare directly with AP50 probabilities and AP51 expected bps.

For h=1 no model is needed after receipt: whether today's effective CBR is no
higher than the announced next effective rate, and their basis-point difference,
are both already known. Verify the formula exactly against the stored h1 target.

Report 2024 and already-open 2025--2026 metrics without selecting a clock or
weight. Corrupt future candles, labels, maturities and market deltas; earlier
outputs must remain unchanged. No bank execution or customer quote is implied.
