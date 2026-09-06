# Temperature T11 frozen protocol — causal regime gate for benefit magnitude

Registered 2026-09-06 before T11 metrics were computed.

T6 and T10 Ridge magnitude heads improved the already-open 2025--2026 period
but were worse than their simple prior on the 2024 screen. T11 implements the
previously proposed error-regime idea without looking ahead: at each quarterly
origin, decide how much to trust the OOS model using only earlier predictions
whose future-only target is fully mature before origin minus a two-day embargo.

For every state and h=1/3/5/10/20 independently, score the fixed convex weights
0, .25, .5, .75 and 1 on the last 730 calendar days of mature prior OOS rows.
The loss is half-life-730 weighted MAE; ties prefer the smaller model weight.
Require at least 500 finite training rows, otherwise use weight zero. Hold the
chosen weight fixed for the entire next quarter. Prediction is
`prior + weight * (model - prior)`.

States are premarket, the T10 10:00 availability route, T6 cutoffs 10:30 through
15:30 and T6 updates 16:30/17:30. At 10:00 the model leg uses the T10 market
Ridge when a completed CNY candle exists and the T6 premarket Ridge otherwise;
the prior leg is always the matching T6 premarket prior.

Report model/prior/adaptive metrics on 2024 and open 2025--2026, quarterly
weights and 20/50-date paired bootstrap MAE deltas versus both legs. Rebuild all
persisted arrays exactly and corrupt future targets/maturity; no earlier weight
or prediction may change. Do not choose window, weight grid or state from the
opened scorecard. This experiment changes expected bps only, never the push or
probability temperature.
