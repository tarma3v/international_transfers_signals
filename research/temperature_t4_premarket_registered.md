# Temperature T4 frozen protocol — history-only premarket expert

Registered 2026-09-06 before T4 metrics were computed.

T4 closes the overnight-to-first-candle gap using only the latest effective CBR
history available before the requested day. It predicts all official horizons
h=1/3/5/10/20 and does not use MOEX candles or tomorrow's unpublished CBR rate.

Use one pooled quarterly HistGradientBoosting classifier per horizon with a fixed
compact feature list: causal level/range, returns, volatility, gaps, cyclic
calendar, holiday/payday flags, peer/reference returns and currency one-hots.
Train only rows dated on or after the externally fixed 2022-02-24 regime boundary
whose complete target horizon matured strictly before quarterly origin minus a
two-day embargo. Fixed parameters: 220 iterations, learning rate .035, nine leaf
nodes, minimum leaf 42, L2 15, random seed 20260904. No model, feature or parameter
selection is allowed from T4 scores.

As a transparent control, use one minus the same-currency causal percentile of
the frozen multiscale range anchor (.5 range90 + .3 range30 + .2 range180).
Calibrate both the model and control quarterly with the T3 post-2022 Platt method.
Report Brier, log loss, ECE, AUC and average precision on 2024 and the already-open
2025--2026 period. T4 is retrospective research, not a fresh holdout claim.

Physically corrupt future CBR values/features and future target/maturity arrays;
every earlier raw and calibrated prediction must remain unchanged. Historical
`received_at` is not reconstructed here: production must stamp the actual CBR
record used, while replay labels this as a latest-known daily snapshot.
