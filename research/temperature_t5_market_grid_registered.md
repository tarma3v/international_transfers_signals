# Temperature T5 frozen protocol — intraday market-grid calibration

Registered 2026-09-06 before T5 metrics were computed.

T5 turns the already corruption-tested cutoff frontier into continuous widget
probabilities. The eight fixed Moscow query clocks are 10:30, 11:30, 12:30,
13:30, 14:30, 15:00, 15:20 and 15:30. At every clock use only same-day CNY/RUB
TOM candles whose recorded end is strictly earlier than the clock, divided by
the current effective CBR CNY rate. Do not choose a clock or blend from T5
outcomes.

For each clock, convert the raw basis score into a same-currency causal rank
using the frozen 250-row history/minimum 20, then calibrate P(now is best) for
h=1/3/5/10/20 with the fixed T3 quarterly Platt method: training date on or
after 2022-02-24, mature horizon strictly before origin minus two days, 730-day
half-life, currency one-hots. No tomorrow CBR rate is available.

Report Brier, log loss, ECE, AUC and average precision for every clock/horizon on
2024 and the already-open 2025--2026 period. A physical future-candle corruption
audit must pass at every cutoff, and future score/label/maturity corruption must
leave all earlier probabilities unchanged. This is retrospective calibration
research and does not certify historical CBR receipt timestamps.
