# Temperature T3 frozen protocol — post-2022 bridge calibration

Registered 2026-09-06 before T3 metrics were computed.

T2 showed that the raw causal market percentile between 15:30 and receipt keeps
useful ordering in 2025--2026, while a calibrator trained across the old monetary
regime erases it. T3 changes calibration only. The raw score, query clocks,
targets, candle rules and physical-causality contract remain frozen exactly as
in T2.

For every T2 candidate and h=1/3/5/10/20, fit the existing quarterly Platt
calibrator using only mature rows dated on or after the fixed structural boundary
2022-02-24 and strictly before the quarterly origin, with the existing two-day
embargo and 730-day half-life. This date is an external event boundary already
declared throughout the project; it is not chosen from T2 outcome scores. Fit no
clock-specific weights, thresholds, signs or nonlinear alternatives.

Report Brier score, log loss, ECE, AUC and average precision for 2024 and the
already-open 2025--2026 period against both the raw causal rank and the rolling
train prior. Preserve physical future-candle corruption evidence from T2 and add
a label/maturity corruption audit proving that future outcomes cannot change
earlier calibrated probabilities.

T3 is a retrospective methodological repair, not an untouched validation claim.
No tomorrow CBR rate and no uncertified historical receipt timestamp may be used.
