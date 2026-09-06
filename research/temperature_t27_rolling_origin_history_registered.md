# T27 preregistration: pre-2025 rolling-origin history for delayed calibration

Registered 2026-09-06 before computing T27 outputs. T26 found that its 2024-only
screen could not distinguish a 250-publication rolling window from expanding;
the strong w250 result appeared only on the already opened 2025--2026 period.
T27 does not declare w250 from that result. It reconstructs an earlier causal
probability/error history and reruns the same frozen candidate selector.

## Historical reconstruction

- Use the existing T4 quarterly OOS `history_hist` h20 probabilities for 2023.
  T4 fits/calibrates each quarter from earlier mature data and is the exact
  source of the later `identity_early` premarket anchor.
- Build compact h20 scores separately for each 2023 quarter with the exact T24
  41-feature logistic specification. Each fit uses only rows before the quarter
  whose h20 outcome matured before origin minus the two-day embargo.
- Apply the already frozen 2024-H1 T25 residual standardization and alpha 0.40
  to those OOS 2023 scores. This is a retrospective initialization available
  when the T25 map is frozen; it is not claimed to have been served in 2023.
- Append those unique publication-event feedback rows to the unchanged T26
  2024--2026 calendar query. No weekend/hold row is added to feedback.

## Frozen candidates and selector

Reuse T26 without any new window or hyperparameter:

1. global windows 30, 60, 125 and 250 publication dates;
2. global expanding;
3. hierarchical 125-date global/currency correction;
4. global ridge 20, currency ridge 40, minimum 30 dates, logit clips 1.5/0.75.

At every query date, feedback is eligible only if its full h20 maturity is
strictly before query date minus two calendar days. Select on mature 2024-H2
with the exact T26 gate: lower Brier/log-loss than T25, ECE delta <= +0.005 and
AUC delta >= -0.005; choose lowest Brier with the frozen priority above. If no
candidate passes, retain T25.

## Evaluation and interpretation

Evaluate the pre-2025 selected model once on opened 2025--2026, with the same
aggregate/year/currency and paired 20/50-date gates as T26. Report T26-w250 as
an explicitly known diagnostic, not an independent comparator.

Even if T27 selects w250 and passes retrospective gates, 2025--2026 is not a
fresh holdout because T26 already exposed its performance. The strongest valid
claim would be: "the model could be selected without 2025--2026 after adding
causal 2023 history". Production promotion still requires prospective shadow.

## Required audits

- exact T4 alignment and finite 2023 OOS probabilities;
- every quarterly compact fit ends before query quarter with mature labels and
  embargo; no 2024+ label enters historical reconstruction;
- unique `(publication_date, currency)` feedback keys;
- each update's latest feedback maturity precedes its query cutoff;
- selector uses 2024-H2 only and is invariant to 2025--2026 target corruption;
- future-prefix corruption, source/publication time, bounded probabilities,
  selected-column reconstruction and paired interval grid.
