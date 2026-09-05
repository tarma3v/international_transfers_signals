# Round 5 adaptation addendum

Frozen after the first new-family gate failed and before running the candidates
below. The first packet showed a sign/regime reversal: self-normalized models
trained mainly on pre-2022 data failed in 2022--2023 but recovered in every
2024--2026 year. This addendum tests whether stale fitting, rather than the
representation itself, is the bottleneck.

## Operational question

For a product deployed in the current post-24.02.2022 regime, can a model using
only resolved past labels achieve lift 1.30 at 1--2 alerts per currency-week?
The next CBR publication remains forbidden.

## Chronology

- All candidate definitions below are frozen before their results are read.
- Prequential scores start in 2023. At each calendar quarter, the model is
  refitted and scores only later dates in that quarter.
- A training row is admitted only when its full `h=5` outcome resolved before
  the refit timestamp.
- Candidate/policy screen: 2024.
- Confirmation gate: 2025 with the 2024-selected policy unchanged.
- Last chronological audit: available part of 2026 with the policy unchanged.
- These years were seen in earlier research, so the exercise remains
  retrospective at the research-program level even though every prediction is
  causally out of sample.

## Frozen candidates

- quarterly HistGradientBoosting, ExtraTrees and logistic regression trained
  only after 24.02.2022;
- quarterly HistGradientBoosting with rolling 2- and 3-year training windows;
- quarterly HistGradientBoosting with 1-year exponential half-life;
- fixed rank ensembles with the multiscale anchor (25% and 50% anchor weight);
- consensus of reset HistGradientBoosting and reset ExtraTrees.

All candidates use the target-free self-normalized trajectory summaries from
the first round-five packet. Policies are screened at target rates
18/20/22/25/30/35/40%, with fixed or rolling past-score thresholds. A candidate
passes only if 2025 lift and macro-year lift are at least 1.30, frequency is
0.90--2.10 overall and at least 0.65 in every currency, and future benefit is
positive. The 2026 result cannot rescue a failed 2025 gate.

## Error-driven calibration addendum

Frozen after reading the first adaptation packet and before running the
following candidates. The nominal winner changed model scale at a quarterly
refit: its 120-publication rolling cutoff then produced zero alerts in both
2025Q1 and 2025Q2 and a burst in Q3. This is causal, but it is not an acceptable
interpretation of a stable one-to-two-alert product.

The second adaptation packet converts every new model score into a percentile
against that same fitted model's most recent 60, 120 or 250 past feature rows,
separately by currency. The reference rows all precede the quarter being
scored; their targets are not used for percentile calibration. This makes
scores from successive quarterly refits comparable. HistGradientBoosting and
ExtraTrees variants, their consensus, and fixed 25/50% anchor blends are tested.

Policies are selected on 2024 from fixed thresholds and causal trailing
20/40/60/120/250-publication thresholds. In addition to the original gate, an
eligible policy must emit at least 0.30 and at most 2.50 alerts per
currency-week in every evaluated calendar quarter with data. Architecture and
policy remain unchanged in 2025 and 2026. The experiment is multiplicity-
audited and still labelled retrospective because these dates are not pristine.

## Cross-fitted refit calibration addendum

Frozen after the same-model percentile packet failed the cadence gate. Its
reference rows were historical, but they had also participated in model fit;
tree scores were therefore more extreme on the reference than on the next
quarter. The cross-fitted packet removes the preceding one or two full
quarters from training, fits only on older resolved labels, scores the held-out
past quarter(s) and the current quarter with the identical fitted model, and
uses the held-out past scores as the per-currency percentile reference.

HistGradientBoosting and ExtraTrees, their 50/50 consensus, and fixed anchor
blends at 25/50% are screened. Candidate rates and causal alert policies stay
the same as in the preceding packet, including the quarterly 0.30--2.50 cadence
gate. No definition is changed after reading 2025 or 2026 results.
