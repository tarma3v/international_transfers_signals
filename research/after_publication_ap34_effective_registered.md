# AP34-E frozen protocol: announced-anchor decayed residual survival

Registered 2026-09-06 before AP34 model fits, signals or scorecards. AP33 is
frozen and its 84-day warmup, Thursday/Friday rule and 10-day silence are not
tuned. AP34 tests one genuinely new classical predictor and one final policy.

The continuous target is the unknown h20 floor after tomorrow's already
announced fixing, in basis points relative to that fixing:

`residual_floor20 = effective_floor20 - known_change`.

For every quarterly OOS origin, use only rows with publication-h20 fully mature
before origin minus two days, finite target and announced>=current eligibility.
Fit one standardized Ridge(alpha=100) on the frozen AP13 compact feature set.
Winsorize its training target at the train-only 1%/99% quantiles. Estimate a
per-currency residual bias with calendar half-life 365 days and shrink it toward
zero by n/(n+150). Global residual-distribution weights have half-life730 days;
local residual weights have half-life365 days. Convert the announced anchor plus
predicted residual floor into survival probability using a weighted empirical
train-only error distribution. Blend local/global survival probabilities by the
same n/(n+150) local weight. There is no label access inside the query quarter.

Pass this sole new probability score as the pace expert through the exact frozen
AP21 dual-paced-month controller, with the frozen rolling primary and reserve.
This produces the new core decision stream. Apply the exact frozen AP33 calendar
router to that stream and frozen AP23 decisions. No alternative alpha, half-life,
shrinkage, feature set, score blend or policy threshold is evaluated.

The sole fresh candidate is selected on early2023 with the unchanged joint gates;
frozen AP33 is the registered fallback. Opened2024-2026 remains a retrospective
diagnostic. Audit all17 fits, maturity masks, target clipping, local states,
empirical probabilities, nested controllers, future-feature/label corruption,
future-source prefix invariance and paired20/50-date uncertainty. Acceptance is
min h3/h5/h10/h20 lift>2.4,min currency rate>=1,zero empty complete months and
max2/week. H1 remains known validity only; receipts are calendar-assumed and bank
execution is unvalidated.
