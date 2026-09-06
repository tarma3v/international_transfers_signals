# AP37-E frozen protocol: mature precision gate for disagreement-tail fallback

Registered 2026-09-06 before AP37 late signals or scorecards. AP33 remains the
frozen leader. AP37 tests exactly one decision-tail modification; there is no
threshold, window, target, expert-subset or policy grid.

Use the frozen AP33 inputs: AP26 y20-shrink200 core and AP23 fallback. Use three
already audited causal OOS ranks from AP36: AP26 y20, AP34 residual survival and
AP35 distributional CatBoost. For every row define a support stratum as the
number of these three ranks at or above 0.70. The 0.70 boundary is the existing
top-30 decision boundary, not an AP37 fit.

The precision label is strict multi-horizon success
`min(y3, y5, y10, y20)`. Before each decision date use only eligible non-core
rows on earlier dates whose publication-reference h20 maturity is earlier than
the decision date minus two days. Estimates are expanding. Overall precision
shrinks to 0.50 with strength 40; global support-stratum precision shrinks to
overall precision with strength 40; same-currency support-stratum precision
shrinks to the global stratum with strength 40. No current, same-date, immature
or future label enters. The quality gate passes when all three ranks are finite
and local shrunk stratum precision is at least the causal overall precision.

Run one sequential calendar router from scratch. Preserve every eligible AP26
core decision. Preserve the exact AP33 warmup 84, trailing-365 rate below 1,
silence 10 days and max 2 per ISO week. AP23 late-week fallback on Thursday or
Friday is allowed only when the quality gate passes. AP23 silence fallback is
always preserved even when the gate fails. Rejected decisions are not written
to router state, so a later qualifying row may replace them causally.

Select the sole fresh policy on early 2023 with the unchanged joint gates;
frozen AP33 is the registered fallback. Opened 2024-2026 remains retrospective.
Audit all expert inputs, support strata, maturity masks, counts/precisions,
sequential rate/silence/reason/cap state, future expert/label/source corruption
prefix invariance and paired 20/50-date uncertainty.

Acceptance remains min h3/h5/h10/h20 lift above 2.4, min currency rate at least
1, zero empty complete months and max 2 signals per week. H1 is known validity
only. Historical receipt timestamps remain calendar-assumed and bank execution
is not validated.
