# AP30-E frozen protocol: rank-space specialist/competence interpolation

Registered 2026-09-06 before AP30 signals or scorecards. AP26 is the strongest
accuracy core but misses strict cadence; AP23 has strict cadence with slightly
lower accuracy. AP27/AP29 show that adding a separate backstop can consume weekly
capacity with weak emergency points. AP30 tests one smooth interpolation instead.

For each currency and row, compute causal prior250 ranks with warmup40 for the
AP26 y20-shrink200 pace score and the AP23 soft730 mature-competence pace score.
The sole new pace score is fixed before evaluation as
0.75 * AP26_rank + 0.25 * AP23_rank. No alternative weight, raw-score blend,
backstop or threshold is tested. The 75/25 weight deliberately keeps the
specialist as the dominant expert while using competence only as shrinkage.

Feed this score to the exact AP21 dual pace/month policy: rolling ExtraTrees
primary; after84 days, pace prior250 rank>.55 only while trailing365 currency
rate<1 and reserve rank>.70; month24 rescue, known-down veto and max2/ISO-week.
The inner expert ranks and the final pace rank are all strictly previous-row and
outcome-free. AP21, AP23, AP26, AP27 strict/selected, AP29, AP18 and AP17 are
frozen controls.

The only fresh AP30 candidate is selected on early2023 with the unchanged joint
gates. Opened2024-2026 is diagnostic, never a fresh holdout claim. Audit both
inner ranks, exact blend, outer policy ranks/state/reasons, future-score prefix
invariance and paired20/50-date uncertainty. Acceptance is min h3/h5/h10/h20
lift>2.4, min currency rate>=1, zero empty complete months and max2/week. H1 is
known after receipt and excluded; timestamps remain calendar-assumed and bank
execution is unvalidated.
