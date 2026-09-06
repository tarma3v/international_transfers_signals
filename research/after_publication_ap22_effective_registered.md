# AP22-E frozen protocol: causal rolling/local consensus primary

Registered 2026-09-06 before AP22 scores or policies are computed. AP21 showed
that rolling ExtraTrees is a strong primary expert and that a separate cadence
expert can cross the strict point target. Local ExtraTrees is stronger in some
years/currencies but less stable. AP22 tests fixed, label-free causal consensus
geometries for the primary stream while preserving the AP21 state machine.

Compute past-250 per-currency ranks with warmup 40 for frozen quarterly-OOS
rolling and local scores. Five primary scores are fixed: rolling/local weighted
means 75/25, 50/50 and 25/75, rank minimum, and rank geometric mean. Missing one
rank falls back to the other. Test CatBoost-utility as pace expert for all five;
for the 75/25, 50/50 and minimum primaries, also test AP12 expanding ExtraTrees
as pace expert. This creates eight predeclared pairings without fitted weights.

Every pairing uses the exact AP21/AP17 thresholds: primary causal rank >.70;
after 84 days and only while trailing-365 rate <1/week, pace rank >.55 and
reserve rank >.70; day>=24 reserve rescue for an empty currency-month; known
down veto and max2/ISO-week. The inner rolling/local ranks are past-only scale
normalization; the outer rank is the frozen product controller.

Evaluation, selection gates and caveats are unchanged: today-effective CBR,
known first next fixing, h3/h5/h10/h20, early-2023 selection, opened-2024-2026
diagnostic, 1..2 per currency-week, no empty complete months, positive benefit
CI and future-benefit floor. Historical receipt times are calendar-assumed,
bank execution is unvalidated, and later evidence is not a fresh holdout.

Audit both source ranks, all eight pairing states, threshold reasons, future
expert-score corruption and paired 20/50-date bootstrap. Stretch target remains
minimum unknown-h lift >2.4 at valid cadence; practical success should also
improve h5/future benefit without a large short-horizon sacrifice.
