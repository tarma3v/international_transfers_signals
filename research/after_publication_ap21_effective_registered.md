# AP21-E frozen protocol: separate high-quality primary and cadence expert

Registered 2026-09-06 before computing AP21 signals or scorecards. Diagnostics
show that AP13 rolling/local ExtraTrees have the strongest later minimum lift
but miss the required one-signal-per-week floor and leave empty months. AP17
meets cadence by using one score for both primary and pace decisions. AP21 tests
a prespecified dual-score state machine: one expert chooses high-quality primary
days; another is consulted only while the trailing rate is below one/week.

Inputs are frozen quarterly-OOS scores only: AP13 rolling-730-day ExtraTrees,
AP13 partially pooled local ExtraTrees, AP12 expanding ExtraTrees, AP18 fixed
expanding/recent blend, AP19 CatBoost mean-survival utility, and AP12 frozen
known70/hazard30 reserve. No model is refit and no future outcome enters routing.

Six fresh pairings are fixed:

1. rolling primary + AP18 pace;
2. rolling primary + CatBoost-utility pace;
3. rolling primary + AP12-expanding pace;
4. local primary + AP18 pace;
5. local primary + CatBoost-utility pace;
6. 75/25 rolling/local raw primary + 50/50 AP18/CatBoost raw pace.

For each score stream, compute a past-250 per-currency rank with warmup 40.
Issue primary when its rank is above .70. After 84 days, and only while the
prior-365 issued rate is below one/week, issue a pace signal when the separate
pace rank is above .55 and frozen reserve rank above .70. From day 24, reserve
rank above .70 fills an otherwise empty currency-month. Known-down veto and
maximum two signals per ISO week dominate. These are exactly AP17 thresholds;
only the identity of primary and pace experts is separated.

Use the same today-effective CBR target, known first next fixing, h3/h5/h10/h20
selection, early-2023 selection and opened-2024-2026 diagnostic. Early gates:
minimum lift >=1.3, each-currency rate 1..2/week, max two/week, no empty complete
month, positive symmetric-benefit CI and future benefit >=80% of simple sign.
Rank passers by minimum unknown-h lift, mean lift and gap.

Audit exact expert identity, state reconstruction, reason thresholds, trailing
rate, first-in-month rescue, future-score prefix invariance, and paired block
bootstrap against AP17/AP18/AP20 and the pure rolling/local diagnostics. Stretch
target: minimum unknown-h lift above 2.4 at required cadence. Historical receipt
time remains calendar-assumed, bank execution unvalidated, and the later period
is not a fresh holdout.
