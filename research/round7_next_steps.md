# Research continuation checkpoint

**Superseded priority on2026-09-06:** user explicitly redirected the indefinite
loop to already-published tomorrow-CBR research and deleted the hourly automation.
An active target now drives continuation. Read
`after_publication_next_steps.md` and `after_publication_protocol.md` first.
The local-pair/strict-before-publication agenda below is retained as historical
context and control work, not the primary next experiment.

Updated 2026-09-05. Branch: ivan-experiments. User explicitly authorized ongoing
research, PDF summaries and pushing reviewable results to this branch only.
At this historical checkpoint the hourly heartbeat was active; it was deleted
on 06.09 and must not be recreated. Preserve results; never replace main or
force-push. Current continuation uses the active target documented above.

## Current decision

**2026-09-06 publication audit:** do not promote old after-publication 2.459.
`round4_research._publication_matrix` reads the next effective-date row without
an announcement-time gate. Calendar replay of `pub_extra_7y` reproduces 2.459
but flags 147/732 signals (20.08%) as requiring a later announcement, mostly
weekends. Need full causal reconstruction, not just dropping those test alerts.
The remaining alerts are not timestamp-certified either. Q&A Sep5 p7 says app
rates are real-time; old-CBR executable prices after publication are unproven.
Report: `output/pdf/ivan_cbr_after_publication_applicability.pdf`. This audit
does not refit or independently validate the 15:30 incumbent below.

Retain packet ED 15:30 availability_route. Official adjusted h5 lift 2.052908,
pooled 2.058770; h1/3/5/10/20 = 1.780/2.014/2.053/2.005/1.928, rate 1.1946.
Evaluable h5 dates 2025-01-10 through 2026-08-26 (2020 currency-date rows).
No unpublished tomorrow-CBR feature. This is retrospective, not a fresh holdout.

Round 7A collected 99,013 ten-minute CETS candles across TOM and TOD in
KZT/AMD/KGS/TJS/UZS. JSON hashes and FACEVALUE are verified on loading.
KZT and AMD have dense data; KGS and UZS are sparse. TJS has no eligible CETS
observations in 2025-2026. Missing volume is not volume=0 or a VWAP weight.

43 new direct/ML policies + incumbent: two direct bases, three gates, five
nonzero weights; four learned expert types with three blend weights; currency
specific grid. Purged 2024 selection chose per_currency_weights:
KZT last/any75, AMD mean/quality25, KGS last/quality50, TJS incumbent,
UZS mean/any50. Combined later adjusted h5 1.977997; WORSE than incumbent.
Mean horizon delta -0.091822, paired 95% CI [-0.154043;-0.029933].
Best global direct mixture selected on 2024: last_basis_soft_w025 -> 2.043321.
Never describe the per-currency 2024 selections as established optimal weights.

Round 7B: 20 residual-floor policies + incumbent. CNY basis anchor plus model
correction to next-five minimum log-price change. Global Ridge/Hist mean/Hist
quantile25/ExtraTrees, local Ridge; weights .1/.25/.5/1. Purged2024 selected
global_quantile_w10 -> adjusted h5 2.037815; no promotion. The delayed-local
control is close; no proven incremental local information.

Some unselected variants have higher late point h5 (e.g. 7A global_extra_w50
2.077167). Do not select them on the opened later data and present as honest
improvement. All results retained in results/research/round7.

## Widget

Quarterly logit calibration of frozen router score + currency, only mature OOF
labels since 2024. At 15:30 on 2025-2026: Brier .154734 versus raw rank .223518
and training constant prior .199852. Probabilities remain imperfectly calibrated;
see reliability bins. This is NOT an intraday validated widget yet.
24/7 display can show the latest valid snapshot and timestamp. Genuine fresh
scores throughout day require separate historical time-slice replay and
calibration, daily grouped splits, no choosing max score after seeing whole day.
Actual expected savings require executable bank/service quotes and fees.

## Next experiments (pre-register before running)

1. Local basis may contain persistent spreads; estimate deviations from a
   strictly past local median or robust regression on CNY, then predict residual.
   Select windows on earlier rolling folds; cap currency weights using evidence
   across several folds, not just 2024. Consider shrinkage to a common coefficient.
2. Bootstrap/resolved-loss reliability weighting: historical target outcomes
   available only after full horizon resolution; never use unrealized outcomes.
   Compare against frozen constant weights at the same cadence.
3. Check multiple source timing/quote conventions and overseas venues (e.g.
   KASE KZT) if open historical snapshots exist. Do not invent volume/spreads or
   conflate listed instrument with active trading. Prior-session daily activity
   is admissible; same-day full daily aggregates are future leakage at 15:30.
4. Add 10:30/12:00/14:00/15:30/17:00 snapshots for widget, separate pre/post-CBR
   publication branches. Use actual availability timestamps. Freeze push policy.
5. Full prospective shadow after the latest research cutoff; previous
   2025-2026 stays retrospective forever. Do not reuse opened dates as holdout.

## Reproduction

PYTHONPATH=. .venv/bin/python -m research.round7_direct_pairs_data (network)
PYTHONPATH=. .venv/bin/python -m research.round7_direct_pairs
PYTHONPATH=. .venv/bin/python -m research.round7_residual_floor
PYTHONPATH=. .venv/bin/python -m research.round7_audit
PYTHONPATH=. .venv/bin/python -m research.build_round7_report
PYTHONPATH=. .venv/bin/pytest -q

Report: output/pdf/ivan_direct_pairs_and_widget_report.pdf.
Verification: full suite 112 tests passed; the 11-page PDF was rendered and
visually reviewed, with no text outside page bounds or replacement characters.
No live trading, sending client messages, or executing bank transfers authorized.
