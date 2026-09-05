# Active research checkpoint: already published next CBR fixing

Updated 2026-09-06, AP1 complete. The user made after-publication research the
primary indefinite task. The hourly heartbeat `automation` was DELETED at the
user's explicit request. An ACTIVE TARGET drives continuous work in the same
thread and ivan-experiments. Do not restart round7 as the main task, wait for a
schedule, stop at another audit, or mark the goal complete after AP1.

## Read first

- after_publication_tz_decision.md: original authenticated case page rechecked;
  latest published reference is the conservative case-facing interpretation.
  Public CBR evaluation is permitted with an assumption about execution prices;
  a bank quote archive is NOT a prerequisite for the case's experiments.
- after_publication_protocol.md: target conventions, gates, selection rules.
- after_publication_ap1_registered.md: frozen AP1 packet; do not rewrite after
  seeing results. Data/protocol hashes saved in AP1 metadata.
- publication_applicability/report-source.md: Q&A/CBR/bank evidence. This older
  report predates the direct-page recheck; its bank-quote requirements apply to
  real monetary savings, not permission to evaluate the public CBR series.
- results/research/publication_applicability/calendar_alignment_summary.json:
  old2.459 replay and147/732 calendar-inconsistent signals. Preserve old artifacts.

## Completed AP0/AP1

AP0 added research/after_publication_clock.py with explicit received_at versus
effective_date, timezone-aware snapshots, current versus latest/next announced
fields, and mandatory evidence labels. Inferred18:00 timestamps are explicitly
calendar_assumed, never observed. New tests include the real TJS Saturday
failure pattern, UTC equivalence, just-before release, future corruption,
asynchronous/revised records and missing information. All9 targeted tests passed.
Verification saved in
results/research/after_publication/ap0_verification.json.

AP1 reconstructs 19,326 own-currency announcement events with 111 available-prefix
features and independent peer cutoffs. All times are CALENDAR-ASSUMED: effective
date minus one calendar day at assumed 18:00 MSK. No claim of verified historical
timestamps. Signals occur only on new own-currency announcements; no all-day or
weekend fallback evaluation yet. h is next observations; baseline uses identical
event dates. There is no live bank quote panel.

56 policies evaluated, 28 per target convention. Six model families plus simple
rules and stale controls. Annual training from 2016 with 7-year or 3-year windows;
all training labels matured to h20 plus two-calendar-day embargo. Selection uses
2017-2020 and 2022-2023 with all outcomes resolved before 2024. Selection JSON
written before later evaluation. 2024-2026 is already-viewed RETROSPECTIVE data,
not a new holdout. Frozen selector's effective h1 ceiling is a known limitation.

All 127 repository tests passed, including 9 clock and 6 AP1 tests. Model scores,
signals, outcomes, selections, training logs, year/currency breakdowns, paired
1000 date-block bootstrap and benefit/clustering audits are preserved in
results/research/after_publication/ap1/. Code: after_publication_panel.py,
after_publication_ap1.py and after_publication_ap1_audit.py. Run as modules with
PYTHONPATH=. .venv/bin/python -m research.after_publication_ap1, then the audit.

## Results to preserve, not silently reinterpret

Early-selected simple policy change_z_r25: known change / previous 20-return
volatility, above the 75th percentile of strictly previous 250 scores.

| Convention | Adjusted lift h1/3/5/10/20 | h5 cadence |
|---|---|---|
| Publication reference, all future steps unknown | 1.139 / 1.109 / 1.060 / 0.976 / 0.939 | 1.30 |
| Effective reference, first future step known | 1.917 / 2.274 / 2.265 / 2.227 / 2.220 | 1.30 |

Publication h5: 869 signals / 3260 rows, decisions 2024-01-09 through 2026-08-25;
hit .318757, symmetric benefit -36.942 bp. Stable >=1.3 / positive benefit NOT
achieved. Early selector picks this simple rule; HistGB r25's later h5 1.152510
cannot be promoted merely because it is retrospectively higher.

Effective h5: 869 / 3265, 2024-01-09 through 2026-08-26; hit .678941,
adjusted 2.264617 (pooled 2.309108), CI [1.992628, 2.571390]. By year
2.123 / 2.721 / 2.110; currency range 2.212-2.311. All-h symmetric-benefit CIs
positive; h5 +53.945 bp [40.620,70.471], future-only +121.692 bp. Max gap
43 calendar days, max 4 signals/week and one empty full month in some currencies.

Effective sign_cd3: h5 1.953315, cadence 1.43, max gap18 days, max2/week,
no empty full month. Effective early overall selector quantile7y_r25: h5
2.139092, cadence 1.52; delta versus simple2.265 CI [-.301761,.052598], no
ML superiority. z rule versus sign h5 paired delta CI [.142503,.477952], but
different allowable cadence means this is NOT an isolated matched-rate feature
ablation. Stale20 z control h5 1.075876.

Old after-publication 2.459 remains UNCONFIRMED: 147/732 old signals use a later
announcement under the date rule, mostly Saturdays. Preserve old artifacts;
the rebuilt panel does not rehabilitate that number or certify 18:00 timing.

New four-page PDF: output/pdf/ivan_after_publication_ap1.pdf; source manuscript
after_publication_ap1_report.md. README and EXPERIMENTS_SUMMARY now distinguish
these scenarios and keep the older 15:30 result in a historical section.

## Next bounded stage: AP2, actual experiments

Initial archive inspection (no AP2 target scores evaluated): CNY has 74,442
candles, including 7,102 beginning during 17:00 and 7,091 during 18:00, so late
intraday slicing is feasible. USD archive has 46,547 rows and claims observations
through 2026-09-03; verify source history/market availability before relying on
it, rather than presuming either validity or unavailability. Existing CNY/USD
archive lacks per-instrument FACEVALUE metadata; verify normalization as well.

1. Inspect whether saved CNY/USD 10-minute archive and round7 direct-pair data
   cover post-fixing hours. Data: data/moex_spot_fx_10min_2022_2026.json and
   data/moex_direct_pairs/*.json. Respect SHA manifests, nominal units, source
   timestamps and full nominal candle completion, not merely last trade time.
2. Freeze a compact after-publication experiment before seeing scores: market
   movement after the fixing window relative to the NEW announced CBR reference,
   with fixed cutoff justified in protocol (18:00 is not guaranteed publication
   time). Start with simple common/local basis and mature-label calibrations,
   then local anchor + global residual or alternative targets. Respect missing
   sessions and currency availability; no outcome-based drops.
3. Main emphasis is publication-reference lift AND positive symmetric benefit.
   Use chronological early selection and label opened later years retrospective;
   keep same-information controls and matched target/date denominators.
4. Separately improve conditional effective-reference cadence: causal cooldown,
   sequential weekly budget/threshold adaptations from past data only. No
   post-hoc top days within a week. Show Pareto tradeoff with sign_cd3.

Record failures too; preserve frozen packets. Update PDF at meaningful completed
stages. Checked commit/push only to ivan-experiments under standing authorization.
No main/force-push, bank transfers, client notifications or bank contact.
