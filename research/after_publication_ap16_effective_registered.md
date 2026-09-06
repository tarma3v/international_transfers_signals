# AP16-E frozen protocol: causal deficit pacing

Registered 2026-09-06 before any AP16 scorecard. AP15 proved that a single
month rescue removes empty months but does not repair per-currency rate. AP16
tests a different controller class: issue an additional high-score signal only
when that currency is causally behind its one-per-week pace. AP15 late outcomes
set the question, not AP16 parameters or winner.

The target remains TODAY-EFFECTIVE CBR after actual receipt of tomorrow's
announced fixing. Historical receipt is CALENDAR-ASSUMED, h1 is already known,
and bank execution is not validated.

## Frozen support

Reuse5,755 rows,AP12 ExtraTrees quarterly OOS score,known70/hazard30 reserve,
known-down veto,publication-h20 maturity cap,2-day embargo,early2023 selection
and opened2024-2026 diagnostic. Primary always means prior250 per-currency rank
>.70,warmup40,max2/ISO-week. Unknown selection horizons are h3/5/10/h20.

## Five causal controllers

All pacing counts exclude the current decision. Start pacing after84 observed
calendar days. A365-day rate uses selected decisions in [day-365,day) divided
by actually observed weeks. A cumulative rate uses all decisions since the
currency's first row divided by elapsed weeks. No outcome label is used.

1. `pace365_p60`: if trailing365 decision rate<1.0/week, allow primary-score
   rank>.60; otherwise retain top30 only.
2. `pace365_p55_r70`: when behind, require primary rank>.55 and reserve rank>.70.
3. `paceall_p60`: use cumulative rather than365-day deficit and rank>.60.
4. `pace365_p60_month24`: controller1 plus one reserve-rank>.70 rescue from
   day24 only when the currency-month still has no signal.
5. `adaptive105_month24`: reproduce AP14's182-day adaptive thresholds exactly:
   .65 when rate<.95,.675 when rate<1.05,.70 otherwise, plus the same empty-month
   rescue. Before56 days its threshold is.675.

Reason state distinguishes primary, pacing and month rescue. Eligibility veto
and weekly cap dominate every route. The controller cannot reserve a future day
or know whether a better signal will appear later in a week/month.

## Selection and reporting

Carry AP12 top30,AP14 near/adaptive/strict/honest-selected,AP15 selected,
AP13 rolling/local/reserve,AP10,AP11,AP1 and known-sign controls. Select five
fresh policies on early2023 using the existing gates: unknown-h minlift>=1.3,
all-currency rate1..2,positive symmetric CI,max2/week,future benefit>=.8 of
known-sign,zero empty months; rank by min unknown-h lift,mean lift,gap.

Later diagnostic success requires h5 per-currency rate>=1,zero empty months,
max2/week and min unknown-h lift>2.35; also report stricter all-h-scope rate.
Audit exact state,prefix corruption,pace calculations,reason counts,paired
20/50-date CIs and currency/year slices. The opened-period result cannot be
called independently validated.
