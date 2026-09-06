# Transfer-temperature time router

Frozen design note, 2026-09-06. This combines existing pre-publication market
experts with the new after-publication CBR experts without pretending that one
snapshot model is valid at every clock time.

## Core rule

Route by **information state**, not by a hard-coded claim that publication
always happens at 18:00. The production trigger is receipt of a new CBR record
with its real `received_at`. The historical after-publication replay currently
uses 18:30 only as a clearly labelled calendar assumption. T18 enforces that
distinction in code: the default case query cannot activate a receipt-dependent
row without a caller-supplied verified event.

At a query time `as_of`, select the latest expert whose entire input prefix is
available. A newer information event may cause an honest jump in temperature;
the UI should label the cause rather than artificially smoothing through it.

## Proposed daily lifecycle

| Information phase | Score source | What is already supported | Remaining gap |
|---|---|---|---|
| Overnight to first live slice | latest CBR/history-only score | T4 causal history calibration and stale metadata | discrimination remains weaker than intraday |
| 09:00--10:30 | horizon-aware perpetual/spot prefix or history fallback | T13/T14 and T10; only physically completed candles | h5/10/20 often remain history-only |
| 10:30--15:30 | latest completed market cutoff expert | T5 calibration at every fixed cutoff | fresh independent confirmation remains unavailable |
| 15:30 until actual CBR receipt | anchor plus post-window market correction | T3/T12 at 16:30/17:30; T17 removes missing-candle clocks | actual receipt event must come from ingestion |
| After actual CBR receipt | calibrated AP50/AP51 horizon probabilities | T18 verified-event gate; tomorrow's announced CBR becomes a valid feature | historical receipt timestamps remain uncertified |
| After receipt with new market candles | receipt anchor plus completed-candle correction | T7B at 19:00/20:00; T15/T16 benefit updates to 23:00 | later probability candidates did not pass screen gates |
| Closed market/weekend | latest valid score with increasing age | T17 removes fictitious updates; stale state and safe copy tested | user-local display and live bank quote remain pilot work |

## What “the 15:30 model works from 11:00 to 18:00” can mean

It cannot mean running the literal 15:30 feature vector at 11:00: candles from
11:00--15:30 would be future leakage. It can mean using the **same model family**
on a truncated prefix. The repository already constructed and corruption-tested
eight such cutoffs. The 2024 screen gave minimum official lift between 1.50 and
1.58 for every cutoff; 15:20 was selected there. On 2025--2026, 15:20 and 15:30
were close, although strict paired non-inferiority failed only at the h20 margin.

After 15:30, holding the last score is valid but loses new information. The next
registered experiment should compare this hold baseline with a mechanical
correction from completed post-15:30 CNY/USD returns. Because the CBR fixing
methodology window has closed, those returns do not update the already computed
tomorrow fixing; they update the outlook beyond that fixing and therefore the
user's current transfer attractiveness.

## Combination model

Keep three values separate before a final calibrator:

1. `market_anchor`: the last causally available cutoff score/probability;
2. `published_anchor`: AP49 probability, present only after actual receipt;
3. `new_market_delta`: CNY/USD and, where reliable, direct-corridor movement
   since the anchor timestamp, using completed candles only.

Fit a quarterly mature-only logistic calibrator for `P(now best at h)` and a
separate robust regressor for expected future-only basis points. Inputs include
the anchors, market delta, their disagreement, source age, missing flags,
currency and phase. Before publication, `published_anchor` is missing; after
publication it becomes the primary anchor. Missingness is an explicit feature,
not a zero that pretends a quote exists.

Do not hand-pick blend weights from 2025--2026. Either freeze a mechanical
anchor-plus-delta coefficient of one for the first diagnostic or learn weights
quarterly from mature earlier dates. All calibration folds are grouped by date,
and labels must mature before the fit origin plus embargo.

## Proposed fixed query grid for historical validation

- 09:30: start-of-day/stale-CBR control;
- 10:30, 11:30, 12:30, 13:30, 14:30, 15:00, 15:20, 15:30: existing market prefixes;
- 16:30 and 17:30: new post-fixing-window corrections;
- actual receipt event in production; calendar-assumed 18:30 only in the
  historical research branch;
- 18:30 and 20:00: after-publication score plus fresh-market correction;
- market close and next morning: held score with explicit increasing staleness.

The grid is for evaluation, not for forcing the user to arrive at those exact
times. `score_as_of` uses the latest admissible snapshot between grid points.

## Push versus widget

The push policy remains sparse and can use a high temperature threshold plus
cadence/cap logic. The widget returns temperature on every query, including
neutral or unfavourable values. A user arriving after a push sees the latest
recomputed temperature, not the historical push score; the payload may also say
whether temperature rose, held, or fell since the push and which new source
caused the change.

## Completed sequence and next experiment

1. T3--T7B completed causal phase calibration, pre-receipt benefit and later
   market corrections.
2. T8B--T17 assembled latest-valid routing, early/perpetual coverage, separate
   probability/benefit provenance and physical spot availability.
3. T18 now gates every same-day receipt-dependent output on an observed event.
4. T19 completed the unified 20-clock/currency/year audit over 193,400 queries.
   It confirms strong h1/h3 and daytime h5 states, but no h20 state passes the
   strict paired gate and local currency-year calibration drifts materially.
5. T20 preregistered and tested one-shot hierarchical calibration on mature
   pre-2025 data. It passes 0/120 state gates and is not adopted.
6. T21 tested a genuinely new cross-horizon h20 head. Primary HGB+Platt passes
   0/40 gates and is rejected. A logistic control improves AUC in 40/40 states
   (mean 0.576 to 0.682) but catastrophically shifts probability calibration,
   so it remains a rank-only research feature.
7. T22 constrained a small rank correction using only disjoint pre-2025
   calibration. It is harmful as an all-day model, but all six post-receipt
   replay states pass: mean h20 AUC 0.534 to 0.671 and Brier 0.12058 to 0.11379.
   The correction is therefore a receipt-gated shadow challenger only.
8. T23 tested monthly delayed calibration using only matured past outcomes. It
   passes 0/40 states and worsens mean AUC/Brier; short trailing screens select
   unstable mappings. Do not update the receipt mapping frequently.
9. T24 found a new history-only compact ranker: pre-2025 selection AUC 0.742 and
   opened 2025--2026 AUC 0.702 versus 0.374 early identity. It was not selected
   because its selection-period probability calibration failed; keep it as a
   rank-only shadow, not a temperature.
10. T25 tested that predeclared map. Selection-2024 chose a 0.40 residual-logit
    blend; open AUC/Brier improve 0.374→0.563 and 0.12429→0.11825, but both
    Brier block intervals cross zero. It remains a premarket shadow.
11. Next: keep T22 receipt-only and T25 premarket shadows frozen. If attempted,
    adapt only the probability intercept slowly from already mature outcomes;
    do not select a post-2022 switch, weight or clock on opened 2025--2026.
12. T26 tested that delayed intercept. No candidate passed the 2024-H2 joint
    screen; T25 remains primary. Open-period w250 is only a post-hoc hypothesis.
    The next selector needs earlier rolling-origin history or prospective data.
13. T27 supplied 247 OOS publication dates from 2023 and rejected w250 on the
    pre-2025 screen. T25 remains primary; a weak preregistered shrink toward
    w30 is the only remaining delayed-calibration direction supported there.
14. T28 nested Q3 selection and Q4 validation for weak w30 doses. Q4 rejected
    the selected beta on AUC. Daily intercept updates alter cross-day rank; a
    future test must hold level fixed over a coarser period.
15. T29 held the intercept fixed inside each month or quarter. Q3 selected a
    monthly w30 full correction, but disjoint Q4 lost 0.04486 AUC against the
    frozen T25 anchor. The candidate is rejected. Slow base-rate correction is
    not stable enough to promote without a prospective shadow period.
16. T30 trained rank only before 2022, mapped its level on fixed post-SVO 2022,
    screened on 2023 and validated on 2024. Nothing passed 2023, although the
    all-history family later reached open AUC 0.750. Keep `all_platt_b050` as a
    frozen prospective control only; a calendar switch learned from hindsight
    is not an admissible information-state router.
17. T31 mixed identity, all-history and recent2y experts with mature-only
    fixed-share Hedge updates. Nothing passed the pre-registered 2023 screen.
    The unselected `eta=2, gamma=0` line later reached open AUC 0.750 and Brier
    0.11881, showing a causal regime-adaptation mechanism but not a fresh
    winner. Keep it as a prospective shadow; do not change the runtime router.
18. T32 gave the same family a disjoint OOS Q4-2022 warm-up. It concentrated
    weight on recent2y and reduced 2023 screen AUC to 0.552 versus 0.588
    identity. Reject blind recent-data warm starts. The next adapter must hold
    cross-date rank stable or route on an observable information state.
