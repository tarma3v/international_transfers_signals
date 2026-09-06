# Transfer-temperature time router

Frozen design note, 2026-09-06. This combines existing pre-publication market
experts with the new after-publication CBR experts without pretending that one
snapshot model is valid at every clock time.

## Core rule

Route by **information state**, not by a hard-coded claim that publication
always happens at 18:00. The production trigger is receipt of a new CBR record
with its real `received_at`. The historical after-publication replay currently
uses 18:00 only as a clearly labelled calendar assumption.

At a query time `as_of`, select the latest expert whose entire input prefix is
available. A newer information event may cause an honest jump in temperature;
the UI should label the cause rather than artificially smoothing through it.

## Proposed daily lifecycle

| Information phase | Score source | What is already supported | Remaining gap |
|---|---|---|---|
| Overnight to first live slice | latest CBR/history-only score | daily CBR history and stale metadata | a dedicated start-of-day calibration is not yet built |
| 10:30--15:30 | latest completed market cutoff expert | causal cutoffs at 10:30, 11:30, 12:30, 13:30, 14:30, 15:00, 15:20, 15:30 | only 15:20/15:30 received full later comparison; each slice needs widget calibration |
| 15:30 until actual CBR receipt | 15:30 anchor plus post-window market correction | holding the 15:30 score is causal; post-15:30 CNY/USD candles exist in the archive | correction model at fixed later slices is not yet trained |
| After actual CBR receipt | calibrated AP49 horizon probabilities | tomorrow's announced CBR is now a valid feature; four causal OOS heads exist | probabilities need causal calibration; receipt timestamps are assumed historically |
| After receipt with new market candles | AP49 anchor plus since-receipt market correction | data archive contains later candles | combined event-time model and honest timestamp replay are not yet built |
| Closed market/weekend | latest valid CBR score, held with increasing age | explicit missing/stale behavior is possible | validate weekend text/bands and avoid fictitious interpolation |

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
- actual receipt event in production; calendar-assumed 18:00 only in the
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

## Immediate experiments

1. Finish causal calibration of AP49 probabilities and future-only benefit.
2. Build `score_as_of` with held-score/staleness semantics and tests.
3. Extend the CNY/USD candle feature builder to 16:30 and 17:30 and compare
   anchor hold versus anchor-plus-post-window-delta without using tomorrow CBR.
4. Under the calendar-assumed research branch, add 18:30 and 20:00 corrections
   on top of AP49. Keep this separate from timestamp-certified claims.
5. Only after those stages, train a single phase-aware calibrator and compare it
   to the piecewise experts. Preserve each simpler expert as a control.
