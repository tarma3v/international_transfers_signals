# AP3 registered packet: stable score scale, joint utility, sequential cadence

Registered 2026-09-06 before evaluating AP3 variants. Previous goal turn was
PROGRESS: AP2/D20 actual fits,135 tests,PDF and code pushed as0fc3d1a. Goal active.

## Starting evidence and unchanged information set

Read AP2-D20 files, not just past commentary. Quarterly AP2 score inspection:
TJS local Ridge median -273.603bp in2023Q1 and+21.624bp in2023Q4; signal counts
1 versus45. Residual medians also jump. Later diagnostic slices show long
pauses, but they are already-opened observations, NOT a new holdout. Scale and
refit drift are hypotheses to test, not a proven unique cause of all gaps.

Use AP2-D20 source panel exactly: own announcement events,18:30 MSK decisions,
assumed CBR receipt previous-to-effective calendar day18:00,20min market delay.
Same new-published-reference targets h1/3/5/10/20, same baseline event dates,
133 original features. No extra future CBR step inserted. Keep all original
AP2 outputs; reconstruct and compare source rows before fitting.

## Fixed feature/target and model experiments

Scale s(T)=max(announced past20-return volatility in bp,1). It is known at T.
- AP2 local Ridge and full residual scores, divided by s(T) without refitting,
  are separate causal rescaling ablations, not newly trained normalized models.
- Refit local standardized Ridge(alpha100) to floor5_log_bp/s(T).
- Refit global HistGB squared and quantile.25 models on the same normalized floor.
- Global HistGB correction of normalized local Ridge using genuinely earlier
  quarterly OOS anchor residuals, weights25/50/100%. No in-sample anchor errors.
Same quarterly origins2022Q3 onward, expanding training since2022, allh20 mature
strictly beforeorigin-2calendar days. HistGB160iter/.05/15leaves/minleaf40/l2=5,
no early stopping, seed20260906. Local min60,global min400,residual min200;
otherwise past-train-mean or zero-correction fallback as in AP2.

Five additional global HistGB absolute-error models forecast log(mean future h
prices/current announced price)*10000/s(T), separately h1/3/5/10/20. Reconstruct
predicted symmetric benefit with EXACT known past-price sum plus P(T) plus
forecast future-price sum. Only the future component is predicted. Clamp
predicted log relative mean to[-1,1] solely for numeric safety. Training future
mean targets and known past sums live in separate functions; true future utility
never enters a signal gate. Gate5=predicted symmetric benefit h5>0;
gate_all=min five predicted benefits>0. No claim these are calibrated lower bounds.

## Fixed scores and policies

14 primitive scores: AP2 CNY, AP2 selectedmix,market HistGB,market ExtraTrees,
local_raw,local_rescaled,residual_raw,residual_rescaled;6 newly fit normalized
scores (local,globalfloor,globalquantile,residual25/50/100).
3 hybrid scores: equal mean of prior63-date CDF scores for CNY+normalizedlocal,
CNY+normalizedresidual50,CNY+AP2marketHistGB. CDF strictly compares to previous
scores only,40warmup; constant scores/ties get midrank, not all alerts.

Three policies for each of17 scores (51 total):
1. AP2 prior250-score r25,40warmup,strict threshold, no weekly cap (control).
2. prior63-score r35,40warmup,strict threshold, max2 signals/ISO calendar week.
3. prior63 CDF, max2/week, at least2calendar days between signals; threshold
   max(.45,.80-.04*days_since_last_signal), initial age7days. This is a frozen
   urgency heuristic, not a guaranteed weekly minimum or optimized weekly top-k.
Apply the last controller to CNY/HistGB/ExtraTrees with gate5 andgate_all,
6 additional candidates; total57 policies. Gating is applied BEFORE consuming
budget/updating last sent date. State crosses years; weekly cap resets only on
observed ISO week change, never using future event calendars. There is no
unconditional quota-fill, although a neutral midrank can pass the relaxed
urgency threshold after a long pause; its quality must be measured.

## Frozen early selection and diagnostics

Only2023 for selection, allh20 resolved strictly before2024-01-01. Every candidate
uses the same eligible date support. Early300 paired circular20-date bootstrap
estimates lower2.5% symmetric benefit perh. Score = min adjustedlift -2*cadence
penalty -max(-min benefit lowerCI/50,0) -.05*max(max_week_signals-2,0).
Cadence penalty=max(1-min_currency_rate,0)+max(max_currency_rate-2,0), acrossh.
Tie meanlift then stable insertion order. Record whether any candidate jointly
passes pointlift>=1.3, rate1..2, positive allh benefit lowerCI, max2/week. If any
pass, select the highest scored among them; otherwise use the penalized best
and do not pretend it meets all gates. Apply this separately to simple candidates.
Save overall and simple
selection before later scorecards; no replacement based on later rankings.

Opened2024-2026 remains retrospective. Preserve every score/signal/selection,
train and residual cutoff logs, raw/rescaled/normalized comparisons, quarterly
score diagnostics. Later uncertainty1000 paired20-date blocks; peryear/currency
lift, symmetric/forward benefit and communication clustering. Compare AP2 mix
and pureCNY controls, not legacy effective-reference lift2.265.

Required tests: future-only price corruption cannot change known past sum or
normalizer; utility reconstruction identity; mature h20 labels and genuine OOS
anchor checks; policy no future scores; ISO week cap/year boundary; gated-off
rows do not consume budget; future true utility cannot directly trigger a gate.
