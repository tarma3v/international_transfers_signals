# AP5 frozen: mature OOS calibration and delayed expert weighting

Registered2026-09-06 before AP5 evaluation. Previous turn PROGRESS: AP4 fits,
150tests, PDF, commitcb69c08 pushed. Goal stays active, no hourly automation.
Same AP4/AP3/AP2-D20 information,5755events,publication reference18:30MSK,
20min additional marketdelay, calendar-assumed receipts, no bank validation.

## Calibration, not retraining base classifiers on evaluation labels

Six experts: CNY z score, AP2 directHistGB/ExtraTrees h5 scores, AP4 survival
HistGB/globalLogit/localLogit five-h curves. Features are one-dimensional:
arcsinh(CNYz), or clipped logit probability (clip1e-4..1-1e-4). Direct h5 scores
are calibrated separately against each h; survival experts use corresponding h.
Base forecasts remain genuine quarterly OOS, source arrays frozen and hashed.

Monthly calibrators from2022July, trained only on prior base OOS forecasts with
FULLh20 mature strictly before monthorigin-2calendar days. Five modes:
1. expanding positive-slope logistic calibration;
2. last365calendar days logistic;
3. rolling365 global + local logistic, local fraction n/(n+250), min60local;
4. expanding non-decreasing isotonic;
5. expanding logistic until2023-01-01, then freeze that calibrator forever.
Globalmin200 and bothclasses; otherwise train-only smoothedprior from mature
history (also if insufficient base OOS observations). Initial CNY probability
is not treated as a known true probability. Positive logistic slope enforced
by bounded optimization; training-only standardization and ridge slope penalty
equivalent to C=1. Independently calibrated horizon probabilities projected to
non-increasing by cumulative minimum; causal but can affect calibration.
Before source forecasts exist, output NaN; after start fallbacks are finite.
All original2023/later comparison dates retained, never chosen by outcomes.

## Delayed weighting with explicit information clock

Use ONLY rolling365-logistic probability experts as the adaptive pool. Perrow
loss=mean five-h Brier error of the probability actually issued at that row.
An update is allowed only if h20 mature < currentday-2days, hence effective
availability=matureday+3calendar days. Prediction dates must also precede T.
Not retroactively recalibrated predictions; same-day currencies all receive
the same predecision global state. No reward from unsent future signals.

For each T compute exponentially decayed mean loss from eligible past rows,
half-lives63/252 CALENDAR days measured from feedback availability. Eta2/10/30.
Weights=.9*softmax(-eta*meanloss)+.1*uniform; initial equal weights. Compare global
and local-loss shrunk to global with fraction effective_local_mass/(mass+50).
12adaptive cases. This is a discounted delayed exponential-weight heuristic,
not claimed to reproduce a paper's regret guarantee or optimize lift directly.
Save each row's six weights, count/lastlabel/maturity cutoff and losses.
Frozenweight control: global eta10/half252 as of2023-01-01, equal beforethat.
No fit on2023 outcomes to issue2023 signals.

## Exactly66 fixed signal policies

Unchanged AP3 sequential controller: prior63midrank40warmup, urgency threshold
max(.45,.80-.04*days_since_last_sent), min2days,max2/ISOweek. No futureweektop-k.
- 5calibration modes *6 experts, h5 =30.
- Equal probabilities, h5 andmeanfiveh, for5modes=10.
- 50%CNY probability +50%averageotherfive, h5, for5modes=5.
- 12adaptive mixtures, h5=12.
- Six GLOBAL adaptive mixtures blended50/50 with raw CNY past-CDF; adaptive
  mixture also past-CDF before blend; controller then ranks mixture=6.
- Frozenweight h5 control=1.
- Exact AP3 and AP4 selected controls=2. Total66.

Early selection exactlyAP4 jointfeasible-first, including its forward-only
guard>=80% AP3 perh;2023 outcomes allh20 mature before2024.300 dateblock bootstrap.
Selection file saved BEFORE opened-later scorecard. Do not promote on laterBrier
or bestlate lift. Later1000paired20/50dateblocks, allhbenefit/forward/frequency,
year×currency, calibration and weight diagnostics. No new holdout claim.

Required tests: calibrator future-label corruption; same-date ordering invariant;
strict maturity+embargo boundary; unreleased labels cannot alter earlier weights;
issued probabilities never replaced by later calibrations; weights finite/sum1
and10%uniformfloor; frozen controls never consume2023 labels; localshrink/global
limits; nonnegative-slope andmonotone-h calibration; exactAP3/AP4 reproduction.
