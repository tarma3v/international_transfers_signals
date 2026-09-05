# AP4 frozen packet: first-passage targets and soft utility

Registered 2026-09-06 before fitting/scoring AP4. Previous goal turn was
PROGRESS: AP3 models, 142 tests, report and commit203ae4a pushed. No hourly task.
The active indefinite goal is unchanged. This packet does not claim a new holdout.

## Information and evaluation remain fixed

Use AP3/AP2-D20 rows and source hashes: 5755 own-announcement events from2022,
133 features, latest announced CBR reference, decision18:30MSK, additional20min
market delay. Receipt times remain CALENDAR-ASSUMED, not historically certified.
Rebuild features/outcomes and assert exact row/outcome/control agreement.
All model fits quarterly from2022Q3, expanding since2022; allh20 mature strictly
before origin minus2calendar days. Early selection:2023, allh20 mature before2024.
Later2024-2026 already opened, retrospective only. No UZS2026-specific fitting.

## Distinct models (not just tree-depth tuning)

Define a failure as first future price STRICTLY BELOW the latest announced
price. Equality is survival, consistent with the case target. Five intervals
end at1/3/5/10/20 next observations. A training event contributes intervalj only
if it survived through the preceding endpoint; response=1-y_hj. Features at
the original decision are repeated with a5-column interval one-hot. No realized
future prices or future covariates enter this matrix. Train only completeh20
outcomes; do not infer censored late rows as successes.

Fit global pooled LogisticRegression(C=.1,standardized,max_iter1500), global
HistGB classifier (AP2 settings, no early stopping), and one such logistic
hazard model percurrency with AP2 local feature subset. For small samples use
training-only empirical hazards; min400global/min60local events. Conditional
survival products must stay monotone in h. Scores: survivalh5 and meanfive-h.

Also fit HistGB regression of log(1+min(first-cheaper index,21)), with21 meaning
no cheaper observation within20. This is restricted waiting time, NOT an
uncensored estimate beyond20. Same AP2 regressor settings.

Correct the LOCAL hazard model's meanfive-h survival with global HistGB trained
only on genuinely previous quarterly OOS residuals (meantruey - prior prediction).
Weights25/50%, min200 mature OOS rows; no in-sample anchor errors. These corrected
scalar scores are NOT survival curves/probabilities and are labelled separately.
Total9 new scalar scores:6survival +1restrictedwait +2OOSresidual.

## Soft utility and ensemble ablations

Fixed AP3 controller everywhere: prior63midrankCDF,40warmup, threshold
max(.45,.80-.04*days_since_last_sent), min2calendar days, max2/ISOweek. No future
week top-k; neutral rank may pass a relaxed threshold. No guarantee everyweek.

CNY and model scores converted separately to prior63CDF before blending; the
mixture passes through the unchanged controller's second past-CDF stage.
- CNY+AP2HistGB and CNY+AP2ExtraTrees, other-model weights25/50/75%, plus equal
  CNY/HistGB/ExtraTrees triple.7blends, includes exact AP3 incumbent.
- Nine new scalar scores, each alone and50/50 with CNY:18policies.
- Six soft utility sources: knownpast-flatfuture symh5 and minallh; AP3 forecast
  symh5 and minallh; AP3 futuremean forecast h5 and meanallh. Symmetric sources
  divided by the same available scale before prior63CDF; future forecasts are
  already normalized. Knownpast variant reconstructs benefit with zero forecast
  return, not true future prices. Each blended with AP3 raw mixture at weights
  10/25/50%, no hard gate:18policies.
- Three primitive AP3 controls: CNY/HistGB/ExtraTrees urgent.46policies total.

## Frozen selection and preservation

Primary feasible-first AP3 joint selection, with an additional EARLY future-only
guard: each h's mean forward benefit must be >=80% of incumbent's positive early
mean for that h. Allh lift point>=1.3, percurrency rate1..2, allh symmetric
benefit lower95%CI>0 (300 circular20-date bootstrap), max2/week.
Rank by minlift -2*cadencepenalty -max(-minbenefitlowerCI/50,0), tie meanlift then
stable insertion order. Incumbent is included and must reproduce AP3, otherwise
stop. If none feasible, retain incumbent and report the failed selection.
Write selection before any later scorecards. No later winner replacement.

Preserve all46scores/signals, survival matrices, restricted labels, quarterly
fit/maturity/residual logs, source hashes, early selection. Later1000 paired
20-date lift/utility blocks, peryear/currency andyear×currency slices. Fixed
diagnostic families plus selected; test50-date blocks for selected vs incumbent
as uncertainty sensitivity, not re-selection. Freshness and bank caveats unchanged.

Required tests: person-period at-risk handling/ties/monotone targets; missing
h20 rejected; cumulative survival identity andbounds; future labels cannot enter
prediction design; restricted waiting cap/index; immature/current-quarter OOS
residual exclusion; soft utility no-future corruption; exact AP3 reproduction.

Delayed online expert allocation is deferred to AP5 if useful; do not claim it
was tried in AP4. No publication/bank audit needs repeating before these fits.
