# AP18-E frozen protocol: new predictors under the AP17 policy

Registered 2026-09-06 before fitting or scoring any AP18 model. AP17 is the
frozen strict-cadence control. AP18 changes only the predictor score; every new
score passes through the identical pace365_p55_r70_month24 state machine.
No policy threshold may respond to AP18 results.

Reference is TODAY-EFFECTIVE CBR after actual receipt of tomorrow's announced
fixing. Historical receipt near18:00 remains CALENDAR-ASSUMED, h1 is known after
receipt, and CBR benefit is not executable bank P&L.

## Frozen data and causality

Reuse5,755 AP12/AP13 rows,133 as-of18:30 features,publication-h20 maturity cap,
2-day embargo,17 quarterly origins,early2023 selection and opened2024-2026
diagnostic. Fit only eligible announced>=current rows whose full publication-h20
label matured before origin minus2 days. Target is conditional h5 survival of
the still-unknown steps after the known tomorrow fixing.

Training residuals use frozen quarterly OOS AP12 Extra predictions, never
in-sample predictions. Query rows use their frozen AP12 OOS score. Features and
labels after the quarterly origin cannot affect the fit.

## Six prespecified scores

Use the fixed AP12 compact59 subset. No hyperparameter grid.

1. `full_recent50`: arithmetic50/50 AP12 expanding Extra and AP13 rolling730-day
   Extra. No fitted weight.
2. `resid_hist100`: HistGradientBoostingRegressor on y5-AP12score;160 iterations,
   learning_rate.04,max_leaf_nodes15,min_samples_leaf40,L2=10,no early stopping.
   Add the full predicted residual to AP12 and clip to[0,1].
3. `resid_hist50`: same fitted residual, but add only50% as fixed shrinkage.
4. `resid_ridge`: StandardScaler+Ridge(alpha100) on compact residual; add full
   residual and clip.
5. `local_resid_ridge`: fit the same global ridge and a ridge per currency when
   at least100 rows; combine residuals by n/(n+200), then add to AP12 and clip.
6. `stack_logit`: StandardScaler+LogisticRegression(C=.05,max_iter2000) on
   compact features plus clipped logit(AP12 score), predicting y5 directly.

Fallback below100 usable rows or one target class is the frozen AP12 score.
All model seeds use the project seed. Write fit masks/counts and predictions.

## Frozen AP17 state machine

For every score independently: primary prior250 rank>.70,warmup40; after84 days
and while prior365 decision rate<1/week, allow score rank>.55 only if frozen
known70/hazard reserve rank>.70; from day24 allow reserve>.70 only when the
currency-month has no signal. Known-down veto and max2/ISO-week dominate.

## Selection and evidence

Select six fresh candidates on early2023 only. Gates: min lift across
h3/h5/h10/h20>=1.3,each-currency rate1..2,zero empty months,max2/week,
positive symmetric-benefit CI and future benefit>=80% of known-sign. Rank by
min unknown-h lift,mean lift,gap. If no joint passer, use registered rate/cap
fallback and label it.

Report later all-h metrics,Brier,currency/year slices,residual scale,paired20/50
date CIs versus AP17,AP12,AP13 and prior strict controls. Audit source hashes,
17 maturity masks,OOS base/rolling inputs,independent deterministic refits,
all score/policy states and future feature/label/score corruption. Later target
is minunknownlift>2.4 under AP17 cadence, but any success is retrospective.
