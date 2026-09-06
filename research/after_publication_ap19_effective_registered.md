# AP19-E frozen protocol: multi-horizon boosting and direct ranking

Registered 2026-09-06 before fitting or scoring any AP19 model. AP18 showed
that a fixed expanding/recent blend can help slightly, while learned residual
corrections hurt. AP19 therefore changes the learning objective and model
family. It does not tune the AP17 decision state machine.

The reference remains the TODAY-EFFECTIVE CBR rate after actual receipt of the
announced tomorrow fixing. The first next fixing is known, so h1 is a validity
diagnostic rather than model skill. Historical receipt time is
CALENDAR-ASSUMED; official-CBR benefit is not executable bank P&L.

## Frozen causal evaluation

- Reuse the 5,755-row AP12/AP13 panel and 133 as-of-18:30 features.
- Use 17 quarterly walk-forward origins beginning 2022-07-01.
- At each origin, fit only eligible rows whose publication-h20 target matured
  before the origin minus the frozen two-day embargo.
- All four conditional binary targets h3/h5/h10/h20 must be complete.
- Select only on early 2023, before opening the saved later scorecard.
- Report 2024-2026 as repeatedly opened retrospective evidence, never as a
  fresh holdout.

## Fixed model definitions

Full 133-feature matrices are used. No hyperparameter search.

1. Fit four AP12-style ExtraTrees classifiers, one per conditional horizon.
   Each has 400 trees, depth 8, min leaf 25 and max_features 0.6.
2. Fit four CatBoost classifiers, one per conditional horizon, with 320 trees,
   depth 6, learning rate 0.035, L2 10, random strength 0.5, Bernoulli
   subsample 0.8, Logloss, fixed project seed and no early stopping.
3. Fit one CatBoost regressor with the same tree parameters to the mean of the
   four survival labels.
4. Fit one CatBoost PairLogit ranker to h5, grouping training rows by currency.
   Apply a fixed logistic transform to its raw score before the controller.

The ten prespecified predictor scores are:

- arithmetic and geometric means of the four ExtraTrees probabilities;
- CatBoost h5 probability;
- arithmetic and geometric means of the four CatBoost probabilities;
- CatBoost mean-survival regression;
- CatBoost PairLogit h5 rank score;
- 75/25 and 50/50 AP12-Extra/CatBoost-h5 probability blends;
- 75/25 AP12-Extra/CatBoost-multi-mean blend.

If a fit has fewer than 100 usable rows or only one class, fall back to target
prevalence for raw models. For AP12 blends, a missing base value falls back to
the CatBoost component. Probabilities and regression outputs are clipped to
[0,1]. No AP19 target from a query quarter is used in its fit.

## Frozen controller and selection

Every score independently enters the exact AP17
`pace365_p55_r70_month24_cap2` controller: prior-250 primary rank above 0.70;
after 84 days, rank above 0.55 plus the frozen reserve above 0.70 while the
prior-365 signal rate is below one per week; from day 24, the frozen reserve
may fill an otherwise empty currency-month. The known-down veto and maximum
two signals per ISO week dominate.

Early gates are unchanged: minimum lift over h3/h5/h10/h20 at least 1.3,
each-currency rate 1..2/week, zero empty complete months, maximum two/week,
positive symmetric-benefit CI and future benefit at least 80% of the simple
known-sign control. Rank passing fresh candidates by minimum unknown-h lift,
then mean lift, then gap. If none passes, use the registered rate/cap fallback.

Audit source hashes, 17 maturity masks, 170 fit-log rows, deterministic model
refits, score/controller reconstruction, bounded outputs, and prefix invariance
after physical corruption of future features, labels and AP12 scores. Compare
the selected and best-later diagnostic to AP17/AP18/AP12/rolling/local controls
using paired 20- and 50-date block bootstrap. The stretch target is a strict
minimum unknown-h lift above 2.4 at 1-2 signals per currency-week.
