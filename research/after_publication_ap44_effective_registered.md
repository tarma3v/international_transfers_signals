# AP44-E registered protocol — quarterly meta-CatBoost core quality

Registered before inspecting AP44 predictions or scorecards. The 2024–2026
period is already opened, so all AP44 later results are retrospective.

## One new model

Train one global quarterly CatBoost classifier for the strict 20-publication
target `y20`: today's effective CBR fixing is no higher than every one of the
next 20 effective fixings. The model is a nonlinear stack over the 20 AP40
current-only features (three independently generated OOS expert ranks, their
agreement/spread, already announced change, cyclic weekday, AP37 causal pace
state, and calendar guards), plus:

- five one-hot corridor identities;
- annual sine/cosine;
- a fixed post-2022-02-24 regime flag.

Fixed fit: CatBoostClassifier, 240 trees, depth 5, learning rate 0.03,
L2=10, Bernoulli subsample 0.8, balanced classes, seed 20260906. Training rows
receive a 730-day half-life. Each calendar quarter is predicted by a fit using
only earlier rows whose publication-h20 label maturity is at least two calendar
days before the quarter origin. No prediction is fitted on its own quarter.

## One fixed policy

Use AP37's core/fallback opportunities and the AP40 sequential controller:

- accept an AP26 core opportunity when meta probability >=0.50;
- override a rejection only for Friday, trailing rate below 1.0, >=10-day
  silence, or an empty currency-month at day 24+;
- retain AP37's mature-quality late-week/silence fallback and cap two/week.

The 0.50 threshold is the classifier decision boundary under balanced training,
not a scorecard-tuned quantile. AP37 is the registered fallback if the 2023
early gate fails.

## Evaluation

Primary reference is today-effective CBR; h=1 is validity-only. Model selection
uses 2023 and h=3/5/10/20. Opened 2024–2026 is reported once after selection.
Strict point success requires min lift >2.4, per-currency rate 1–2 at every
horizon, max two/week, and no empty complete currency-month. Benefits, paired
20/50-date block uncertainty and prefix-corruption causality are mandatory.
