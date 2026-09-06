# AP46-E frozen protocol — joint multi-horizon survival meta-model

Registered 2026-09-06 before AP46 predictions, signals, or scorecards were
computed. The 2024–2026 period is already opened, so every later result remains
retrospective rather than a fresh holdout.

## One new model

The target is deliberately stricter than AP44. For an eligible announcement
row, `joint_good=1` only when all four today-effective-CBR targets
`y3`, `y5`, `y10`, and `y20` equal one. Thus the model estimates whether taking
the current opportunity survives every unknown horizon required by the product,
rather than optimizing the longest horizon alone. The label is considered
mature only at publication-h20 maturity, with the existing two-calendar-day
embargo.

Use one global quarterly CatBoost classifier over the 20 frozen AP40
current-only features plus five currency one-hot indicators. Deliberately omit
annual sine/cosine and the post-2022 flag: AP44 assigned about 43% importance to
annual phase, and this experiment asks whether a joint target transports without
that potentially fragile seasonal shortcut.

Fixed fit: 240 trees, depth 5, learning rate 0.03, L2=10, Bernoulli subsample
0.8, balanced classes, seed 20260906, and a 730-day half-life. Every calendar
quarter is predicted by a model fitted only on earlier eligible rows whose joint
label matured before quarter origin minus two days. No threshold, feature,
decay, or hyperparameter grid is allowed.

## One fixed policy

Use AP37 core/fallback opportunities and the existing AP40 sequential router:

- accept an AP26 core when joint probability is at least 0.50;
- override a rejection for Friday, trailing rate below 1.0, a ten-day silence,
  or an empty currency-month from day 24;
- retain AP37's mature-quality late-week/silence fallback;
- allow at most two signals per currency and ISO week.

The 0.50 cutoff is the balanced classifier boundary, not a scorecard-tuned
quantile. AP37 is the registered fallback if the sole AP46 candidate does not
pass the unchanged 2023 early gate.

## Evaluation

The primary reference remains the today-effective CBR fixing. `h=1` is a
validity diagnostic because the next fixing has already been announced; model
selection uses only `h=3/5/10/20`. Opened 2024–2026 is disclosed once after the
early decision. Report adjusted lift, signal cadence, symmetric and future-only
benefit, currency/year slices, paired 20/50-date bootstrap, decision changes,
and prefix-corruption causality. Historical receipt timestamps and executable
bank quotes remain unvalidated.
