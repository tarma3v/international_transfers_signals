# AP49-E frozen protocol — four horizon experts with geometric consensus

Registered 2026-09-06 before AP49 predictions, signals, or scorecards were
computed. The 2024–2026 period is already opened, so late results are explicitly
retrospective.

Train four independent global quarterly CatBoost classifiers for the
today-effective targets `y3`, `y5`, `y10`, and `y20`. Each head uses its own
publication-horizon maturity and the same two-calendar-day embargo, so shorter
heads may use labels that are already known even while longer labels are not.
All heads use the season-light AP46 matrix: the 20 frozen AP40 current-only
features plus five currency one-hot indicators, with no annual sine/cosine or
post-2022 flag.

Every head has the fixed AP44 configuration: 240 trees, depth 5, learning rate
0.03, L2=10, Bernoulli subsample 0.8, balanced classes, seed 20260906, and a
730-day half-life. Predict each quarter only from earlier mature rows. Combine
the four probabilities by their geometric mean after numerical clipping to
`[1e-6, 1-1e-6]`. This fixed consensus penalizes a day that looks weak on any
horizon without letting one head act as a hard minimum. No head weights,
features, hyperparameters, or aggregation alternatives are screened.

Feed the consensus to the unchanged AP40/AP46 router with threshold 0.50 and
rate floor 1.0, using AP37 core/fallback inputs and max two signals per
currency-week. AP37 is the registered fallback if the sole candidate fails the
unchanged 2023 early gate.

Report h=3/5/10/20 lift, cadence, symmetric and future-only benefit,
currency/year slices, decision changes, paired 20/50-date bootstrap, and
future-feature/label/maturity corruption. `h=1` remains validity-only;
historical receipt timestamps and executable bank prices remain unvalidated.
