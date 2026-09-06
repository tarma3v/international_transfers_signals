# AP24-E frozen protocol: direct within-currency day ranking

Registered 2026-09-06 before AP24 fits, scores, signals or scorecards. Previous
classifiers and regressors predict each row independently, although the product
chooses only one or two days per currency-week. AP19's first PairLogit used only
h5 and one whole-history group per currency and was weak. AP24 tests whether
smaller decision-like groups and the full unknown-horizon utility improve direct
ranking. This is a new fitted-model round, not a late blend of AP23 results.

Use the frozen AP13 feature matrix and quarterly chronological OOS origins.
Training rows must pass the shared publication-h20 maturity mask with a two-day
embargo, be eligible under known-next-not-lower, and have complete h3/h5/h10/h20
labels. H1 is excluded. Five fixed CatBoostRanker scores are trained:

1. `rank_qtr_mean_full_pair`: PairLogit, currency-quarter groups, target
   mean(y3,y5,y10,y20), expanding history;
2. `rank_qtr_mean_roll3_pair`: the same, trailing1095-day history;
3. `rank_month_mean_full_pair`: PairLogit, currency-month groups, mean utility;
4. `rank_qtr_y20_full_pair`: PairLogit, currency-quarter groups, y20 survival;
5. `rank_qtr_mean_full_yeti`: YetiRankPairwise, currency-quarter groups,
   mean utility, expanding history.

Groups with fewer than two rows or no target variation are omitted from a fit.
All models use the fixed AP19 CatBoost geometry: 320 iterations, depth6,
learning_rate .035, L2=10, random_strength=.5, Bernoulli subsample=.8, seed42.
Raw rank scores receive only a monotone sigmoid before the product rank.

Each ranker is tested in exactly two roles under the AP21 state machine:

- ranker primary + frozen AP19 CatBoost-utility pace;
- frozen rolling-730 ExtraTrees primary + ranker pace.

This creates ten fresh policies. Every policy uses outer past-250 per-currency
ranks with warmup40, primary>.70, pace>.55 only when trailing365 rate<1/week and
reserve>.70, month24 rescue, known-down veto and max2/ISO-week. Thresholds are
unchanged. AP21 strict and AP23 competence-pace remain frozen controls.

Early-2023 selection, opened-2024-2026 diagnostic, today-effective CBR reference,
already known next fixing and all acceptance gates are unchanged. Audit shared
masks, group construction, fit logs, OOS completeness, exact refits, policy
state, future-feature/label corruption prefix invariance and paired20/50-date
uncertainty. Stretch target is strict min h3/h5/h10/h20 lift>2.4 with rate1..2,
zero empty complete months and max2/week. Historical receipt times remain
calendar-assumed and actual bank execution is not validated.
