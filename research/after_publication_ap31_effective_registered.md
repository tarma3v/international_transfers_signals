# AP31-E frozen protocol: factorized short/long survival CatBoost

Registered 2026-09-06 before AP31 fits, signals or scorecards. AP28-AP30 show
that another static shrinkage or rank blend does not improve the frontier.
AP31 changes the prediction target itself and tests exactly one score.

The y20 event is rare and nested inside y3. At each quarterly OOS origin fit two
CatBoost classifiers using only eligible rows whose publication h20 is mature
at least two calendar days before the origin. Model A predicts y3 on all usable
rows. Model B predicts y20 only among rows with y3=1. The score is the causal
factorization P(y3=1) * P(y20=1 | y3=1). Both use the same existing133 features,
including currency and market/CBR context; no current or future outcome enters
the features. If a stage has fewer than100 rows or one class, use its mature
class mean.

Both fixed classifiers use Logloss,360 trees,depth6,learning_rate.035,
l2_leaf_reg10,random_strength.5,Bernoulli subsample.8,seed20260906 and two
threads. There is no target weight, class rebalance, calibration, local fit,
blend, threshold grid or late-period selection.

Use the factorized score only as pace expert in the exact AP21 policy: rolling
ExtraTrees primary; after84 days, pace prior250 rank>.55 while trailing365
currency rate<1 and reserve rank>.70; month24 rescue, known-down veto and
max2/ISO-week. AP21, AP23, AP26, AP27, AP29 and AP30 are frozen controls. Select
the sole fresh candidate on early2023 with unchanged joint gates; opened
2024-2026 remains diagnostic.

Audit all34 stage fits, maturity/train/conditional masks, deterministic refit,
factorized probabilities, policy state, future-feature/label prefix invariance
and paired20/50-date uncertainty. Acceptance remains min h3/h5/h10/h20 lift>2.4,
min currency rate>=1, zero empty complete months and max2/week. H1 is known
after receipt and excluded; timestamps remain calendar-assumed and bank
execution is unvalidated.
