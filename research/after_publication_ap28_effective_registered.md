# AP28-E frozen protocol: hierarchical pooled y20 pace expert

Registered 2026-09-06 before AP28 fits, signals or scorecards. AP25 showed that
training only on the outcome-free hard cadence pool can improve accuracy but has
a severe cold start. AP26 repaired it by score-level count shrinkage. AP28 tests
exactly one different low-data model and no threshold grid.

The sole fresh score is `hier_y20_weight4`. For every quarterly OOS origin it
fits one CatBoost y20 classifier on all eligible, publication-h20-mature rows.
Ordinary rows have weight1; rows in the causal hard pool have weight4. The hard
pool is unchanged from AP25: known next is not lower, rolling ExtraTrees causal
rank is at most.70 and reserve causal rank is above.70. The 133 existing features
already include currency one-hot columns. Three outcome-free columns are added:
rolling rank, reserve rank and hard-pool flag. This lets currencies share the
tree prior while the model can learn a separate correction in the sparse regime.

Model constants are fixed: Logloss CatBoostClassifier,400 trees,depth6,
learning_rate.035,l2_leaf_reg12,random_strength.5,Bernoulli subsample.8,
seed20260906 and two threads. There is no class rebalance, target calibration, local
refit, post-hoc blend or search over hard-pool weight. If a fit has fewer than100
usable rows or one target class, use its mature weighted class mean.

The score is used only as the pace expert in the exact AP21
`dual_paced_month_policy`: rolling ExtraTrees primary rank>.70; after84 days,
pace rank>.55 only while trailing365 currency rate<1 and reserve rank>.70;
month24 rescue, known-down veto and max2/ISO-week remain exact. AP21, AP23, AP26
and both AP27 outcomes are frozen controls. The only fresh candidate is selected
on early2023 with the unchanged joint gates; 2024-2026 remains an opened late
diagnostic and cannot create a fresh winner.

Audit the 17 quarterly fits, exact maturity masks, hard-pool weights and added
features, deterministic refit, policy state, future-prefix corruption and
paired20/50-date uncertainty. Acceptance is min h3/h5/h10/h20 lift>2.4,
min currency rate>=1, zero empty complete months and max2/week. H1 is known after
receipt and excluded from selection. Receipt timestamps remain calendar-assumed;
actual bank execution is not validated.
