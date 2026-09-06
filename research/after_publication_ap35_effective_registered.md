# AP35-E frozen protocol: monotone distributional residual CatBoost

Registered 2026-09-06 before AP35 model fits, signals or scorecards. AP34 showed
that a linear conditional residual model is feasible but over-selects pace rows
and loses long-horizon lift. AP35 tests one nonlinear distributional model, not
a grid or an AP34 hyperparameter repair.

Use the same mature conditional residual target:
`residual_floor20 = effective_floor20 - known_change`. For every eligible mature
training row create five threshold examples whose synthetic announced anchors
equal the actual known change plus -200,-100,0,+100,+200 basis points. The label
is whether the observed residual floor stays above the negative synthetic anchor.
This teaches a conditional residual CDF rather than only the mean or one observed
binary boundary.

Use the frozen AP13 compact feature set, excluding direct `known_change`,
`known_change_z` and `local_minus_common`, then append the synthetic anchor as
one explicit feature. Fit one CatBoostClassifier per quarterly OOS origin with
320 iterations,depth6,learning_rate.035,l2=10,random_strength.5,Bernoulli
subsample.8 and fixed seed. Constrain the appended anchor monotonically positive.
Apply train-only half-life730 calendar recency weights. Training rows must have
publication-h20 fully mature before origin minus two days; no query-quarter label
is available.

At inference append the actual announced anchor. Pass the sole probability score
through the exact frozen AP21 dual-paced-month controller using the frozen
rolling primary and reserve. Treat its output as core and apply the exact frozen
AP33 calendar router with frozen AP23 fallback. No alternate offsets, features,
CatBoost settings, weights or policy thresholds are evaluated.

Select the sole fresh candidate on early2023 using unchanged joint gates; frozen
AP33 is the registered fallback. Opened2024-2026 remains retrospective. Audit
all17 fits, expanded labels/weights/monotonicity, mature masks, nested policies,
future-feature/label corruption and paired20/50-date uncertainty. Acceptance is
min h3/h5/h10/h20 lift>2.4,min currency rate>=1,zero empty complete months and
max2/week. H1 is known validity only; receipts remain calendar-assumed and bank
execution is unvalidated.
