# T40 registration: long annual rolling-origin h20 history expert

Registered after T39 was rejected and before any T40 model was fit or scored.
The 2019-2024 outcomes, candidate predictions, and gates were not inspected for
this design. T40 adds earlier chronological OOS information instead of further
splitting the two T39 development years.

## Question

Can one fixed, explainable annual rolling-origin logistic expert produce a
stable h20 probability across 2019-2024 and then improve T37's `cbr_history`
state without changing any intraday or receipt route?

## Frozen model and information contract

- use the 41 T24 compact features from the causal CBR publication feature
  matrix; no market, receipt, or future feature is added;
- for evaluation year `Y`, fit `StandardScaler + L2 LogisticRegression`
  (`C=0.1`, `lbfgs`, `max_iter=5000`, fixed seed) on rows dated before
  `Y-1-01-01` whose h20 label matured before that origin minus the existing
  two-day embargo;
- score the complete `Y-1` calibration year with that already frozen raw
  model, use only labels matured before `Y-01-01` minus embargo, and fit one
  Platt logistic (`C=1.0`) on the raw logit;
- apply both frozen stages to publication rows in year `Y`;
- compute a global causal h20 prior from all labels matured before each
  publication date minus embargo;
- final publication probability is the fixed 50% log-odds blend of that prior
  and the annual Platt probability;
- generate annual OOS rows for 2019-2026 with no hyperparameter or feature
  selection.

Every training/calibration cutoff, latest maturity, row count, coefficient, and
publication prediction must be saved. Changing targets or features after a
query year may not change its fitted model or historical predictions.

## Frozen historical evidence split

- historical screen: OOS publication predictions from 2019-2022;
- historical validation: OOS publication predictions from 2023-2024;
- open diagnostic: 2025-2026, accessible only after the full historical gate
  is computed and passes;
- paired circular moving-block bootstrap resamples whole publication dates and
  all five currencies together, with 1,000 deterministic draws and blocks of
  20 and 50 dates.

For screen and validation separately, the pooled gate requires:

- Brier and log-loss delta versus the causal global prior below zero;
- AUC delta above zero;
- ECE delta no greater than +0.01;
- every 20/50-date Brier CI upper bound below zero;
- every 20/50-date AUC CI lower bound above zero.

Within both stages, every year and every currency must also be non-inferior:
Brier delta <= +0.001, log-loss delta <= +0.003, ECE delta <= +0.01, and AUC
delta >= -0.005. No local exception or later weight is allowed.

`historical_gate_passed` requires every pooled and local condition. If it
fails, 2025-2026 model metrics are not opened, T37 remains exact, and the
experiment ends as a historical negative result.

## Frozen open route and success rule

Only if the historical gate passes, join the annual publication prediction
backward by `currency + publication_date` and replace T37 probability only
when the observed `snapshot_source_kind == cbr_history`. Every market, bridge,
perpetual, hold, and receipt-dependent row must remain bitwise equal to T37.

The open repair passes only if:

- all 40 `scenario x clock` rows are non-inferior to T37 under +0.001 Brier,
  +0.003 log-loss, +0.01 ECE, and -0.005 AUC allowances;
- both pooled scenarios are non-inferior to T37;
- all four pooled year groups pass the strict T38 evidence rule;
- clock-local pass count exceeds 619/680;
- pooled local pass count exceeds 25/34;
- source timestamps never exceed query timestamps and non-history rows are
  bitwise equal to T37.

No open result may alter the annual training rule, features, hyperparameters,
blend, split, or gate.

## Interpretation

T40 cannot promote production even if both stages pass. The design follows
inspection of T39 and the final 2025-2026 period is already open. A pass creates
a stronger frozen prospective challenger. A failure preserves T37 and proves
that a longer daily-CBR history alone does not solve local h20 calibration.
Sparse push, benefit magnitude, receipt policy, client copy, and runtime remain
unchanged. `production_promoted=false`.

## Required audit

- source hashes for registration, code, audit, raw CBR/cache inputs, T37, and
  T38 evidence;
- exact reconstruction of annual fits, prior, predictions, historical metrics,
  bootstrap, gates, and any conditional open outputs;
- complete annual/currency grids and uniqueness;
- fit/calibration maturity and embargo proofs for every year;
- future-prefix corruption across model features and targets;
- exact non-history preservation and final decision reconstruction.
