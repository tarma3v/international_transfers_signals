# T44 registration: disjoint mature calibration-quality gate

Registered after T43 failed its historical proper-score gate and before any
T44 fit, gate decision, prediction, or outcome metric was computed. T40-T43
and every 2025-2026 result are already open. T44 is therefore a mechanism test
and cannot become production-proven from this replay.

## Question

Can an observable, delayed and fully mature validation slice prevent the
nonlinear T43 history expert from influencing years in which its probability
mapping is unreliable, while retaining useful nonlinear rank in stable years?

## Frozen candidate

Use T43's exact 41 causal CBR features, h20 target, annual raw
`HistGradientBoostingClassifier`, training origin, maturity rules, embargo,
model parameters, causal prior and final 50% log-odds blend. For query year
`Y`:

1. Fit the raw tree model exactly as T43 on labels strictly before `Y-1`, with
   the same mature-label cutoff and embargo.
2. Split calendar year `Y-1` once at July 1. Fit an early Platt calibrator on
   January-June rows whose h20 labels matured strictly before July 1 minus the
   existing embargo.
3. Evaluate the early-calibrated 50% log-odds blend on July-December rows whose
   h20 labels matured strictly before January 1 of `Y` minus the embargo. This
   second half is not used to fit the early calibrator.
4. Open the nonlinear expert for year `Y` only if all frozen quality conditions
   below pass on that second-half gate. Otherwise emit the causal prior exactly.
5. When the gate opens, refit only the Platt calibration on all eligible rows
   of `Y-1`, as T43 already does, and apply the unchanged T43 blend throughout
   `Y`. The raw model, features and weight are never refit or selected from the
   query year.

There is one candidate, one calendar split, one set of gates and no grid.
No currency, year, row, feature, hyperparameter, alpha or threshold is selected
from the query year or from 2025-2026.

## Frozen gate conditions

On the July-December gate slice, compare the early-calibrated blend with the
causal prior using point metrics. Open the expert only when:

- the early Platt slope is strictly positive;
- pooled Brier delta is below zero;
- pooled log-loss delta is below zero;
- pooled ROC AUC delta is above zero;
- pooled ECE delta is at most `+0.01`;
- all five currency groups satisfy T40's unchanged local non-inferiority
  tolerances: Brier `<= +0.001`, log-loss `<= +0.003`, ECE `<= +0.01`, and
  AUC `>= -0.005`.

The zero proper-score and AUC boundaries come from the requirement to beat the
prior. The ECE and local tolerances are inherited exactly from T40; they are
not tuned in T44. Require at least 400 eligible rows in each half and both
classes in every pooled fit/evaluation slice. A support failure closes the
expert.

## Historical and open evaluation

- historical screen: annual OOS publication rows in 2019-2022;
- historical validation: annual OOS publication rows in 2023-2024;
- open diagnostic: 2025-2026 only if both historical stages pass;
- use T40's exact paired circular moving-block bootstrap, gates, complete
  year/currency grids and conditional T37 route audit.

Thus the gate itself may learn only from `Y-1`; T44's experiment-level decision
still requires the disjoint historical screen and validation. A failed stage
must prevent opening any 2025-2026 model metrics.

## Interpretation

A pass supports delayed competence gating as a causal regime mechanism. A
failure rejects this exact half-year quality gate; it does not prove that
nonlinear CBR structure or all causal expert selectors are useless. T44 never
changes T37, sparse push, expected future-only bps, direct-pair routing, runtime
timestamps or customer copy. `production_promoted=false` regardless of result.

## Required audit

- exact reconstruction of annual raw fits, both Platt fits, gate metrics,
  decisions, publication probabilities, historical metrics and bootstrap;
- early-calibration and gate maturity cutoffs strictly respected;
- query year never used by its own gate;
- rejected years equal the causal prior bit-for-bit;
- fixed estimator parameters and 160 iterations in every annual raw fit;
- corruption of later features, labels and maturities cannot change an earlier
  prediction prefix;
- source hashes and conditional non-history preservation if the historical gate
  reaches the open route.
