# AP7 registered before fitting: distributions of normalized future paths

2026-09-06. Previous AP6 turn is PROGRESS: 16 meta models, 51 policies,
168 tests and checked push 71fcd0e. No confirmed superiority. AP7 changes the
forecast representation, not the early criteria or opened evaluation dates.

## Clock and library

Same publication-reference events and AP2-D20 timing: 18:30 decision, assumed
CBR receipt, 20 minute market feed delay. Not certified historical receipts,
not a new holdout, not executable bank savings. All h1/3/5/10/20.
Queries start July2022, library starts January2022. A library path is the next
20 cumulative log price ratios from that row's already announced fixing,
times10000 divided by its own known vol20 floored at1bp. Query paths are
rescaled by the query's available volatility. Target paths are never features.
Only fully mature20 < monthly origin minus2days can enter the library. There
is no requirement for an old expert prediction to exist: these models learn
from raw available context, not the AP6 experts. Earlier library support is
therefore larger than AP6, while query/evaluation support is identical.

Core7: arcsinh of CNY basis with known-change fallback, own known-change z,
announced ret5z/ret20z, log_vol20, arcsinh CNY late change, CNY quality. Wide
context is the 24 source-only AP6 features, no expert outputs/disagreement.
KNN standardization uses its own current library only, including local or
rolling-library restriction. Nearest ties use stable date/currency order.
Gaussian distance weights exp(-d2/median(selected d2)), with1e-8 denominator
floor; the normalizer uses only query distances, never query outcomes.

## Eleven distribution families

1. Core global nearest64.
2. Core global nearest128.
3. Wide global nearest128.
4. Core same-currency nearest32 (global128 fallback if fewer than20 local rows).
5. Shrink local32/global128 distributions with local_n/(local_n+200).
6. Core global nearest128, rolling730 calendar-day library.
7. Unconditional uniform distribution of all mature normalized paths.
8. Chronologically split distribution tree ensemble on core features: first
   60% of distinct library dates define the structure-training time boundary;
   structure labels must also mature before boundary-2days; the remaining40%
   dates supply disjoint outcome paths in query-matching leaves. 32 randomized
   trees, max_leaf_nodes12, min_samples_leaf30, max_features.8, fixed seeds.
   Multioutput split target at h1/3/5/10/20 divided by sqrt(h). Empty leaf uses
   uniform estimation sample. Global128 fallback if structure<100/estimate<30.
9. Core global multivariate Ridge(alpha100) conditional mean path plus each
   empirical residual path from the same chronological estimation sample.
   Standardizer fit on structure rows only. It never refits the old mean on
   estimation rows. Same global minimum and fallback as family8.
10. Same Ridge mean with Gaussian residual paths: estimation residual mean,
    covariance .5*empirical+.5*diagonal+1e-8I,256 fixed antithetic draws.
    Same draws for every query; no dependence on row order or future results.
11. Same empirical Ridge per currency, structure>=30/estimate>=12, otherwise
    same-currency nearest32 fallback. No local hyperparameter tuning.

Every family supplies non-increasing first-cheaper survival probabilities,
expected future-only benefit and expected symmetric benefit at all five h.
Compute each benefit for EACH scenario, then average, not benefit of an average
path. Current known-past price sums enter symmetric benefit only. Numerical
log-ratio guard[-20,20] is only overflow protection and every clipped value is
counted. No realized future gaps, label errors or future path volatility used
as query features. No early stopping on evaluation rows.

## Seventy signal policies and unchanged selection

Each family has p5, mean five-h probability,75% p5 pastCDF+25% expected future5
benefit/knownvol pastCDF,75% p5CDF+25% minimum five-h expected symmetric benefit
/knownvol CDF,50% CNYCDF+50% p5CDF,75% AP4raw+25% p5CDF. Unchanged urgent_cap2
controller and prior63/warmup40 ranks. Six policies*11 families + exact AP3,
AP4, AP5 equal and AP6 selected controls =70. No future week ranking.
Use existing AP4 early2023 joint selector incl all-h lift>=1.3, rates1..2,
positive symmetric benefit lower CI, max2/week and forward>=.8*AP3 eachh.
Write selection before later scorecard. Both early and late periods were
previously studied; no claim of fresh generalization or search-adjusted CIs.

## Evidence and continuation

Persist normalized labels, inputs, all probabilities/benefits/signals, source
hashes, train cutoffs, split dates, neighbour row IDs/weights for global128 and
local32, scenario-support logs and numerical clipping counts. Tests cover
path/label equality with original targets, nonlinear utility expectation,
future corruption, local/global transfer, strict maturity, temporal structure
split, forest leaf distributions, Gaussian determinism and exact controls.
Audit train masks, source timestamps, probability monotonicity, support,
all-h paired20/50-date bootstrap, currency/year and cadence gaps. Retain all
negative results; summary/checkpoint/PDF, tests and checked push ownbranch.
