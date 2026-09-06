# T25 preregistration: anchor-preserving map for the T24 h20 rank

Registered 2026-09-06 before computing T25 outputs. T24 found a strong causal
premarket h20 rank but its ordinary Platt probability failed the pre-2025
selection gate. T25 tests whether that rank can be imported while preserving
the frozen anchor's probability level. The opened 2025--2026 period remains
diagnostic and cannot select a map or weight.

## Frozen data and rank

- Refit exactly the T24 compact logistic on publication rows before 2024 whose
  h20 outcomes mature before the origin minus the two-day embargo.
- Map each calendar query to the latest publication no later than it.
- Mapping reference: mature calendar queries in 2024-H1.
- Candidate selection: mature 2024-H2 queries resolving before 2025 minus
  embargo.
- Evaluation: opened 2025--2026, 00:15 history-only state. The output may be
  held at 06:00/09:15 until a fresher source appears but duplicated clocks are
  not independent evidence.

## Frozen maps

1. `identity_early`.
2. `residual_a001/a0025/a005/a010/a020/a040`: regress T24 compact logit on
   frozen logit using 2024-H1 only, standardize that residual on H1, then add
   the fixed alpha times residual-z to frozen logit.
3. `copula_global`: map compact ranks to the empirical H1 distribution of
   frozen probabilities.
4. `copula_currency`: same empirical map separately by currency, with global
   fallback.
5. `daily_permute`: within each date, assign that day's five frozen
   probabilities in compact-rank order. The probability multiset and mean for
   the date remain exactly unchanged.
6. `daily_permute_blend50`: equal logit blend of identity and daily-permute.
7. `copula_global_blend50`: equal logit blend of identity and global copula.

No candidate uses a target when mapping evaluation. On 2024-H2, a candidate is
feasible only if AUC is above identity, Brier and log-loss are no worse, and ECE
is no more than 0.01 worse. Select maximum AUC with fixed tie order residual
small-to-large, permute blend, permute, global blend, global copula, currency
copula. If none passes, retain identity.

## Evaluation gate

Promotion requires positive AUC delta, negative Brier/log-loss deltas, ECE delta
<= 0.01, and both 20/50-date paired moving-block intervals strictly positive
for AUC and negative for Brier. Report currency/year/local calibration and exact
daily-mean preservation. Any pass is only a prospective shadow challenger.

## Required audits

- T24 model fit and all mapping/selection labels mature before cutoffs;
- publication/source timestamps no later than query;
- H1 alone determines residual and copula mappings;
- daily-permute exactly preserves the five-value multiset per date;
- changing H2/evaluation targets leaves all maps, selection inputs and
  predictions unchanged (selection outcome may depend on H2 by design, never
  evaluation);
- probabilities bounded; hashes, maps and outputs persisted.
