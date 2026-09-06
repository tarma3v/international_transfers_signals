# T20 preregistration: hierarchical probability calibration

Registered 2026-09-06 before computing T20 scores. T19 is already open and is
used only to define the weakness to address. T20 does not change AP37, the
sparse push policy, expected-benefit heads or the T17/T18 availability route.

## Question

Can a calibration map trained only on mature pre-2025 outcomes improve the
meaning of the 0--100 temperature on opened 2025--2026, especially for h10/h20
and currency-specific regimes, without using any future target at prediction
time?

## Frozen data and split

- Input probabilities and routing states: frozen T17 snapshots selected with
  the T18 receipt semantics used by T19.
- Horizons: h=5/10/20. H5 is a strong reference; h10/h20 are the target gaps.
- Scenarios: `calendar_assumed_replay` and `no_same_day_receipt`, reported
  separately.
- Fit origin: 2025-01-01 00:00 Moscow.
- Training rows: query dates before the fit origin whose full target maturity
  is strictly earlier than fit origin minus the existing two-day embargo.
- Evaluation rows: 2025-01-01 through the frozen T19 end date. This period is
  already open and remains diagnostic, not a fresh holdout.
- The same 20 T19 clocks are retained. No clock is added or removed after
  scores are inspected.

## Frozen candidates

1. `identity`: unchanged frozen probability.
2. `global_platt`: logistic regression on the clipped base logit, `C=1.0`.
3. `fixed_logit_shrink_80`: `sigmoid(0.8 * logit(base))`, with no fit.
4. `hierarchical_beta`: the primary candidate. Logistic regression with
   `C=0.05`, using `log(p)`, `-log(1-p)`, currency intercepts, currency-specific
   logit slopes and regime intercepts. Regime is the exact causal tuple
   `phase|source_kind|confidence|freshness`; unseen categories map to zero.

All learned candidates fit separately for `scenario × clock × h`. No mapping
uses evaluation labels, same-day outcomes or later clocks. If a training group
has fewer than 200 rows or only one class, the candidate falls back to identity
and records the reason.

## Metrics and gates

Report Brier, log-loss, ECE, AUC and average precision overall, by currency and
by currency-year. Discrimination is expected to remain essentially unchanged
because calibration should be monotone within a group; a changed AUC is a
diagnostic for subgroup reordering, not the main objective.

The primary candidate passes a state only when, versus identity:

- mean Brier delta is below zero;
- both 20-date and 50-date paired moving-block 95% CI upper bounds are below
  zero;
- log-loss delta is below zero;
- ECE is not worse by more than 0.01.

Additionally report how many currency-year-clock-h rows have ECE above 0.08.
No model is promoted from the opened period. A useful result becomes a frozen
prospective challenger; a failed result remains a documented negative result.

## Causality audit

- verify every training target matured before the embargo cutoff;
- verify every selected snapshot/source timestamp is no later than query time;
- corrupt evaluation targets and assert fitted coefficients/predictions on the
  earlier prefix do not change;
- save source hashes, row counts, fallback reasons and exact output tables.
