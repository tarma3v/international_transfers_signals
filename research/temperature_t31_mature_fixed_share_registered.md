# T31 preregistration: mature-only fixed-share ensemble across regime experts

Registered before computing T31 outputs. T30 exposed two frozen rank experts
with opposite regimes: `recent2y_platt_b100` is useful in 2023, while
`all_platt_b050` becomes much stronger in 2024--2026. T31 tests whether a
causal online mixture can adapt using only already matured h20 outcomes rather
than a hindsight calendar switch.

## Frozen inputs and experts

Use the immutable T30 publication-event predictions and exactly three experts:

1. frozen T4 `identity_early`;
2. T30 `all_platt_b050`;
3. T30 `recent2y_platt_b100`.

No expert is refit. T30 source hashes must pass before T31 runs. The stream
starts on 2023-01-01 with uniform weights. One query row is one real CBR
publication/currency event; no weekend holds are duplicated.

## Online update

At each publication date, first admit only earlier feedback batches whose h20
`maturity_ord` is strictly less than `(query_date - 2 days).toordinal()`. For
each newly admitted original publication date, average Bernoulli log-loss over
its currencies, multiply weights by `exp(-eta * loss)`, normalize, then apply
fixed share `w <- (1-gamma) * w + gamma/3`. The same pre-query weights produce
all currency probabilities on that date. Every feedback batch is consumed
exactly once.

Candidate probabilities are arithmetic mixtures of the three expert
probabilities. Frozen grid:

- `eta ∈ {0.25, 0.50, 1.00, 2.00}`;
- `gamma ∈ {0.00, 0.01, 0.05, 0.10}`;
- 16 candidates, deterministic priority eta low-to-high then gamma low-to-high.

## Chronological decision

- Screen: rows in 2023 whose h20 labels mature before 2024 origin minus
  embargo. Feasible means AUC delta versus identity >= 0.02, Brier and log-loss
  lower, ECE delta <= 0.005. Select minimum Brier, fixed priority on ties.
- Validation: exact screen winner on mature 2024 rows. Apply the same gates;
  no second-best substitution.
- Evaluation: open 2025--2026, diagnostic only. Continue the weight stream
  causally with matured feedback; never reset or select using open labels.

If screen or validation fails, `t31_selected` equals identity. Also persist
`t31_candidate` so the exact screen choice remains inspectable even if rejected.
Any successful result remains a prospective shadow, not a production change,
because 2023--2026 have already been inspected in prior experiments.

## Open diagnostic gate

Report overall/year/currency/currency-year Brier, log-loss, ECE, AUC and AP
against identity and publication-event T25. Report paired moving-block
20/50-date intervals for AUC/Brier/log-loss. A formal research pass additionally
requires validation pass, open Brier/log-loss improvement versus both anchors,
ECE delta <= 0.005, AUC delta >= 0.02 versus identity and >= -0.005 versus T25,
both Brier interval upper bounds below zero, and identity AUC interval lower
bounds above zero.

## Required audits

- exact T30 source hashes and candidate columns;
- unique publication/currency keys and complete three-expert rows;
- each consumed feedback batch is earlier than the query, mature before cutoff,
  and consumed once;
- weights finite, non-negative, sum to one and are identical for all currencies
  on a query date;
- independent rebuild of all mixture probabilities, screen winner, validation
  verdict and fallback;
- changing targets whose maturity is after a query cannot change its prefix;
- complete 16-candidate grid, local metric slices and paired interval grid;
- open 2025--2026 never selects eta, gamma, expert set or fallback.
