# T30 preregistration: pre-SVO rank with frozen post-SVO probability map

Registered before computing T30 candidate metrics or outputs. T26--T29 show
that updating an already chosen T25 level on opened history is unstable. T30
tests a different construction: learn the ordering strictly before 2022,
learn only its probability scale on a fixed post-SVO 2022 window, choose one
fully frozen candidate on 2023 and validate it on 2024. Open 2025--2026 is
diagnostic only and cannot choose a fit window, mapping strength or model.

## Frozen information split

- Rank fit cutoff: 2022-01-01 minus the existing two-day embargo; every h20
  label must be fully mature before that cutoff.
- Three rank histories using the same 41 T24 features and logistic C=0.1:
  all available pre-2022, 2018--2021 and 2020--2021.
- Post-SVO mapping window: 2022-04-01 through 2022-11-30, again requiring h20
  maturity before 2023-01-01 minus embargo.
- Candidate screen: all valid 2023 publication events.
- Frozen validation: all valid 2024 publication events.
- Open diagnostic: 2025--2026 publication events.

The unit is one real CBR publication event per currency, not a duplicated
calendar hold. Publication dates, maturity and targets come from the same
causal publication index used by T24--T29.

## Frozen candidate family

For each rank history, fit one positive-slope Platt map on the fixed 2022
mapping window. Minimize summed Bernoulli log-loss plus
`5 * (intercept^2 + (slope - 1)^2)` with deterministic L-BFGS-B bounds
`intercept ∈ [-4,4]`, `slope ∈ [0.05,5]`. Compare raw probability plus four
fixed logit blends between
raw and Platt: beta 0.25, 0.50, 0.75 and 1.00. This gives 15 candidates.

On the 2023 screen a candidate is feasible only if, against the frozen T4
history probability, AUC improves by at least 0.02, Brier and log-loss both
decrease, and ECE increases by no more than 0.005. Select minimum Brier with a
fixed order: all-history, recent-4y, recent-2y; raw then beta low-to-high.

The exact selected candidate passes 2024 validation only if the same four
conditions hold there. If validation fails, the final output is the frozen T4
identity probability. No second-best candidate may replace it using 2024 or
2025--2026.

## Open diagnostic and promotion gate

Report 2025--2026 overall, year, currency and currency-year metrics against T4
identity and T25 publication-event scores. Report paired 20/50-publication-date
moving-block intervals for AUC, Brier and log-loss. Even if all point gates
pass, T30 is at most a prospective shadow because 2024--2026 has already been
inspected in earlier work.

## Required audits

- rank and Platt labels mature before their origins;
- three frozen rank windows and all 15 candidates are present;
- Platt slopes and every blended map are monotone increasing in raw rank;
- 2023 alone rebuilds candidate selection and 2024 alone rebuilds validation;
- corrupting 2025--2026 targets cannot change fits, maps, selection or
  validation;
- one unique currency/publication key, finite bounded probabilities, complete
  local slices, source hashes and paired interval grid;
- publication-event T4/T25 anchors match the corresponding saved query rows.
