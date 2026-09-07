# T47 frozen protocol: mature local-pair competence gate

Registered on 2026-09-07 before computing any T47 output or scorecard.

## Question

T46 showed that an own-currency MOEX pair can help AMD/KZT on 2023-2024 and
then fail after a regime change, especially for KZT in 2025. Can a deliberately
small, causal competence gate use only already matured **OOS errors** to admit a
weak local-pair correction, while otherwise returning the CNY-based T5
probability exactly?

This is a post-T46 hypothesis. The T46 open period has already been inspected,
so T47 is a prospective engineering shadow regardless of any 2025-2026 result.
It cannot be promoted into the frozen final-temperature v1 model from this
experiment.

## Frozen inputs and target

- Input packet: T46 `outputs.npz`; the local and baseline probabilities are not
  refit or altered.
- State: 15:30 pre-receipt only.
- Horizons: `h = 1, 3, 5, 10, 20`.
- Target: existing future-only event
  `v[t] <= min(v[t+1:t+h+1])`.
- A row can contribute competence feedback only after its horizon maturity is
  strictly earlier than the quarter origin minus a two-calendar-day embargo.
- Gate decisions are frozen for a whole calendar quarter.

## One fixed, explainable gate

For each currency, horizon and quarter origin:

1. Take only earlier rows where the T46 local head was actually active and its
   prediction was generated causally out of quarter.
2. Sort by date and retain at most the latest 180 mature rows (`long`).
3. Use the latest 60 of those rows as `recent`.
4. The local expert is competent only when all conditions hold:
   - at least 120 long and 40 recent observations;
   - mean paired Brier delta `local - CNY` is below zero on both windows;
   - mean paired log-loss delta is not above zero on the long window;
   - local ROC AUC is no more than 0.005 below CNY on both windows, with both
     classes present in each window.
5. If the gate is competent and the current T46 row is active, mix in exactly
   25% of the T46 probability in log-odds:

   `logit(T47) = 0.75 * logit(T5) + 0.25 * logit(T46)`.

6. Otherwise T47 equals T5 bit-for-bit.

There is no parameter grid, target-driven row-level router, tree, refit or
manual currency exception. The 180/60 windows, 120/40 support, 0.005 AUC
tolerance and 25% weight are fixed before T47 results. The weak weight is an
explicit safety choice because T46 has already demonstrated regime risk.

## Evaluation

- Historical screen: 2023.
- Historical validation: 2024.
- Already-open diagnostics: 2025, 2026 and 2025-2026 for every horizon; they
  are reported but never used for promotion or tuning.
- Report pooled, active-gated, currency, year and currency-year Brier,
  log-loss, ROC AUC, average precision and ECE.
- Report 20- and 50-date circular moving-block paired bootstrap intervals for
  pooled Brier delta.
- Report activation counts and every quarterly gate decision.

A horizon is called a historical pass only if both 2023 and 2024 have:

1. at least one genuinely gated local row;
2. pooled Brier delta below zero;
3. pooled log-loss delta not above zero;
4. pooled AUC delta at least -0.005;
5. gated-row Brier delta below zero;
6. every gated AMD/KZT slice with at least 30 rows has non-positive Brier
   delta; and
7. both block-bootstrap Brier upper bounds are non-positive.

Even a pass remains `production_promoted=false` and
`prospective_shadow_only=true`.

## Mandatory audit

- SHA-256 of T46 outputs, metadata, audit and this registration.
- Exact fallback outside an approved active row.
- All gate feedback dates precede the origin and all maturity dates precede
  `origin - 2 days`.
- Quarterly decisions are constant within each currency/horizon quarter.
- Re-running after corrupting every future target, maturity and T46 probability
  leaves the full prediction prefix unchanged.
- The frozen final-temperature v1 manifest and runtime outputs remain unchanged.
