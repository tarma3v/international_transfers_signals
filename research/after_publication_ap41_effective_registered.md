# AP41-E registered protocol — frequency-protected optimal stopping

Registered before inspecting any AP41 scorecard. AP40 has already opened
2024–2026, so AP41 is a retrospective development experiment, not a fresh
holdout claim.

## Fixed hypothesis

AP40 improved point lift by vetoing weak core opportunities, but its trailing
one-year rate guard of exactly 1.0 produced a scored minimum below the required
1 signal per currency/week. AP41 keeps the same fitted AP40 probabilities and
all AP37 fallback rules, changing exactly one operational constant:

- protect a core opportunity whenever the causal trailing 365-day rate before
  the decision is below **1.10** signals/currency/week;
- use the same 1.10 floor for AP37 fallback activation;
- retain AP40's fixed probability threshold 0.50 and hard Friday, 10-day
  silence, month-end, warm-up and cap-two guards.

The 0.10 buffer is fixed as a conservative operational margin for right-edge
censoring and horizon-specific scoring. It is not selected from an AP41 grid.

## Model, target and causality

Feature set, weekly take-now target, latest-contributing-outcome label maturity,
quarterly origins, two-day embargo, logistic specification, recency weighting,
and current-only feature availability are byte-for-byte inherited from AP40.
No later signal, future outcome, future rank, future rate, or later-opened
scorecard may enter a decision.

## Frozen evaluation and decision

- early gate: calendar 2023, horizons 3/5/10/20 only;
- later report: calendar 2024–2026, opened and therefore retrospective;
- today-effective CBR is the main baseline; h=1 is validity-only;
- target rate 1–2 per currency/week, max two in each ISO week, no empty complete
  currency-months;
- accuracy success: minimum adjusted lift > 2.4 on h=3/5/10/20;
- strict success additionally requires minimum rate >= 1;
- AP37 is the registered early-gate fallback.

All currencies are pooled for the global model; every frequency state and cap
is maintained separately by currency.
