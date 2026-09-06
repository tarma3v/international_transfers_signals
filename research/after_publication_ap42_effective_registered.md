# AP42-E registered protocol — specialist substitution router

Registered before inspecting any AP42 scorecard. The 2024–2026 interval has
already been opened by earlier packets, so this is retrospective development,
not a fresh holdout.

## Fixed hypothesis

AP38 showed that its mature-history core-quality gate can remove a weak tail,
but its calendar-restricted fallback cannot replace enough rejected decisions.
AP32 showed that a separate global AP23 expert can supply deficit decisions at
useful precision. AP42 therefore separates the jobs:

1. AP26 remains the primary/core opportunity.
2. A core opportunity is accepted when the frozen AP38 mature-quality gate is
   true, or when the router's own causal trailing rate is below 1.0.
3. If there is no core opportunity, the frozen AP23 decision may substitute
   whenever the router is warmed up and its trailing rate is below **1.05**.
4. The prior AP37 late-week/silence restriction is removed for this fallback;
   AP23's own frozen score and decision gate still apply.
5. Maximum two signals per currency/ISO week remains hard.

The 1.05 fallback pace is fixed before the AP42 scorecard. It is the midpoint
of the requested 1–2 range's low-rate operating neighbourhood and provides a
small right-edge/scoring buffer; no threshold grid is evaluated.

## Information and evaluation

- Every AP26/AP23 decision and AP38 quality bit was produced OOS from mature
  history with the existing two-day embargo.
- Rate, warm-up and weekly-cap state use only prior AP42 decisions of that
  currency.
- Today-effective CBR is the primary reference; h=1 is validity-only.
- Early gate: 2023 and h=3/5/10/20. Later report: opened 2024–2026.
- Strict target: minimum lift >2.4, every currency/horizon rate >=1 and <=2,
  max two per ISO week, and no empty complete currency-month.
- AP37 is the registered early-gate fallback.

This packet changes only the decision router. It does not refit or tune either
expert on the opened later interval.
