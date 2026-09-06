# AP43-E registered protocol — continuous-rank specialist substitution

Registered before any AP43 scorecard. Later 2024–2026 is already opened and is
reported only as retrospective development evidence.

## Hypothesis and one fixed candidate

AP42 demonstrated that reusing AP23's final binary policy cannot refill gaps
created by the AP38 core-quality veto: that policy's own historical controller
has already discarded potential substitutes. AP43 therefore uses AP23's saved
causal continuous OOS ranks before its old controller:

- primary opportunity: frozen AP26 core;
- accept core when AP38's mature-history core-quality gate is true or the new
  router's trailing 365-day rate is below 1.0;
- when there is no core, after an 84-day warm-up and while rate <1.05, accept a
  substitute only if AP23 pace rank >0.55 and reserve rank >0.70;
- if no signal exists in the currency-month by day 24, allow the same reserve
  rank >0.70 as a month rescue;
- maximum two signals per currency/ISO week.

All thresholds are inherited from the already frozen AP23 controller; AP43
does not search a grid. The 1.05 replacement pace was registered in AP42 and is
retained unchanged.

## Causality and evaluation

AP23 ranks are causal 250-observation percentiles of OOS expert scores. AP38's
quality bit uses only labels whose publication-h20 maturity precedes the
decision by two calendar days. AP43 state depends only on earlier AP43 choices
of the same currency. Today-effective CBR is the primary reference; h=1 is
validity-only. Early gate is 2023, later reporting is opened 2024–2026, and
strict success requires min lift >2.4 across h=3/5/10/20, per-currency rate
1–2 for every horizon, max two/week, and zero empty complete months. AP37 is
the registered fallback if the early gate fails.
