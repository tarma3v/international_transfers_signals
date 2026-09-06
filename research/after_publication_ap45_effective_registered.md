# AP45-E registered protocol — conservative dual-gate veto

Registered before any AP45 scorecard. This is retrospective development because
2024–2026 has already been opened.

## Fixed candidate

AP38's mature nonparametric quality gate and AP44's nonlinear meta-CatBoost
produce different estimates of core quality. Reject an AP26 core opportunity
only when **both** gates reject it. Otherwise keep it. The causal trailing-rate
guard is fixed at 1.20 signals/currency/week, an operational safety buffer inside
the requested 1–2 range; below it, no core opportunity may be vetoed. AP37's
late-week/silence fallback, Friday, 10-day silence, day-24 month rescue and hard
two-per-ISO-week cap are unchanged.

No threshold grid is evaluated. AP38 gate, AP44 probability >=0.50 boundary and
all fitted predictions are inherited byte-for-byte. The 1.20 floor is a single
conservative correction after AP41 established that a 1.10 internal rate does
not cover horizon-edge scoring loss.

## Evaluation

Today-effective CBR is primary; h=1 is validity-only. Early gate uses 2023 and
h=3/5/10/20. Opened 2024–2026 is reported only after the early decision. Strict
success requires min lift >2.4, per-currency rate 1–2 at every horizon, max
two/week, and zero empty complete months. AP37 is the registered fallback.
