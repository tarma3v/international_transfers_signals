# AP47-E frozen protocol — joint-survival model with cadence buffer

Registered 2026-09-06 before AP47 signals or scorecards were computed. AP46
passed the frozen 2023 early gate but missed the late per-currency cadence floor:
its retrospective minimum was 0.9476 signal/week. The later period is already
opened; AP47 is a registered retrospective repair, not a fresh validation.

Reuse AP46 probabilities without refitting or changing any feature, label,
model, or decision threshold. Re-run exactly the AP40 sequential controller with
the cadence guard changed from 1.0 to the previously registered AP45 buffer of
1.20 signals/week. Therefore a model-rejected AP26 core opportunity is restored
whenever the currency's trailing-365 selected rate is below 1.20. Friday,
ten-day silence, month-day-24, mature fallback, warmup, and max-two-per-week
rules remain unchanged.

There is one fresh policy and no parameter grid. The value 1.20 is carried
forward from the prior AP45 product rule rather than chosen from AP46 late
scores. AP37 is the registered fallback if the sole policy fails the unchanged
2023 early joint gate. Report all horizons, cadence, symmetric/future-only
benefit, decision changes, 20/50-date paired bootstrap, and future-prefix
causality. `h=1` remains validity-only; historical receipt timestamps and bank
execution remain unvalidated.
