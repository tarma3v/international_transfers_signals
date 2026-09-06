# AP23-E frozen protocol: mature-only competence router

Registered 2026-09-06 before AP23 scores, signals or scorecards are computed.
AP21 showed that rolling ExtraTrees is a strong primary expert and CatBoost mean
utility is a useful cadence expert. AP22 label-free rolling/local consensus did
not transport. AP23 therefore tests one new idea: adapt expert weights only from
their fully matured historical top-rank precision, never from the current or
future target.

Frozen quarterly-OOS inputs are AP13 rolling-730 ExtraTrees, AP13 partially
pooled local ExtraTrees, AP18 full/recent50 and AP19 CatBoost mean utility.
For every input compute its past-250 per-currency percentile with warmup40.
At decision date d, an expert's competence may use only eligible rows j with
date[j] < d, publication-target mature20[j] < d-2 calendar days, finite
h3/h5/h10/h20 outcomes, and the expert's causal rank[j] >.70. The response is
mean(y3,y5,y10,y20), so known h1 never enters routing.

For each expert and date, compute a global precision shrunk to prior .5 with
strength40, then a currency precision shrunk to that global precision with
strength40. Only rows in the fixed trailing window enter. Soft weights are the
two-expert softmax of these precisions with eta25; hard weights select the higher
precision, breaking ties toward the first expert. Missing inner ranks fall back
to the available one. This is online supervised aggregation, not post-hoc late
selection.

Five fresh pairings are fixed:

1. `soft730_both`: competence-blend rolling/local primary and AP18/Cat pace;
2. `soft365_both`: the same with a 365-day competence window;
3. `hard730_both`: hard 730-day expert selection for both roles;
4. `soft730_primary`: competence rolling/local primary with fixed Cat pace;
5. `soft730_pace`: fixed rolling primary with competence AP18/Cat pace.

All five pass through the exact AP21/AP17 policy: outer past-250 ranks; primary
rank>.70; after84 days and while trailing365 rate<1/week, pace rank>.55 jointly
with frozen reserve rank>.70; day>=24 reserve rescue for an empty currency-month;
known-down veto; maximum2 signals per ISO week. No threshold is changed.

Evaluation is unchanged: today-effective CBR reference, already known next CBR
fixing, h3/h5/h10/h20 for selection and h1 for validity only, early-2023
selection and opened-2024-2026 diagnostics. Early gates require min lift>=1.3,
each-currency rate1..2/week, zero empty complete months, max2/week, positive
symmetric-benefit CI and future benefit at least80% of simple known-sign.

Audit independently reconstructs maturity masks, expert ranks, precisions,
weights, state transitions and signals. Corrupting every expert score, target and
maturity date from 2025 onward must leave the earlier prefix unchanged. Compare
with frozen AP17/AP18/AP21 controls using paired20/50-date bootstrap. Target is
strict min unknown-h lift>2.4 without cadence loss. Historical receipt timestamps
remain calendar-assumed, bank execution is not validated and the later period is
not a fresh holdout.
