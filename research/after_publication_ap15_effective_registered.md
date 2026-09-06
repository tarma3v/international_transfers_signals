# AP15-E frozen protocol: close the last cadence gap

Registered 2026-09-06 before computing any AP15 early or later metric. AP14's
2024-2026 table is open and may motivate this bounded family, but no AP15
parameter may be selected from AP15 late lift. Reference remains TODAY-EFFECTIVE
CBR after receipt of tomorrow's announced fixing. Receipt time is historically
CALENDAR-ASSUMED; CBR benefit is not executable bank P&L.

## Frozen information and predictor

Reuse AP14's5,755 rows, effective outcomes, publication-h20 maturity cap,
2-day embargo, early2023 selection, later2024-2026 diagnostic, known-down veto
and frozen AP12 ExtraTrees OOS score. h1 is known after receipt and excluded
from selection; unknown horizons are h3/h5/h10/h20. No predictor is refit.

## Five prespecified policies

All policies compute per-currency prior250 causal ranks with warmup40, strict
comparisons and max2 signals per ISO week. The current and future rows never
enter ranks or controller state.

1. `top3125`: fixed score rank>.6875, chosen from AP14's required signal-count
   interpolation, not from AP15 outcomes.
2. `silence14_month24`: original top30; after14 calendar days without a signal,
   allow reserve rank>.80; additionally, from day24 allow one rescue only if the
   currency has no signal in that month and reserve rank>.70.
3. `top3125_month24`: top31.25 plus the same empty-month rescue.
4. `adaptive100`: before84 observed days use threshold.6875; afterwards use
   .675 only when the currency's decision rate over the prior365 observed days
   is below1.0/week, otherwise.70. It sees decisions, not outcomes.
5. `adaptive100_month24`: the same narrow controller plus the empty-month
   rescue. Month rescue cannot override known-down veto or weekly cap.

The month controller is causal: it knows whether the current month has already
received a signal but cannot know whether a later day would have fired. This may
add more than the final number of empty months; report exact additions/removals.

Frozen controls: AP12 top30; AP14 near-cadence silence14, adaptive105, strict
top35 and honest early-selected rolling silence21; AP13 rolling/local primary
and reserve7; AP10 known-z; AP11 hazard; AP1 cap2; known-sign cd3 as benefit
benchmark.

## Selection and success criterion

Select among five fresh policies on early2023 only. Joint gates: minimum lift
over h3/h5/h10/h20>=1.3; every currency rate1..2; max2/week; zero empty complete
months; symmetric-benefit block-bootstrap lower bound>0 on all unknown h;
future-only benefit>=80% of known-sign. Rank passers by minimum unknown-h lift,
then mean lift, then smaller maximum calendar gap. If none pass, apply the same
registered rate/cap-only fallback and label it.

Later success is diagnostic, not selection: each-currency h5 rate>=1, ideally
the same on every unknown-h maturity scope; zero empty complete months;
max2/week; minimum unknown-h lift>2.35. Report paired20/50-date intervals versus
all controls, exact reason counts, currency/year slices and prefix-corruption
invariance for all five policies. Any late success remains retrospective until
future live shadow data with actual receipt and executable bank pricing.
