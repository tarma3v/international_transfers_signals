# AP11-E frozen protocol: predict only the unknown remainder

Registered 2026-09-06 before fitting or viewing AP11 target scorecards. AP10-E
was real progress and is pushed as 981ed3c. Primary reference is TODAY-EFFECTIVE
CBR per user clarification; tomorrow's received fixing is a feature/known first
step. Publication-reference results remain separate diagnostics. Goal active,
no hourly automation.

## Fixed clock, support, target and training

Reuse AP10-E extended panel, market matrix, early2023 selector, later2024-2026
scorecard, corridor-year groups, 18:30 decision and20minute candle delay exactly.
Same5755 event rows; announced_index=current_index+1. Current <= minimum next h
observations for h1/3/5/10/20. If announced<current, every target is known zero
and all AP11 deployable policies veto. If announced>=current, h1 is known one;
models only learn the unknown observations2..h. Do not advertise h1 as forecast.

Same17quarter origins from2022Q3. Use identical full publication-h20 maturity
cap before origin-2calendar days as AP10-E, even though effective target resolves
one observation earlier. This isolates model structure. Only known-next-not-lower
training rows enter conditional learned models; known-next-lower is handled by
the deterministic gate. Same133 announced+market features; no new path data,
clock, threshold grid or late regime selection. Historical CBR receipts remain
CALENDAR-ASSUMED, bank execution unvalidated, later period not fresh.

## Prespecified model packet

All probability curves are [1, P(y3|known y1=1), P(y5|...), P(y10|...),
P(y20|...)] on eligible events and0 on vetoed events. Eight families:

1. Four separate global HistGB classifiers.
2. Same HistGB with deterministic exponential observation weights, half-life
   730calendar days, based only on event date before each origin.
3. Four standardized global logistic classifiers, C=.1.
4. Per-currency standardized logistic, shrunk in probability to global logistic
   by n_local/(n_local+150), fit only when >=60 rows and both classes.
5. Conditional four-interval HistGB hazard for steps2-3/4-5/6-10/11-20;
   at-risk rows stop after first strictly cheaper observation.
6. The same conditional hazard with standardized logistic C=.1.
7. Four global HistGB .25-quantile regressors of minimum remaining log-margin
   vs current price, divided by causal effective vol20 (floor1bp).
8. Global/local Ridge regressors of the same margins, with local probability-free
   prediction shrunk to global by n_local/(n_local+150).

For every family issue h5 and all-h mean scores through unchanged urgent-cap2:
past63 CDF,40warmup,min2days,max2 per ISO week. Always apply known-down veto.
Add only two fixed anchored mixtures:75% past-CDF known_change_z +25% past-CDF
of conditional Hist-h5 or conditional hazard-h5, then the same controller.
No weight tuning.

## Rule controls and selection

- exact frozen AP10 `known_change_z_urgent_cap2` and sign-cooldown3;
- exact AP1 `change_z_r25` mapped by currency/date, using its full historical
  calibration and unchanged firings;
- exact AP1 firings passed through a causal min2day/max2weekly filter;
- known_change_z fixed historical75th percentile, window250/warmup40 plus
  min2day/max2weekly, using AP10 rows only;
- frozen AP10 gated market Hist and survival-h5 diagnostics.

The exact AP1 policy is a required old-rule benchmark but cannot win if its
weekly maximum exceeds2. Select deployable policies by the unchanged AP10 early
joint gates: min lift>=1.3, per-currency average1..2/week, symmetric-benefit
bootstrap lower bound>0, max2/week, future-only ratio>=.8 relative sign-cd3.
Rank by minimum all-h lift, then mean lift. Write selection before opening later.

## Prespecified evidence

All h metrics, cadence and year/currency slices; paired20-date bootstrap for
every AP11 candidate vs AP10 known-z and old exact/AP1-capped controls; selected
50-date sensitivity. Separately compare each family h5 vs mean and two anchored
mixtures vs their components. Conditional probability calibration only on
eligible events and horizons3/5/10/20.

Audit every source hash, mapping, training mask, eligibility, at-risk interval,
failure count, local/global sample count and curve monotonicity. Tests: known
down => all labels/predictions/signals zero; h1 known; no step1 hazard training;
no rows after failure; future feature/label corruption outside maturity; local
fallback/shrink; exact old mapping; controller prefix invariance. Save all
negative results, report/PDF/summary, checked push only ivan-experiments. Packet
does not complete the indefinite goal.
