# AP12-E frozen protocol: conditional ranking and cadence repair

Registered 2026-09-06 before fitting any AP12 model or opening any AP12
scorecard. AP11-E is already opened and therefore its models are controls, not
fresh challengers. Primary reference remains TODAY-EFFECTIVE CBR. The received
tomorrow fixing is a known feature and the first future observation. Historical
receipt at 18:00 Moscow remains CALENDAR-ASSUMED; executable bank prices remain
unvalidated.

## Frozen support, split, target and information

Reuse AP11's 5,755 rows, 18:30 decision, 20-minute market delay, source hashes,
shared publication-h20 maturity cap, 17 quarterly walk-forward origins,
early-2023 selector, later-2024--2026 opened scorecard, and corridor-year
adjustment groups exactly. `announced_index=current_index+1`. If the announced
fixing is below today's effective fixing, all policies veto. Conditional models
train only on rows where it is not below and learn only observations 2..h.

Report h1 as a deterministic validity check, but do not use h1 to rank models:
the primary unknown-horizon set is h3/h5/h10/h20. This correction is fixed
before fitting AP12 because AP11 showed that its all-h minimum was dominated by
the already-known h1.

## New scores

Fit the following once per frozen quarterly origin, using only the shared
mature training prefix and the same AP11 133-feature market matrix.

1. `extra_h5`: ExtraTrees probability for conditional y5, 400 trees,
   max_depth=8, min_samples_leaf=25, max_features=.6.
2. `compact_hist_h5`: HistGradientBoosting probability for conditional y5 on a
   fixed compact set: known change/buffer; announced and effective returns,
   volatilities and ranges; peer mean/std and local-minus-common; source ages;
   calendar/regime flags; currency one-hot. Same AP11 Hist parameters.
3. `compact_extra_h5`: the fixed ExtraTrees model on the same compact set.
4. `local_hist_h5`: per-currency compact Hist probability, shrunk to the global
   compact Hist probability by n_local/(n_local+150); fall back globally below
   100 usable rows or one class.
5. `first_failure_extra_h5`: ExtraTrees multiclass model for the first failure
   bucket among steps2--3,4--5,6--10,11--20 or survival through20. Convert class
   probabilities to monotone survival and use its h5 survival probability.
6. `known70_hazard30`: fixed 70/30 blend of causal per-currency past-CDF ranks
   of known-change-z and the frozen AP11 conditional hazard-h5 score. This is a
   new fixed score but uses no refitting or late-tuned weight.

Also carry eight frozen controls without alteration: AP10 known-change
urgent-cap2, AP10 known-next-not-lower cooldown3, AP10 gated market Hist,
AP11 hazard-h5 urgent-cap2, AP11 early-selected margin-Ridge-mean, AP1 exact
r25, AP1 exact capped, and fixed known-change top25 capped.

## Prespecified online policies

For each of the first five new model scores, create only:

- prior-250 strict top30%, warmup40, at most2 per ISO week, no extra day gap;
- prior-250 strict top30%, warmup40, at most2 per ISO week, with one full
  calendar day between signals (date difference at least2).

For raw known-change-z and `known70_hazard30`, use the same two policies and add
prior-250 top27.5% with no extra gap. This yields 16 fresh policies and 24 total
policies (5x2 + 2x3 plus eight fixed controls). The controller processes rows
chronologically and never sees the remainder of the current week. The no-gap
variant means distinct announcement events may both fire; it does not permit
more than two in an ISO week.

## Frozen selection and evidence

The early joint gates are: minimum adjusted lift across h3/h5/h10/h20 >=1.3;
per-currency average cadence across those horizons between1 and2/week;
symmetric-benefit bootstrap lower bound >0 across h3/h5/h10/h20; max2 signals
in any ISO week; and minimum future-only benefit at least80% of sign-cooldown3.
Among the 16 fresh policies only, rank passers by minimum unknown-horizon lift,
then mean unknown-horizon lift, then lower signal rate. Frozen controls are
reported but cannot become a fresh winner. Write selection before opening later
scorecards.

On later data report all h, paired circular20-date block bootstrap against AP10
known-z, frozen AP11 hazard and AP1 exact/capped controls; 50-date sensitivity
for the selected candidate; per-year/currency slices, cadence clustering and
probability Brier/log loss on eligible h5 rows. A later point winner is only a
diagnostic unless it was early-selected. Audit every training mask, feature
subset, class mapping, source hash, causal controller prefix, veto, weekly cap
and rebuilt score/signal. Preserve all negative results. This packet advances
but does not complete the indefinite research goal.
