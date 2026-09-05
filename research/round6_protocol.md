# Round 6 protocol: causal rate control and resolved-outcome state

Frozen on 2026-09-05 before reading round-six 2025/2026 results.

## Goal

Without the next CBR rate, achieve future-only `fav_h5` lift at least 1.30 while
emitting 1.00--2.00 alerts per currency-week. Predictions, threshold updates
and model refits must be causal. Results remain research-retrospective because
the available 2024--2026 period was inspected in earlier rounds.

## Packet A: online weekly rate controller

The input score architectures are fixed before this packet: the round-five
quarterly reset HistGradientBoosting, the multiscale anchor, and their 50/50
rank blend. No model is refitted for policy search.

For every currency independently, the controller:

- processes publications in date order;
- resets its score-history scale at each calendar quarter, matching model
  refits;
- compares today's score only with earlier scores in the same quarter;
- keeps the number of alerts already emitted in the current ISO week;
- uses a strict score threshold early in the week and a lower threshold on
  Thursday/Friday if the week has no alert;
- never exceeds a frozen weekly cap of one or two alerts.

History windows 10/20/40/60, early percentile thresholds
75/80/85/90%, late thresholds 0/30/50/65%, late weekdays Thursday/Friday and
weekly caps one/two are screened on 2024. Each architecture selects one policy
by lift subject to overall frequency 1.00--2.00, every currency at least 0.80,
every non-empty quarter at least 0.70 and positive future-only benefit. The
unchanged architecture/policy is confirmed on 2025 and audited on 2026.

## Packet B: resolved-outcome state features

For a row at date `t`, an older `fav_h5` outcome may be used only when its
actual fifth-publication reach date is strictly earlier than `t`. Features are
computed separately per currency and for the five-currency panel:

- last 1/2/3/5/10 already resolved outcomes;
- rolling resolved hit rates over 5/10/20/40/60 outcomes;
- time since the last resolved positive outcome;
- exponentially weighted resolved hit rate;
- panel mean, dispersion and breadth of the most recently resolved outcomes;
- changes between short and long resolved hit rates.

These features are appended to the fixed 258-feature round-five matrix.
Quarterly post-24.02.2022 HistGradientBoosting, ExtraTrees, regularized logistic
regression and simple rank ensembles with the anchor are screened on 2024.
Packet-A rate controllers and ordinary causal rolling thresholds are eligible.
Architecture/policy choices remain unchanged in 2025 and 2026.

## Success and reporting

A formal pass requires in 2025 and 2026 separately:

- lift at least 1.30;
- 1.00--2.00 alerts per currency-week overall;
- at least 0.80 per currency;
- at least 0.70 in every calendar quarter with data;
- positive future-only benefit.

Always report annual, currency and quarterly breakdowns, four-week block
bootstrap uncertainty, circular-shift multiplicity control, exact signal
policy, and a physical future-truncation leakage audit. A 2026 result cannot
rescue a candidate that failed the frozen 2025 confirmation gate.

## Packet C: target-aligned frequency repair

Frozen after packets A/B failed 2025 and before evaluating this policy on
2025/2026. The original round-five 50/50 reset-Hist/anchor blend missed the
desired lower rate by only 0.09 alerts per currency-week in 2025. Rather than
introduce another architecture, choose its alert policy solely on the 2024
screen by maximum lift subject to 1.00--2.00 overall frequency, at least 0.80
for every currency and positive future-only benefit. This selects target rate
20% with a causal trailing 250-publication cutoff (`rate=.20`, `rolling=250`,
`cooldown=0`): 2024 frequency 1.190 and lift 1.317. The exact policy is now
frozen for 2025 confirmation and 2026 audit. Neighbouring rates may be shown
only as post-hoc sensitivity and cannot replace this selected policy.

## Packet D: post-2022 release-aware macro models

Frozen after packet C failed and before reading packet-D 2025/2026 results.
Reuse the already archived external data with its original availability rules:
RUONIA uses the explicit CBR update timestamp, the key rate uses its effective
date, Brent is delayed five calendar days and the broad dollar index ten days.
The fixed lags are deliberately conservative, but the FRED/EIA files are
latest-vintage rather than point-in-time vintage; external results therefore
remain sensitivity evidence even when the model information flow is causal.

Quarterly post-24.02.2022 HistGradientBoosting, ExtraTrees, XGBoost and
CatBoost variants are trained on macro-only, compact+macro,
trajectory+macro and resolved-state+macro matrices. Fixed rank ensembles with
the multiscale anchor and the round-five reset model are included. The same
predeclared rolling and causal weekly policies are selected on 2024 under the
1.00--2.00 overall, 0.80 per-currency, 0.70 per-quarter and positive-benefit
constraints, then frozen for 2025 confirmation and 2026 audit.

## Packet E: trusted CBR-only macro ablation

Frozen after packet D produced an aggregate lift/rate pass but before running
this ablation. Remove Brent, the broad dollar index and every derived/missing
feature associated with them. Retain only RUONIA fields with explicit CBR
`DateUpdate`, the effective key rate, their causal changes/spread and the
original past-only CBR currency features. Train full and compact quarterly
HistGradientBoosting/ExtraTrees models and fixed 25/50% anchor blends. Select
policy on 2024 with the same rate and cadence constraints; confirm unchanged on
2025 and audit on 2026.

## Packet F: weekly controller for the external challenger

Frozen at the same point. Take the already fitted
`macro_full_hist_anchor50` score from packet D and restrict policy selection to
the causal weekly controllers predeclared in packet A. Select on 2024 by the
minimum of aggregate and minimum-currency lift subject to 1.00--2.00 overall,
0.80 per currency, 0.70 per quarter and positive benefit. Apply the exact
policy unchanged to 2025 and 2026. This tests whether the external score keeps
its lift after eliminating zero-alert quarters and quarterly bursts.

## Packet G: broad official-CBR reference panel

Frozen after packet E cleared the literal annual lift/rate goal but before any
packet-G 2025/2026 scores are read. Extend the information set using only the
official Bank of Russia currency directory and historical-rate endpoint. The
five target corridors are excluded from the reference panel. Reference
currencies are selected without target labels: liquid/major ISO currencies
with at least 65% publication-date coverage relative to USD over the available
period. Every CBR observation is divided by its row-specific nominal.

For a target row dated `t`, the reference panel is joined as-of using values
whose CBR date is no later than `t`. Predeclared target-free features comprise
1/2/5/10/20/60-publication log changes, 20/60-publication volatility, equal-
weighted mean/median/breadth/dispersion of reference-currency changes, the
same statistics after removing the USD/RUB move, and target returns relative
to those common factors. No fitted factor weights, target correlation screen,
next CBR rate, or future-filled values are allowed.

Quarterly post-24.02.2022 HistGradientBoosting and ExtraTrees variants augment
the trusted packet-E matrices. Fixed 25/50% anchor blends are eligible. Policy
is selected on 2024 with the existing annual rate/corridor constraints and is
then frozen for 2025 confirmation and 2026 audit. One fixed 50/50 rank blend
of the broad-panel HistGradientBoosting score and the already frozen packet-E
CBR-macro score is also eligible. A physical truncation test
must prove that changing all reference observations after a cutoff cannot
change any feature row at or before that cutoff. The broad-panel result is a
methodological pass only if its official CBR provenance, chosen currency list,
coverage and SHA-256 payload digest are archived.

## Packet H: weekly stabilization of the broad-panel score

Frozen after packet G revealed a strong but frequency-drifting
`broad_full_extra` score and before evaluating any new packet-H policy on
2025/2026. Do not refit or alter that score. Restrict the policy family to the
causal weekly controllers already declared in packet A. From the 2024 screen,
keep policies with 1.00--2.00 overall frequency, at least 0.80 per currency,
at least 0.70 per quarter and positive forward benefit; select the policy
maximising `min(overall lift, minimum-currency lift)`, then overall lift and
quarter frequency. Apply it unchanged to 2025 and 2026. This packet tests the
specific diagnosis that packet G found useful ranking information but an
unstable rolling score scale; no 2025/2026 retuning is permitted.

## Packet I: high-confidence/baseload score hybrid

Frozen after packet H stabilized cadence but destroyed lift, and before any
packet-I 2025/2026 evaluation. Treat the frozen `broad_full_extra` output as a
high-confidence opportunity score and frozen packet-E
`cbr_macro_full_hist_anchor50` as the stable baseload score. Construct
per-currency calibration-rank blends with broad-score weights 25%, 50% and
75%; include the unchanged components as references. No model is retrained.
For each blend, select one policy on 2024 from the already declared rolling
and weekly grid using the existing feasibility rule and robustness objective.
Freeze it for 2025/2026. Because packet I was motivated after inspecting
packet-G 2025/2026 behaviour, its later-period figures are protocol-controlled
retrospective evidence, not a pristine holdout claim.

## Packet J: causal trailing-rate baseload regulator

Frozen after packet I passed the annual average target but retained a nearly
empty 2026Q2, before packet-J 2025/2026 evaluation. Keep the packet-G primary
score and policy (`broad_full_extra`, rolling 30%, 250 observations) and the
packet-E baseload score and policy (`cbr_macro_full_hist_anchor50`, rolling
35%, 120 observations) unchanged. A primary candidate fires subject to a cap
of two signals per ISO week. A baseload candidate may fire only when the
number of already emitted signals in the preceding 21/35/56 calendar days is
below a floor of 0.75/1.00/1.25/1.50 per week; fallback activation weekdays
Monday/Wednesday/Thursday are screened. The current row is excluded from the
trailing-rate calculation.

Select on 2024 by `min(overall lift, minimum-currency lift)` subject to 1--2
overall frequency, at least 0.80 per currency, at least 0.90 in every quarter
and positive forward benefit. Freeze the regulator for later years. This is
a signal-level mixture: low-score baseload rows cannot dilute the primary
ranking unless the observable past alert cadence is already deficient.

## Packet K: activity-aware fallback shutoff

Frozen after packet J showed that a long trailing-rate deficit caused stale
fallback alerts at the start of a newly active regime, before packet-K later-
period evaluation. Use the frozen packet-I 75/25 hybrid score with its frozen
rolling 35%, 250-observation policy as primary; use the same packet-E baseload
as fallback. Over 14/21/28 prior calendar days, track both primary-candidate
activity and total emitted alerts, strictly excluding the current row. Permit
a fallback only when primary activity is below 0.50/0.75/1.00 per week and
emitted cadence is below 0.75/1.00/1.25 per week. Screen fallback start on
Monday/Wednesday/Thursday and cap at two alerts per ISO week.

Select solely on 2024 with the packet-J constraints and robustness objective,
then freeze. The dual gate is intended to switch fallback off promptly when
the high-confidence regime returns instead of waiting for a long cadence
deficit to decay.

## Packet DK: delayed multi-horizon weighting of the perpetual expert

Frozen after the packet-DJ paired audit established that the lagged MOEX
perpetual-futures expert contains significant fresh information but that a
fixed minimum geometry does not significantly beat the incumbent. Keep the
incumbent, packet-DH `futures_extra`, their stale-20 control and the fixed
rolling signal policy unchanged. Map incumbent and futures scores to causal
same-currency trailing percentile ranks over 250 rows with a minimum history
of 20.

For each signal date, update expert losses only from earlier target outcomes
whose own horizon reach date is **strictly earlier** than the signal date.
Each resolved `(row, horizon)` event contributes an equal-weight Brier loss;
all five official horizons `1/3/5/10/20` participate. Screen exactly twelve
online mixtures on 2024: global, local-currency and hierarchical loss scopes;
trailing loss windows 250 and 1,000 resolved events; and exponential-weight
learning rates 2 and 5. The hierarchical estimate shrinks the local loss mean
toward the global mean with fixed prior strength 250. A static 50/50 rank
blend and the unchanged components are references. Apply the 2024-selected
architecture unchanged to 2025/2026 and to the stale-20 expert.

The selection objective is the maximum worst official case lift across all
five horizons, followed by mean lift, subject to positive symmetric and
future-only benefit at every horizon. A physical label-corruption test must
show that changing every outcome whose horizon reach is after a cutoff cannot
change any mixture score on or before that cutoff. Report the selected
incumbent-weight path so regime adaptation remains auditable. Because the
expert components' later performance was already inspected before this packet,
the result is protocol-controlled retrospective evidence, not a pristine
holdout claim.

## Packet DL: paired audit of the delayed perpetual weighting

Frozen after packet DK selected `online_global_w250_eta5` as the best member
of the predeclared online-mixture family on 2024, before computing any paired
uncertainty comparison. Keep that candidate, the incumbent, the matched
stale-20 online system, the policy and every prediction unchanged. On the
combined 2025--2026 interval, compare online versus incumbent and fresh versus
stale at each official horizon using identical moving four-week block draws.
Apply Holm correction separately to lift, symmetric-benefit and future-benefit
differences across the ten `(comparison, horizon)` tests.

Also compare the draw-wise minimum and mean lift over the five horizons. The
online candidate can replace the incumbent only if its point minimum is
higher, the paired 95% lower bound for minimum-lift improvement exceeds zero,
annual h5 lift and frequency pass in both years, combined minimum-currency h5
lift is at least 1.30, and both benefit definitions are positive at all five
horizons. Failure of this gate is recorded rather than repaired with another
post-hoc weight choice.

## Packet L: delayed-feedback online expert mixture

Frozen after packet K showed that hand-written fallback gates still sacrifice
too much ranking quality, before packet-L 2025/2026 evaluation. The frozen
expert pool is `broad_full_extra`, `broad_compact_tree_consensus`,
`broad_factor_hist_anchor25`, packet-E CBR macro without anchor, and the
multiscale anchor. Convert each expert to a percentile rank against the prior
calibration year separately by currency. Combine ranks with exponentially
weighted Brier-loss experts. Update an expert loss only once that row's full
`h=5` reach date is observable; current and unresolved outcomes cannot affect
weights.

Screen global, local and hierarchical loss scopes, learning rates 2/5/10/20
and forgetting factors 0.97/0.99/1.00 on 2024, together with the existing
causal alert-policy grid. Selection requires 1--2 overall alerts, at least
0.80 per currency, at least 0.70 per quarter and positive benefit, maximizing
the minimum of overall and minimum-currency lift. Freeze architecture and
policy for 2025/2026. This revisits Online Hedge with the new broad-CBR experts
and an explicit delayed-feedback audit; it does not reuse the old post-hoc
headline result.

## Packet M: direct within-regime ranking objectives

Frozen after packet L demonstrated that average Brier loss is misaligned with
top-tail alert lift, before packet-M 2025/2026 evaluation. Use the broad-CBR
compact and full matrices with quarterly post-24.02.2022 refits. Train global
XGBoost rankers on contiguous `currency x calendar-quarter` or `currency x
calendar-month` query groups. Predeclared relevance targets are binary
`fav_h5`, the 0--5 count of future publications no cheaper than the current
rate, and within-query deciles of forward-only benefit. Objectives are
pairwise ranking for all relevance types and NDCG for the non-negative ordinal
target. Hyperparameters are fixed across variants.

All training labels must have their fifth-publication reach date strictly
before the refit quarter. Query-group sorting and relevance transformation use
training rows only. Screen alert policy on 2024 with the existing constraints
and robustness objective, then freeze for later years. This packet optimizes
relative ordering of actionable days rather than whole-sample probability
calibration.
Fixed 75/25 per-currency rank blends of each ranker with the multiscale anchor
and, separately, the packet-E CBR baseload are eligible; no fitted blend
weights are used.

## Packet N: resolved-performance weekly expert router

Frozen after packet M exposed a 2026Q2 failure of the benefit ranker while the
CBR baseload remained positive, before packet-N router evaluation. The frozen
experts and their frozen policies are: packet-M benefit-ranker+25% anchor
(rolling 22%, 60 observations), packet-I broad75/baseload25 (rolling 35%, 250),
and packet-E CBR baseload (rolling 35%, 120). At the start of each ISO week and
for each currency, choose one expert for the whole week using only the binary
outcomes of that expert's older would-have-fired signals whose `h=5` reach
date is already observable.

Screen trailing 10/20/40/80 resolved expert signals, beta-prior strengths
5/20/50 and global/local/hierarchical reliability scopes on 2024. Initialize
ties deterministically in the listed order. For later evaluation, carry all
resolved router history forward from 2024; never reset with information from
the target year. Select with the existing lift/rate/corridor/quarter
constraints. A physical unresolved-label corruption test must leave every
past routed decision unchanged.

## Packet O: purged causal score stacking

Frozen after packet N showed that week-level expert selection cannot isolate
the 2026Q2 point failures, before packet-O later-period evaluation. Build a
meta panel from strictly out-of-sample base scores available from 2023 onward:
benefit-ranker+anchor, broad ExtraTrees, broad75/baseload25, CBR baseload and
ordinal-NDCG/baseload. For every expert and currency, transform the current
score to a percentile against at most 250 strictly earlier scores; add expert
mean, dispersion, range, pairwise differences and fixed 70/80/90% tail flags.
No current or future target
enters this transformation.

Quarterly meta refits use only rows whose `h=5` reach date is before the refit
quarter. Predeclared models are regularized logistic regression on expert
features, HistGradientBoosting on expert+core features, and Hist/ExtraTrees on
expert+core+resolved-outcome-state features. Base-model scores must themselves
be prequential for every meta-training row. Screen meta alert policy on 2024
under the existing constraints and freeze it for 2025/2026. This is stacked
generalization, not in-sample fitting to base-model predictions.

## Packet P: fixed classification/benefit score consensus

Frozen after packet O improved annual `fav_h5` lift but had slightly negative
2025 forward benefit, before packet-P later-period evaluation. Blend the frozen
packet-O `stack_resolved_extra` score with the frozen packet-M
benefit-ranker+anchor score at 25/75, 50/50 and 75/25 per-currency calibration
ranks. Add one fixed three-way 50/25/25 blend of stacking, benefit ranker and
packet-E CBR baseload. No weights are fitted. Select alert policy on 2024 with
the existing frequency/cadence constraints, positive forward benefit and
robustness objective, then freeze. This explicitly trades classification lift
against attainable future monetary benefit.
The stack has no 2023 calibration score. Therefore its 2024 component rank is
computed causally against strictly earlier 2024 stack scores; components with
a prior calibration block continue to use that block. This expanding-rank
fallback is fixed before the corrected packet-P evaluation.

## Packet Q: sparse baseload top-up for the multi-objective score

Frozen after corrected packet P showed positive but sparse 2026Q2 signals,
before packet-Q later-period evaluation. Keep every primary signal from the
50/50 stack/benefit blend and its frozen rolling 22%, 60-observation policy.
Permit at most one additional packet-E baseload signal per currency-week only
when, over the strictly previous 14/21/28 calendar days, primary-candidate
activity is below 0.50/0.75/1.00 per week and total emitted cadence is below
1.00/1.25/1.50. Screen fallback start Monday/Wednesday/Thursday. The baseload
itself retains rolling 35%, 120 observations.

Select on 2024 with 1--2 overall and >=0.80 per-currency frequency, >=0.90 in
each quarter, positive benefit and the usual lift robustness objective. There
is no cap on primary signals; the fallback-only weekly cap avoids repeating
packet K's mistake of deleting clustered high-confidence primary signals.

## Packet R: refit-aligned quarter-reset threshold

Frozen after packet Q suggested that low quarterly cadence is partly a score-
scale discontinuity at quarterly model refits, before packet-R later-period
evaluation. Keep the packet-P 50/50 stack/benefit score unchanged. For each
currency, reset threshold history exactly when the base model refits at the
start of a calendar quarter. Fire only after 5/10/20 scores have accumulated
in the current quarter and the current score exceeds the causal trailing
20/40/60/120-observation percentile for target rates 18/20/22/25/30/35%.
Append the current score only after the decision. There is no forced alert,
fallback or future quarter calibration.

Select on 2024 under 1--2 overall, >=0.80 per-currency and >=0.70 per-quarter
frequency with positive benefit, maximizing usual lift robustness. Freeze the
policy for 2025/2026. A future-score corruption test must leave all decisions
at or before the cutoff unchanged.

## Packet S: dual-scale threshold with sparse reset fallback

Frozen after packet R showed that refit-aligned reset repairs cadence but adds
weak 2025Q4 rows, before packet-S later-period evaluation. On the same frozen
packet-P score, keep every primary signal from the ordinary rolling 22%,
60-observation policy. The reset policy is fixed to packet R's 35%, 20-
observation, 10-row warm-up. It may add at most one non-primary signal per
currency-week only when the strictly previous 14/21/28/42 days contain fewer
than 0.50/0.75/1.00 primary signals per week and fewer than 0.75/1.00/1.25
total emitted signals per week; fallback starts Monday/Wednesday/Thursday.

Select on 2024 requiring 1--2 overall, >=0.80 per currency, >=0.90 per quarter
and positive benefit, maximizing usual lift robustness. Freeze for later
years. This preserves all high-lift cross-quarter calibrated signals while
using the within-refit scale only as a sparse, observable-cadence fallback.

## Packet T: target-free refit-scale score normalization

Frozen after packet S showed that OR-ing a reset fallback repairs some cadence
but admits too many low-quality rows, before packet-T later-period evaluation.
Keep the frozen packet-P `stack50_benefit50` score and change only its
target-free scale. For each currency and row form: (a) a strictly expanding
percentile rank against earlier scores in the same calendar quarter, with no
decision-row admission; and (b) a same-publication-date cross-sectional rank
among the five target currencies, which are all observable simultaneously.
Until 10 same-quarter observations exist, set the quarterly rank to 0.50.

Predeclared candidate scores are fixed convex blends of the original score and
quarterly rank at 75/25, 50/50 and 25/75; the same three blends with 20% of the
within-date rank; and quarterly-rank/date-rank blends at 75/25 and 50/50.
No target, future score, outcome, fitted calibration parameter or mandatory
alert enters the transformation. Apply the existing causal rolling-policy grid
and select only on 2024 under 1--2 overall, >=0.80 per-currency and >=0.70
per-quarter frequency, positive forward benefit, and the usual minimum of
overall/currency lift. Freeze for 2025/2026. Physical corruption of all scores
after a cutoff must leave every earlier normalized score unchanged.

## Packet U: within-week learning-to-rank

Frozen after packet T showed that target-free uniformization restores cadence
by purchasing weak rows, before packet-U later-period evaluation. Instead of
repairing alert frequency after prediction, train the score to distinguish the
best observable day inside the desired operational unit. Reuse the frozen
post-24.02.2022 quarterly-refit broad-CBR feature matrices, but group training
queries by `currency x ISO week`. All labels must have their fifth-publication
reach date strictly before the refit. Predeclared pairwise objectives are:
binary `fav_h5`, the 0--5 forward barrier count, within-week forward-benefit
rank, and an integer lexicographic relevance `20 * fav_h5 + within-week
benefit decile`. Add one NDCG version of the last relevance. Query relevance is
constructed on training groups only.

For every raw ranker, predeclare fixed 75/25 and 50/50 per-currency rank blends
with the multiscale past-range anchor, plus a 50/50 causal blend with packet-P
`stack50_benefit50`. No blend weight is fitted. Screen the existing rolling
alert-policy grid on 2024 with 1--2 overall, >=0.80 per currency, >=0.70 per
quarter and positive benefit, maximizing the usual lift robustness. Freeze
the chosen policy for 2025/2026. This objective can learn weekly relative
quality but does not select the best future day of a live week: every daily
score and decision still uses only information available on that day.

## Packet V: shared five-horizon barrier model

Frozen after packet U failed to transfer its within-week ordering to 2026,
before packet-V later-period evaluation. Replace the single rare-event target
with a shared horizon-conditioned binary problem. For every resolved training
row create five examples with one-hot horizon 1--5 and label whether today's
normalized rate is no greater than the rate at that future publication. Fit
one pooled model per quarterly refit, not five independent models, so the five
horizons share statistical strength. All five future publications must be
resolved strictly before the refit even for the shorter-horizon replicas.

Predeclared learners are regularized HistGradientBoosting, ExtraTrees and XGB
on the compact core + trusted official macro + broad-CBR panel. For each live
row predict five probabilities and aggregate them as minimum, geometric mean,
and conservative `mean - 0.5 * standard deviation`; these are ranking scores,
not an independence claim. Predeclare fixed 50/50 causal blends of every score
with packet-P `stack50_benefit50`, and 75/25 blends with the multiscale anchor.
Screen only the ordinary rolling policy grid on 2024 under the existing 1--2
overall, >=0.80 per-currency, >=0.70 per-quarter, positive-benefit constraints
and robustness objective; then freeze for 2025/2026. No future target is used
as a feature or threshold input.

## Packet W: low-dose shared-horizon consensus

Frozen after packet V showed complementary behavior: packet-P is strong in
2025 but sparse in 2026Q2, while the shared-horizon ExtraTrees variants are
stronger and more active in 2026 but too weak in 2025 when given 50% weight.
Before packet-W later-period evaluation, predeclare causal per-currency rank
blends with 80/20, 75/25 and 67/33 weights between packet-P
`stack50_benefit50` and each shared-horizon ExtraTrees aggregation (minimum,
geometric mean, conservative), plus the shared-Hist minimum and shared-XGB
conservative for model-family diversity. Weights are fixed, not fitted.

Screen the ordinary rolling-policy grid only on 2024 under 1--2 overall,
>=0.80 per currency, >=0.70 per quarter and positive benefit, maximizing
overall/minimum-currency lift. Freeze for 2025/2026. The aim is not to force
cadence but to let a separately learned five-horizon expert raise a day only
when it agrees with most of the primary ranking.

## Packet X: policy-preserving low-dose consensus

Frozen after packet W showed that its 2024 screen switched every useful
ExtraTrees blend from the primary 22%/rolling-60 rule to an inertial
35%/rolling-250 rule, before packet-X later-period evaluation. Reuse all 15
packet-W score blends unchanged, but do not select a new alert policy: apply
the packet-P primary rolling 22%, 60-score, zero-cooldown policy to every
candidate. Rank candidates for reporting by 2024 robustness only; no 2025 or
2026 label chooses a weight or threshold. This isolates whether the
shared-horizon expert adds information from whether policy transfer caused the
failure. The same 1--2 overall and >=0.80 per-currency rate remain the primary
operational criteria; >=0.70 per quarter is a stronger diagnostic.

## Packet Y: business-preserving low-dose consensus

Frozen after packet X improved minimum annual classification lift but made
2025 forward benefit slightly negative, before packet-Y later-period
evaluation. Use two frozen business-aligned bases from packet P: the direct
benefit-ranker+anchor under its 22%/rolling-60 policy, and the 25% stack / 75%
benefit consensus under its 20%/rolling-60 policy. Blend each base at fixed
80/20 and 75/25 weights with the shared-horizon ExtraTrees minimum, geometric
mean and conservative scores, plus shared-XGB conservative. Preserve the
base's own policy; do not rescreen thresholds. Candidate reporting order is
determined by 2024 robustness only. The desired result is annual lift >=1.30,
1--2 alerts per currency-week, and positive forward benefit in both later
years; quarterly cadence remains a secondary diagnostic.

## Packet Z: nonlinear agreement consensus

Frozen after packet Y showed that the shared-horizon classification expert
cannot restore 2025 monetary benefit, before packet-Z later-period evaluation.
Return to the two frozen packet-P components: causal resolved ExtraTrees stack
and direct benefit-ranker+anchor. Convert each component to the same causal
per-currency calibration ranks used in packet P. Predeclare target-free
nonlinear consensus scores: geometric mean, harmonic mean, minimum, and
weighted geometric means with stack/benefit exponents 25/75 and 75/25. These
operators reward agreement rather than letting one very high component fully
compensate for a low one.

Screen the existing ordinary rolling-policy grid on 2024 under 1--2 overall,
>=0.80 per currency, >=0.70 per quarter and positive benefit, maximizing
overall/minimum-currency lift. Freeze for 2025/2026. The transform uses no
labels; when the 2024 stack lacks prior calibration, its rank uses strictly
earlier 2024 scores exactly as in corrected packet P.

## Packet AA: causal tail meta-labeling

Frozen after packet Z showed that target-free nonlinear score algebra cannot
separate the monetary false positives in 2025, before packet-AA later-period
evaluation. Use the prequential packet-M benefit-ranker+anchor as a primary
candidate generator because it has scores from 2023 onward. Transform its
score to a percentile against at most 250 strictly earlier scores per
currency. At every quarterly refit train a second-stage classifier only on
fully resolved rows whose causal primary rank is at least 0.50 or 0.65. The
meta matrix is the causal five-expert feature panel + compact past-only core +
resolved-outcome state used by packet O. Predeclared meta learners are
regularized HistGradientBoosting and ExtraTrees; each predicts all live rows.

Predeclare raw meta probability and fixed per-currency rank blends with the
primary generator at 75/25 and 50/50. Screen ordinary rolling policies only on
2024 under 1--2 overall, >=0.80 per currency, >=0.70 per quarter and positive
benefit, maximizing overall/minimum-currency lift; freeze for 2025/2026. Every
training target must have reached publication `h=5` before the refit, and the
candidate gate for a training row is its historical prequential score, never a
score fitted on that row or a future percentile.

## Packet AB: low-dose tail meta-label

Frozen after packet AA found that a 50/50 tail50 ExtraTrees/primary blend has
positive benefit and strong cadence in both years but only 1.246 lift in 2026,
before packet-AB later-period evaluation. Keep the frozen `tail50_extra` meta
score and direct benefit-ranker+anchor primary. Predeclare primary/meta causal
rank weights 80/20, 75/25 and 67/33. Apply the primary's unchanged
22%/rolling-60/zero-cooldown policy to every blend; do not rescreen thresholds.
Reporting order is based on 2024 robustness only. The target is annual lift
>=1.30, 1--2 alerts per currency-week and positive forward benefit in both
later years, with >=0.70 every quarter as the stronger cadence goal.

## Packet AC: rich causal target-currency panel

Frozen after packet AB showed that reweighting existing scores cannot create
new 2026Q2 signal, before packet-AC feature construction and later-period
evaluation. Build an as-of panel from the current and past published rates of
all five target currencies; current same-date rates are jointly observable.
For windows 1/2/3/5/10/20/40/60 publications add every currency's return,
other-currency mean/std/min/max/positive breadth, the current target-minus-peer
return and its cross-sectional rank. For 10/20/60 aligned past returns add the
target/common correlation, beta and residual volatility. All features end at
the row publication date and use physical nominal-normalized series already
in the repository. A future-corruption rebuild must leave every earlier row
byte-identical.

Quarterly post-24.02.2022 refits use only fully resolved `h=5` targets.
Predeclared learners are regularized HistGradientBoosting, ExtraTrees and XGB
on core + trusted official macro + broad CBR + new panel, plus ExtraTrees on
core + trusted macro + panel without broad references. Predeclare raw scores,
75/25 anchor blends, and fixed 50/50 causal blends with packet-P
`stack50_benefit50` and the direct benefit-ranker. Screen ordinary rolling
policies on 2024 under 1--2 overall, >=0.80 per currency, >=0.70 per quarter
and positive benefit, maximizing the usual lift robustness; freeze for
2025/2026.

## Packet AD: low-dose target-panel consensus

Frozen after packet AC produced a positive-benefit annual pass and repaired
2026 quarterly cadence with a 50/50 target-panel ExtraTrees/primary blend, but
left a sparse 2025 quarter, before packet-AD later-period evaluation. Keep the
raw full target-panel ExtraTrees, panel-only ExtraTrees and full target-panel
XGB scores. Blend packet-P `stack50_benefit50` at fixed primary/panel weights
80/20, 75/25 and 67/33 using causal per-currency ranks. Apply the primary's
unchanged 22%/rolling-60 policy to every candidate; do not rescreen thresholds.
Order only by 2024 robustness. Target annual lift >=1.30, 1--2 alerts per
currency-week and positive forward benefit in both years; use >=0.70 in every
quarter as the stronger diagnostic.

## Packet AE: lagged official MOEX FX market data

Frozen before downloading the full history or evaluating packet-AE models.
The public Moscow Exchange ISS history endpoint was schema-checked on short
date slices for `CNYRUB_TOM`, `USD000UTSTOM` and `EUR_RUB__TOM`; these three
liquid RUB crosses are predeclared without selecting on target performance.
Archive daily OPEN/HIGH/LOW/CLOSE/WAPRICE/NUMTRADES from 2010-01-01 through
the data cutoff with endpoint URLs and a payload SHA-256 manifest.

For a signal row dated `t`, use only MOEX records with `TRADEDATE < t`; same-day
market closes are forbidden. Add lagged close returns 1/2/5/10/20, intraday
range, open-close, close-WAP deviations, rolling volatility and log-trade-count
changes, plus missingness/age features. USD/EUR cessation after 2022 remains
missing and is never backward-filled; CNY supplies the continuous post-shock
market leg. A physical corruption of MOEX observations dated at or after a
cutoff must leave every earlier signal row identical.

Quarterly post-24.02.2022 HistGradientBoosting, ExtraTrees and XGB refits use
only fully resolved `h=5` labels on core + trusted official macro + broad CBR +
MOEX features. Predeclare raw, 75/25 anchor, 50/50 packet-P primary and 50/50
benefit-ranker blends. Screen ordinary rolling policies only on 2024 under the
existing rate/currency/quarter/positive-benefit constraints and freeze for
2025/2026. This packet tests genuinely preceding market information, not the
next published CBR rate.

## Packet AF: MOEX matched ablation and timestamp audit

Frozen immediately after packet AE's large lift jump and before reading any
ablation later-period score. Treat the packet-AE winner as suspicious until it
passes matched controls. Refit the identical ExtraTrees architecture on the
identical core + trusted macro + broad-CBR matrix with: no MOEX columns,
CNYRUB_TOM columns only, USD000UTSTOM only, EUR_RUB__TOM only, and all three.
Use the full winner's already selected 22%/rolling-20 policy for every variant;
there is no new policy screen. Report 2024, 2025 and 2026, but order variants
in the predeclared sequence rather than by later performance.

Additionally, prove for every signal row that the selected market observation
date is strictly smaller than the signal date; modify same-date and future
market rows in a physical test; report age/missingness by instrument and year;
and fit one final-quarter all-MOEX model only on resolved training labels to
record impurity feature importance. The full result is accepted as new signal
only if at least one real MOEX subset improves the matched no-MOEX control and
the timestamp/source checks pass.

## Packet AG: CNY signal decomposition and negative controls

Frozen after packet AF showed that CNY-only reproduces the gain with full
post-2022 coverage, before packet-AG later-period evaluation. Refit the same
ExtraTrees and apply the same fixed 22%/rolling-20 policy to four predeclared
feature subsets added to the matched no-MOEX base: (1) all CNY features;
(2) CNY close returns, volatility, age and missingness only; (3) CNY intraday
open-close/range/close-WAP/overnight and trade-count features only; and (4) all
CNY features delayed by 20 target-publication rows per currency as a causal
stale-signal negative control. No policy or architecture is rescreened.

The aligned market-information interpretation is strengthened only if a real
aligned subset materially beats both no-MOEX and the stale control. Report
the decomposition in the fixed listed order, with yearly/currency/quarter
metrics and block bootstrap; it is diagnostic and does not retroactively
change packet AF's candidate definition.

## Packet AH: low-dimensional explainable CNY models

Frozen after packet AG established that aligned CNY intraday information beats
both the no-market and stale-market controls, before packet-AH later-period
evaluation. Keep the fixed 22%/rolling-20 policy and quarterly resolved-label
refits. Compare, in predeclared order: (1) logistic regression using only CNY
intraday features plus target-currency identity; (2) logistic regression after
adding the transparent 30/90/180 past-range anchor and target returns 1/5/20;
(3) HistGradientBoosting on that same low-dimensional matrix; (4) logistic
regression on the full matched base plus CNY intraday features; and (5) the
packet-AG intraday ExtraTrees control. No policy or architecture is selected on
2025/2026. The purpose is explanatory compression: determine how much of the
tree ensemble's gain survives in a small auditable model, not to replace the
frozen packet-AF candidate post hoc.

## Packet AI: fixed CNY model consensus

Frozen after packet AH exposed complementary quarterly errors, before any
packet-AI blend is evaluated. Convert each component score to a causal
per-currency rank against its own preceding calibration scores, then compare
fixed blends under the unchanged 22%/rolling-20 policy: logit/hist 50/50 and
67/33, logit/ExtraTrees 50/50 and 67/33, and equal thirds of logit, hist and
ExtraTrees. Weights are deliberately coarse and unfitted. This is a
retrospective robustness experiment; it may suggest a future frozen candidate
but cannot manufacture a new untouched holdout.

## Packet AJ: policy plateau and component-overlap audit

Frozen after packet AI, before reading any sensitivity cell. Do not fit another
model. Audit the fixed `logit50_extra50` score over the Cartesian neighbourhood
of target rates 18/20/22/25/30 percent and causal trailing windows 20/40/60.
The original 22%/20 result remains primary regardless of the grid. Report every
cell for 2024, 2025 and 2026, and count cells clearing annual lift 1.30 and rate
1--2 in both later years. Under the original policy, also report signal-set
overlap and conditional target/benefit for logit-only, ExtraTrees-only,
intersection, union, logit-only residual and ExtraTrees-only residual. This is
a tuning-robustness and complementarity audit, not a post-hoc policy screen.

## Packet AK: pre-2022 transport stress test

Frozen after packet AJ and before generating any pre-2022 CNY model score.
Recreate the packet-AH 19-feature logit and matched intraday ExtraTrees with
quarterly expanding refits beginning in 2016. Admit only labels whose h=5 reach
date precedes the refit and train only rows on/after CNYRUB_TOM history begins
(2013-04-15). Build yearly outputs with the preceding calendar year as
calibration, then evaluate the unchanged fixed 50/50 causal-rank consensus and
22%/rolling-20 policy on each of 2017--2021 and combined. Do not rescreen
features, weights or policy. This packet tests cross-regime transport; failure
would narrow the claim to the post-2022 market structure rather than invalidate
the causal post-2022 result.

## Packet AL: 2022 shock bridge and reset hand-off

Frozen after packet AK demonstrated transport across 2017--2021, before any
2022/2023 bridge score is generated. Refit the same logit and ExtraTrees each
quarter with two causal histories: expanding from 2013-04-15, and hard-reset at
2022-02-24 once at least 700 resolved rows exist. Define a mechanical hybrid
that uses the expanding score until both reset components are available and the
reset score thereafter; blend each history 50/50 by causal per-currency ranks.
Keep the 22%/rolling-20 policy. Report pre-shock 2022Q1, post-shock 2022Q2--Q4,
2023, and quarters. No transition date, minimum sample size, weight or policy
may be chosen from the bridge results.

## Packet AM: long-history versus post-2022 weighting

Frozen after packet AL showed that an immediate hard reset hurts the 2022
transition, before generating any new 2024--2026 score. Refit the same logit
and ExtraTrees quarterly under: (1) all resolved rows since 2013-04-15 with
unit weight; (2) the same history with fixed sample weight 3 for dates on/after
2022-02-24 and weight 1 before; and (3) the existing hard-reset packet-AI
control. Within each history blend logit/ExtraTrees 50/50 by causal currency
ranks. Also form a fixed 50/50 causal-rank blend of the all-history and reset
consensuses. Apply only 22%/rolling-20. The factor 3 and blend weight are coarse
predeclared values, not optimized on later results. This packet tests the
user-proposed partial retention of the pre-shock regime.

## Packet AN: lagged MOEX-versus-CBR CNY basis

Frozen after packet AM and before constructing or evaluating any basis feature.
For each MOEX CNY trade date `s < signal_date`, join only the latest normalized
official CBR CNY/RUB value dated `<= s`. Build close/open/WAP basis in bps,
basis changes over 1/2/5/10/20 trades, rolling mean/z-score over 5/20/60 and
MOEX-minus-CBR return gaps over 1/2/5/10/20. Missing/staleness are explicit;
no same-day market close or future CBR row is permitted. Physically corrupting
either source at/after a cutoff must not change earlier target rows.

Under hard-reset quarterly refits and fixed 22%/rolling-20, compare: existing
CNY-intraday ExtraTrees; ExtraTrees with aligned basis; the same model with
basis delayed 20 target rows per currency; a low-dimensional logit adding basis
to the 19 packet-AH features; and a fixed 50/50 causal-rank blend of the new
logit/ExtraTrees. No feature subset, weight or policy is rescreened. The basis
claim is accepted only if aligned basis beats its delayed control.

## Packet AO: independent MOEX risk and liquidity context

Frozen after schema-only short-range ISS queries and before downloading full
history or evaluating a target. Predeclare four public MOEX instruments without
target screening: `IMOEX` (equity risk, stock/index/SNDX), `RGBI` (government
bond prices, stock/index/SNDX), `RUSFAR` (secured RUB funding, stock/index/MMIX)
and `GLDRUB_TOM` (RUB gold, currency/selt/CETS). Archive daily history from
2010-01-01 through the fixed dataset cutoff 2026-09-03 with URLs, row counts and
payload SHA-256.

For every signal at `t`, use only a record with `TRADEDATE < t`, maximum
staleness seven calendar days. Add returns 1/2/5/10/20, volatility 5/20/60,
open-close and intraday range, level z-scores 20/60/120, available yield and
log activity/value changes; missingness and age are explicit. Future/same-day
physical corruption must leave earlier feature rows identical.

Under the same hard-reset ExtraTrees and unchanged 22%/rolling-20 policy,
compare in fixed order: no added context; IMOEX only; RGBI only; RUSFAR only;
gold only; all four; and all four delayed 20 target rows per currency. Also
form a fixed 50/50 causal-rank blend of the existing CNY primary with the
all-context ExtraTrees. No subset, blend weight or policy is selected on
2025/2026. A context claim requires the aligned group to beat both the no-
context and delayed controls; otherwise it remains a negative result.

## Packet AP: per-instrument stale controls and low-dose consensus

Frozen after packet AO showed small aligned gains for each single instrument
but rejected the all-context group, before fitting any single-instrument stale
control or reading any new blend. For IMOEX, RGBI, RUSFAR and GLDRUB_TOM,
refit the identical context ExtraTrees after delaying only that instrument's
22 features by 20 target rows per currency. Compare each aligned/stale pair
under fixed 22%/rolling-20. Independently create fixed causal-rank blends of
75% existing `logit50_extra50` primary and 25% each aligned single-context
ExtraTrees; weights and policy are not screened. A single source is treated as
fresh information only if aligned beats its own stale control in both combined
lift and minimum-currency lift; a low-dose blend remains only a prospective
candidate if it improves the primary's minimum annual lift without violating
the 1--2 rate band.

## Packet AQ: derived CNY daily microstructure

Frozen after the official ISS history/candle schemas showed that CNYRUB_TOM
exposes `NUMTRADES` and `WAPRICE` but historical candle value/volume are null,
before building or evaluating derived features. From the last completed trade
date only, derive: signed pressure (`open-close + close-WAP`), pressure divided
by intraday range, close and WAP location inside high-low, normalized candle
body, upper/lower wick asymmetry, sign-agreement flags, and trailing z-scores
over 5/20/60 trades for pressure, range, close location, overnight gap and log
trade count. Clip only dimensionless ratios at fixed +/-5. The same strict
`TRADEDATE < signal_date`, seven-day staleness and physical future-corruption
test apply.

Under hard-reset quarterly refits and fixed 22%/rolling-20, compare existing
CNY intraday ExtraTrees, ExtraTrees with aligned derived microstructure, the
same derived block delayed 20 target rows per currency, a small packet-AH logit
augmented with aligned derived features, and a fixed 50/50 causal-rank blend of
the enriched logit/ExtraTrees. No subset, weights or policy are rescreened. The
derived block is accepted only if aligned ExtraTrees beats its stale control;
otherwise any large standalone metric remains a rejected post-hoc candidate.

## Packet AR: local-currency CNY experts with global shrinkage

Frozen after packet AQ rejected derived microstructure, before fitting any
local CNY model. At every quarterly refit, train five independent models using
only resolved post-24.02.2022 labels from the same target currency. Local logit
uses the eight raw CNY intraday features plus past target ranges 30/90/180 and
returns 1/5/20; local ExtraTrees uses the existing matched full+CNY-intraday
matrix. Require at least 140 resolved rows per currency and use the unchanged
global model hyperparameters.

Under fixed 22%/rolling-20 compare local logit, 75/25 global/local logit; local
ExtraTrees, 75/25 global/local ExtraTrees; 50/50 local logit/ExtraTrees; and
75/25 existing primary/local consensus. Every mixture uses causal per-currency
ranks with coarse fixed weights. No corridor champion, weight or policy is
selected from later outcomes. The hypothesis is partial pooling: local models
may add corridor response while the global component supplies shrinkage.

## Packet AS: paired primary-versus-local audit

Frozen after packet AR produced `primary75_local_consensus25`, before computing
any paired uncertainty statistic. Do not fit or select another model. Under the
unchanged 22%/rolling-20 policy, compare the existing `logit50_extra50` primary
and the fixed local challenger on identical 2025--2026 rows. Use the same
four-week block bootstrap draws to report the paired difference in lift and
mean forward benefit, including 95% intervals and one-sided probability of no
improvement. Report Jaccard overlap plus target rate/benefit for intersection,
primary-only and challenger-only signals. The primary is not replaced unless
the paired lift interval excludes zero and minimum annual lift is no worse.

## Packet AT: one-stage hierarchical interaction logit

Frozen after packet AS showed a small but statistically unresolved benefit from
five separately fitted local experts, before fitting any packet-AT model. Build
one regularized logistic regression that contains: (1) the same eight lagged
CNY intraday fields and six past-only target anchor/return fields as global main
effects; (2) five currency indicators; and (3) each of the fourteen numerical
fields interacted with each currency indicator. Standardize columns and keep
the existing fixed `C=0.025` L2 penalty. This is a single partial-pooling model:
the main effects learn from all corridors while regularization shrinks every
corridor deviation toward the shared response.

Refit quarterly on resolved post-24.02.2022 labels under the unchanged
`TRADEDATE < signal_date` rule. Compare the hierarchical logit alone, a fixed
50/50 causal-rank blend with the global 19-feature logit, and a fixed 75/25
causal-rank blend of the existing primary with the hierarchical logit. Keep the
22%/rolling-20 alert policy. No interaction, penalty, blend weight or policy is
selected from 2025/2026. The packet is an architectural test of explicit
shrinkage, not another local-weight search; it replaces neither frozen
candidate without a paired interval excluding zero and non-worse annual gates.

## Packet AU: full-history causal lifecycle policy

Frozen after all component era audits but before constructing or evaluating a
single 2017--2026 output. Do not fit a new model or alter a score. Stitch the
already saved, causally generated 50/50 logit/ExtraTrees consensuses into three
deployment histories under the unchanged 22%/rolling-20 policy:

1. `always_expanding`: packet-AK expanding scores in 2017--2021, packet-AL
   expanding scores in 2022--2023, and packet-AM all-history scores thereafter.
2. `early_reset_700`: packet-AK expanding scores in 2017--2021, packet-AL's
   mechanical reset-at-700 hybrid in 2022--2023, and hard reset thereafter.
3. `resolved2000_handoff`: expanding scores through 2023, then hard reset from
   the first refit with at least 2,000 resolved post-24.02.2022 target rows. The
   threshold represents roughly two post-shock years across five corridors and
   mechanically maps to the 2024-01-01 refit; it is not selected from a metric.

Report every year, 2017--2026 combined, minimum annual and currency lift,
annual/quarterly cadence, future benefit and four-week block bootstrap. The
purpose is to distinguish a stable deployable training-memory lifecycle from
separate era anecdotes. No year may be dropped. This remains a retrospective
composition audit because its component periods have already been inspected.

## Packet AV: low-dose non-market shock bridge

Frozen after packet AU exposed 2022Q3 as the only sub-1.30 quarter in the
otherwise stable lifecycle, before calculating any packet-AV quarter metric.
Do not refit a model. During 2022--2023 only, blend 75% of the packet-AL
expanding CNY consensus causal rank with 25% of exactly one previously frozen,
non-MOEX score: locked multiscale anchor, round-2 global compact ExtraTrees, or
round-2 soft regime router. Also test fixed equal thirds of CNY, anchor and
global ExtraTrees. The pre-2022 and 2024+ lifecycle portions remain identical
to `resolved2000_handoff`; alert policy remains 22%/rolling-20.

Report 2022, 2023, every shock quarter and the complete 2017--2026 lifecycle.
The bridge is useful only if it improves the minimum shock-quarter lift over
1.135 without lowering any annual lift below 1.30, moving annual rate outside
1--2, or lowering the ten-year minimum-currency lift below 1.30. No blend is
chosen by the same quarter it is meant to repair; all four coarse definitions
and the unblended control remain recorded.

## Packet AW: paired lifecycle shock-bridge audit

Frozen after packet AV identified `cny75_anchor25` as the only bridge improving
both shock years, minimum shock-quarter lift, ten-year lift, benefit and minimum
currency lift, before computing any paired interval. Do not fit or select a
model. Compare `cny75_anchor25` with `cny_expanding` on identical rows under the
fixed 22%/rolling-20 policy, separately for 2022--2023 and 2017--2026. Use
paired four-week block-bootstrap draws for lift and benefit differences; keep
the five currencies of each date together. Report signal Jaccard and target
rate/benefit for intersection and each model-only subset.

Promote the anchor bridge only as a retrospective lifecycle challenger if its
shock-period point estimate, both annual lifts, full-lifecycle minimum annual
lift and minimum-currency lift are all higher. Describe statistical superiority
only if the paired shock-period lift interval excludes zero. The already frozen
prospective 2026 primary/challenger pair is unaffected by this historical
deployment-memory audit.

## Packet AX: shock-bridge weight plateau

Frozen after packet AW, before evaluating any additional CNY/anchor weight.
Do not fit a model and do not replace the predeclared 75/25 challenger. Audit
fixed CNY causal-rank weights 50%, 60%, 70%, 75%, 80%, 90% and 100%, with the
remainder assigned to the locked multiscale anchor, only in 2022--2023. Keep
the same pre-2022 expanding and 2024+ hard-reset lifecycle plus the unchanged
22%/rolling-20 policy.

For every weight report both shock years, all eight shock quarters, combined
shock lift/benefit, full 2017--2026 lift, minimum annual/currency lift and
cadence. This grid is a post-diagnostic sensitivity audit, never a same-period
selection. The 75/25 claim is considered structurally robust only if several
neighbouring weights improve both shock-year lifts and full-lifecycle minimum
annual lift over 100% CNY; a single optimal cell is treated as noise.

## Packet AY: multiplicity-aware shock-weight audit

Frozen after the complete packet-AX grid showed a broad 50--80% CNY plateau
and identified 60/40 as the grid maximum, before calculating uncertainty or a
selection-adjusted p-value. Do not fit, blend or select another score. For all
six non-control weights, compare their fixed alert masks with 100% CNY using
paired four-week block-bootstrap differences on 2022--2023 and 2017--2026.

Because 60/40 was identified from seven inspected weights, also circularly
shift the five-currency target matrix over dates and compute the maximum
candidate-minus-control lift difference across all six challengers on every
null draw. Report max-adjusted p-values for the observed differences. The
60/40 variant may be called a statistically supported retrospective shock
bridge only if its ordinary paired interval excludes zero and its max-adjusted
shock p-value is below 0.05. This audit does not turn the inspected period into
a prospective holdout.

## Packet AZ: regularized additive spline models

Frozen after completing the shock-weight audit and before fitting any spline
model. Return to the mature post-2022 task and introduce a genuinely different,
explainable nonlinear class. Use the same eight lagged CNY intraday fields, six
past-only target range/return fields and five currency indicators as packet AH.
Fit L2 logistic generalized additive models with quantile-knot quadratic
splines, exactly five knots, no spline bias and linear extrapolation. Keep
`C=0.025`, quarterly resolved-label refits, hard reset at 24.02.2022 and strict
`TRADEDATE < signal_date`.

Predeclare two global forms: splines on only the eight market fields with the
six target fields linear, and splines on all fourteen numerical fields. Also
fit the all-spline form separately per currency with minimum 140 resolved rows.
Under the unchanged 22%/rolling-20 policy compare both global scores, local GAM,
fixed 75/25 global-all/local-GAM shrinkage, fixed 75/25 primary/global-all-GAM,
and fixed 75/25 primary/local-GAM. All blends use causal per-currency ranks.
Knots, penalty, variants and weights are not screened on 2025/2026. This packet
tests whether smooth additive nonlinearities can recover tree performance; it
cannot alter the frozen prospective candidates without a later paired audit.

## Packet BA: paired and multiplicity-aware GAM consensus audit

Frozen after all six packet-AZ results were written, before computing any
paired difference. Do not fit, retune or create another score. Compare the two
predeclared 75/25 primary/GAM consensuses (`primary75_all_gam25` and
`primary75_local_gam25`) with the unchanged `logit50_extra50` primary on
identical rows in 2024, 2025, 2026 and 2025--2026 combined. Use paired
four-week block-bootstrap lift and benefit differences. Also calculate circular
date-shift maximum-difference p-values across both GAM challengers so that the
better later-period variant is not treated as if selected alone.

Promotion requires both annual lift and rate gates, combined minimum-currency
lift >=1.30, minimum quarterly rate >=1.00, a paired combined lift interval
excluding zero, and max-adjusted p<0.05. A candidate that clears retrospective
gates is still added as a new prospective model version rather than silently
altering the previously hashed freeze manifest.

## Packet BB: causal nearest-neighbour reliability and benefit surface

Frozen after packet BA rejected a primary replacement on uncertainty, before
constructing any packet-BB score. Build a nonparametric meta-model over two
already causal experts: `logit50_extra50` primary and global `all_spline_gam`.
First map each raw expert score to its percentile among at most the preceding
250 scores of the same currency, requiring 20 past scores. The three distance
coordinates are primary rank, GAM rank and their absolute disagreement.

At each quarterly refit, admit only post-24.02.2022 rows whose h=5 reach date is
strictly before the refit. For each test row find 250 nearest pooled resolved
rows and 80 nearest same-currency resolved rows; pooled distances add a fixed
0.05 penalty for a currency mismatch. Estimate hit probability with a fixed
Jeffreys Beta(0.5,0.5) posterior and use mean minus one posterior standard
deviation as an uncertainty score. Estimate forward benefit from the same
neighbours as mean minus one standard error. Shrink local and pooled lower
bounds with fixed local weight `n_local/(n_local+100)`.

Under 22%/rolling-20 compare pooled hit LCB, shrunk hit LCB, shrunk benefit LCB,
their fixed 50/50 causal-rank consensus, and 75% primary plus 25% reliability/
benefit consensus. Neighbour counts, priors, distance penalty, shrinkage and
blend weights are fixed here and not screened on 2025/2026. Physically changing
unresolved future targets/benefits must leave earlier meta-scores unchanged.

## Packet BC: reliability-surface paired audit

Frozen after all five packet-BB metrics and the future-outcome corruption check
were written, before computing a paired difference. Do not alter neighbours,
priors, distance, shrinkage, scores or policy. Compare every packet-BB alert
mask against `logit50_extra50` on identical rows for 2024, 2025, 2026 and
2025--2026 combined. Use paired four-week block bootstrap for lift/benefit and
circular-date maximum-difference adjustment across all five BB variants.

A standalone reliability score may replace primary only if its combined paired
interval excludes zero, adjusted p<0.05, both annual lifts and rates pass, its
minimum annual lift is not below the primary, minimum currency lift >=1.30 and
minimum quarter rate >=1.00. Otherwise retain it only as an independent expert
for genuinely future evaluation; do not tune neighbour counts on these years.

## Packet BD: raw-trajectory historical analogues

Frozen after packet BC and before evaluating a raw analogue. This packet must
not use primary/GAM predictions as distance coordinates. Build target history
from past returns 1/3/5/10/20/60, range positions 30/90/180 and volatilities
10/30/90. Build completed-session CNY history from returns 1/2/5/10/20,
volatilities 5/20/60, open-close, intraday range, close-WAP, overnight gap and
log trade count. Every CNY row obeys `TRADEDATE < signal_date`.

At every quarterly refit robust-scale each coordinate using only resolved
post-24.02.2022 training rows (median and IQR, with zero-IQR fallback). Estimate
a Jeffreys-posterior hit lower bound from 250 pooled and 80 same-currency nearest
analogues and shrink local toward pooled with `n_local/(n_local+100)`, reusing
packet-BB neighbour counts without rescreening. Compare target-only, CNY-only
and joint-path analogues, plus a fixed 75% primary/25% joint-analogue rank blend,
under 22%/rolling-20. A future-outcome corruption test is mandatory. This is a
raw-shape ablation, not another reliability-score neighbour tuning exercise.

## Packet BE: completed-session CNY waveform compression

Frozen after packet BD showed that raw Euclidean analogues are too crude,
before constructing any waveform field. From only the last 20 completed
`CNYRUB_TOM` session-to-session returns (`TRADEDATE < signal_date`), retain all
20 ordered returns; the first eight orthonormal DCT-II coefficients; means and
volatilities over 5/10/20; upside/downside volatility; skew and lag-1
autocorrelation; positive fractions over 5/10/20; sign-flip fraction; maximum
run-up/drawdown and their normalized positions; last-return z-score; last-five
minus previous-five acceleration; age and missingness. No frequency, coefficient
or path statistic is selected from a target result.

Under quarterly resolved post-24.02.2022 refits and fixed 22%/rolling-20,
compare: L2 logit on waveform + currency + target anchors/returns; ExtraTrees on
the existing trusted global base + waveform; the identical ExtraTrees after
delaying the entire waveform block 20 target rows per currency; fixed 50/50
causal-rank waveform logit/ExtraTrees; and fixed 75/25 primary blends with each
aligned waveform model. Physical same-day/future CNY corruption must preserve
past waveform rows. Treat waveform information as fresh only if aligned
ExtraTrees beats the stale control in combined and minimum-currency lift.

## Packet BF: paired audit of waveform freshness and incremental value

Frozen after all packet-BE scores and aggregate metrics were written, before
calculating any paired difference. Do not refit or retune a model and keep the
22%/rolling-20 policy unchanged. Test exactly two predeclared comparisons on
identical 2025--2026 rows: aligned `wave_extra` against
`wave_extra_stale20`, which asks whether recent completed-session waveform
information is genuinely useful; and `primary75_wave_logit25` against the
unchanged `logit50_extra50` primary, which asks whether that information adds
enough to replace the frozen primary.

Use paired four-week block-bootstrap differences for lift and forward benefit,
plus a signal-overlap decomposition. Report one-sided paired bootstrap
p-values both raw and Holm-adjusted across the two hypotheses. Also report a
separate circular-date-shift p-value for each fixed comparison; this is a
diagnostic for calendar alignment, not a substitute for the paired interval.
Call waveform information supported only if the aligned-versus-stale lift
interval excludes zero after the aggregate stale gate from packet BE. Promote
the blend only if its lift interval excludes zero, its Holm p-value is below
0.05, both annual lifts and rates pass, minimum currency lift is at least 1.30
and minimum quarterly rate is at least 1.00. Retrospective support creates a
new prospective candidate; it never changes the existing freeze manifest.

## Packet BG: fixed random-convolution CNY path classifier

Frozen after packet BF confirmed fresh ordered-path information, before
constructing a convolution feature or fitting a model. This is a new transform
of the market path, not a retry of the failed round-five ROCKET model over
target-currency trajectories. Start from the same last 20 completed
`CNYRUB_TOM` returns as packet BE and standardize each path by its own mean and
standard deviation. Generate exactly 64 kernels from seed 20260905: 16 each of
length 3, 5, 7 and 9; centered unit-norm Gaussian weights; dilations cycling
through the values in {1,2,3} whose receptive field fits 20; and biases drawn
uniformly from [-1,1]. For every kernel retain only maximum response and
proportion positive, yielding 128 target-independent features.

Fit the unchanged L2 logistic pipeline on those 128 features plus the packet-BE
50 raw/DCT/summary waveform fields, five currency indicators and six past-only
target anchors/returns. Compare it with the identical matrix whose entire CNY
path block is delayed 20 target rows per currency; the packet-BE waveform
logit; a fixed 50/50 causal-rank blend with waveform ExtraTrees; and a fixed
75/25 primary/random-convolution blend. Keep quarterly resolved post-2022
refits and 22%/rolling-20. Same-day/future CNY corruption must leave every past
convolution feature unchanged. Call the convolution basis fresh only if aligned
logit beats delayed logit in combined and minimum-currency lift; do not tune
kernel count, seed, lengths, penalty or blend weights on 2025/2026.

## Packet BH: paired random-convolution freshness and primary audit

Frozen after all five packet-BG aggregate metrics were written and revealed the
predeclared `primary75_rocket25` point estimate, before calculating any paired
difference or inspecting signal overlap. Do not refit, retune, change a kernel
or test another blend weight. On identical 2025--2026 rows test exactly two
hypotheses: aligned `rocket_logit` versus `rocket_logit_stale20`, and
`primary75_rocket25` versus the unchanged `logit50_extra50` primary.

Use paired four-week block bootstrap for lift and forward-benefit differences,
Holm-adjust one-sided paired p-values across these two hypotheses, and report a
fixed-comparison circular-date-shift diagnostic plus signal-overlap subsets.
Random-convolution freshness requires the aligned-minus-stale lift interval to
exclude zero in addition to packet BG's aggregate and minimum-currency gate.
Blend promotion requires lift >=1.30 and rate 1--2 in both years, minimum
currency lift >=1.30, minimum quarter rate >=1.00, positive combined point gain,
a paired lift interval excluding zero and Holm p<0.05. Even if all gates pass,
the result remains a new frozen prospective challenger because 2025--2026 is
already inspected; the existing primary manifest is not rewritten.

## Packet BI: resolved-error CNY regime stack

Frozen after packet BH and before fitting a regime model. Implement the user's
error-regime idea without a future-aware router. Reconstruct row-level OOF
scores for exactly three already fixed experts: `logit50_extra50` primary,
packet-BE `wave_extra`, and packet-BG `rocket_logit`. Convert each to a causal
same-currency percentile using at most the preceding 250 scores and requiring
20. The regime matrix contains the three ranks, their three absolute pairwise
disagreements, mean/min/max/std, five currency indicators, six past-only target
anchors/returns and the fixed packet-BE waveform block. No future error table,
year label or post-hoc winning-regime flag is allowed.

At each quarterly refit from 2024 onward, train only rows dated from 2023 whose
five-publication target has strictly resolved before the refit. Predeclare two
models: L2 logistic regression (`C=0.02`) and a shallow histogram gradient
booster (200 trees, learning rate 0.03, at most 5 leaves, minimum leaf 80,
L2=25). Also fit the identical booster after delaying both auxiliary expert
ranks and the waveform block by 20 target rows per currency while retaining
the current primary rank. Compare the three direct stacks and fixed 75/25
primary blends with each aligned stack under 22%/rolling-20.

Physically flipping all labels whose h=5 reach lies after a cutoff must leave
earlier regime-stack predictions bit-identical. A fresh regime claim requires
aligned histogram stack to beat its delayed control in combined and
minimum-currency lift. Do not tune depth, penalty, history start, rank window,
delay, policy or blend weight on 2024--2026.

## Packet BJ: paired audit of the resolved-error regime stack

Frozen after all packet-BI aggregate metrics were written and showed the two
predeclared primary blends, before calculating paired differences or signal
overlap. Do not refit, change the regime matrix or repair the observed cadence.
Test exactly three hypotheses on identical 2025--2026 rows: aligned
`regime_hist` versus `regime_hist_stale20`; `primary75_regime_logit25` versus
`logit50_extra50`; and `primary75_regime_hist25` versus the same primary.

Use paired four-week block-bootstrap lift and benefit differences, one-sided
paired p-values with Holm adjustment across all three hypotheses, separate
fixed-comparison circular-date-shift diagnostics, and overlap subsets. Fresh
regime information requires the aligned-versus-stale lift interval to exclude
zero in addition to packet BI's aggregate gate. Each blend independently
requires both annual lift/rate gates, minimum currency lift >=1.30, minimum
quarter rate >=1.00, positive combined point gain, paired lift interval above
zero and Holm p<0.05. No candidate failing cadence may be repaired on the same
period; any apparently useful score is frozen for later prospective evidence.

## Packet BK: full-lifecycle convolution and regime transport

Frozen after packet BJ, before fitting an older-period convolution model or
reading a 2017--2026 aggregate. Reuse the packet-AU mechanical training-memory
handoff date 2024-01-01, determined by the first refit with at least 2,000
resolved post-2022 rows. Fit the unchanged packet-BG convolution logit with
expanding history from the CNY start through 2023; from 2024 onward use the
already saved post-2022 reset convolution scores. Stitch this into one causal
2017--2026 convolution lifecycle.

Under the unchanged 22%/rolling-20 policy compare exactly four lifecycles: the
packet-AU `resolved2000_handoff` primary control; convolution alone with the
same memory handoff; their fixed 75/25 causal-rank blend throughout; and a
mechanical primary-to-`primary75_regime_logit25` handoff on 2024-01-01 (primary
through 2023, already saved regime blend thereafter). Report every year, all
years combined, minimum annual lift/rate, minimum currency lift, quarter rate,
four-week block-bootstrap intervals and circular multiplicity across all four.

No model class, seed, feature, weight, threshold or handoff date may change.
A challenger is lifecycle-feasible only if combined lift exceeds the primary,
every annual lift is at least 1.30, every annual rate lies in [1,2], minimum
currency lift is at least 1.30 and minimum quarter rate is at least 0.90. This
is a transport audit, not permission to repair a weak historical segment.

## Packet BL: paired full-lifecycle challenger audit

Frozen after all packet-BK annual and aggregate results were written, before
calculating any paired difference or overlap. Do not refit, alter the 2024
handoff or relax packet BK's failed minimum-quarter-rate gate. Compare exactly
`primary75_rocket25_lifecycle` and `primary_then_regime2024` with
`primary_resolved2000` on identical 2017--2026 rows.

Use paired four-week block-bootstrap lift and forward-benefit differences and a
circular-date maximum-difference adjustment across both challengers. Report
signal overlap and each candidate's minimum annual lift/rate and minimum
currency lift. Statistical lifecycle superiority requires a paired lift
interval excluding zero and max-adjusted p<0.05. Separately retain the user's
primary operational requirement of annual lift >=1.30 and annual average rate
1--2; do not conceal the stricter predeclared quarter-rate failure even though
the primary control also fails that extra diagnostic.

## Packet BM: low-dose global residual boosting over primary

Frozen after packet BL and before fitting any residual model. Reconstruct the
frozen primary's own-year OOF score and causal same-currency rank using at most
the preceding 250 scores and minimum 20. At each quarterly refit from 2024,
using only 2023-onward rows whose h=5 reach is strictly before the refit, fit an
L2 logistic base calibrator (`C=0.05`) on primary rank plus five currency
indicators. Define training residual as `y - calibrated_primary_probability`.

Fit exactly two global residual regressors over the packet-BI aligned 71-field
regime matrix: shallow HistGradientBoostingRegressor (150 iterations, learning
rate 0.03, max 5 leaves, minimum leaf 100, L2=30) and ExtraTreesRegressor (400
trees, depth 6, minimum leaf 40, max-features 0.65, seed 20260905). The score is
base probability plus exactly 0.25 times predicted residual. Also fit the
identical HistGB residual model on packet-BI's stale20 matrix and construct one
fixed 50/50 causal-rank consensus of the two aligned residual scores.

Compare frozen primary, calibrated primary, both aligned residual scores, the
stale control and their fixed consensus under 22%/rolling-20. No residual
weight, learner parameter, matrix, policy or training start may be screened.
Physically flipping every label whose reach lies after a cutoff must preserve
all earlier two-stage scores. Treat residual regime information as fresh only
if aligned HistGB beats stale20 in combined and minimum-currency lift.

## Packet BN: paired audit of residual correction

Frozen after all packet-BM aggregate metrics were written, before calculating
paired differences or overlap. Do not refit, change the 0.25 residual dose or
repair cadence. Test exactly three comparisons on identical 2025--2026 rows:
aligned `residual_hist25` versus `residual_hist_stale20_25` for freshness;
`residual_hist25` versus frozen `logit50_extra50`; and
`residual_extra25` versus the same primary.

Use paired four-week block-bootstrap lift and benefit differences, Holm-adjust
the one-sided paired p-values across all three hypotheses, report separate
fixed-comparison circular-shift diagnostics and signal overlap. Promotion of a
residual score requires lift >=1.30 and rate 1--2 in both years, minimum
currency lift >=1.30, minimum quarter rate >=1.00, positive combined gain,
paired lift CI above zero and Holm p<0.05. A fresh-Hist claim additionally
requires its aligned-versus-stale paired lift interval above zero. Preserve all
failed gates rather than selecting another dose or threshold on these years.

## Packet BO: unsupervised CNY state and transition tables

Frozen after packet BN and before fitting a cluster or reading any state
metric. At every quarterly refit, robustly standardize the packet-BE 50-field
waveform using only resolved post-24.02.2022 training rows, then fit exactly
eight KMeans states (`n_init=20`, seed 20260905) on one copy of each training
date so the five target currencies do not duplicate the unsupervised fit. Test
rows are assigned to these frozen centroids. No target, benefit, year or expert
score may enter clustering.

Using only the same resolved training rows, estimate a Jeffreys Beta(0.5,0.5)
hit lower bound (`mean - 1 sd`) for each pooled state and each currency-state.
Shrink local toward pooled with fixed weight `n_local/(n_local+100)`. Also
estimate the same shrunk lower bound for previous-state/current-state pairs,
and a forward-benefit lower bound (`mean - 1 standard error`) by state. Compare
pooled hit, local/global shrunk hit, shrunk transition hit, shrunk benefit,
their fixed 50/50 causal-rank state-hit/benefit consensus, the identical
shrunk-state model on a waveform delayed 20 target rows per currency, and a
fixed 75/25 primary/shrunk-state blend. Keep 22%/rolling-20.

Physical corruption of unresolved future targets and benefits must leave every
earlier state score bit-identical; same-day/future market corruption remains
covered by packet BE. Treat aligned state information as fresh only if shrunk
state hit beats its stale20 control in combined and minimum-currency lift. Do
not tune cluster count, state representation, shrinkage, confidence penalty,
transition order, blend weights or policy on 2024--2026.

## Packet BP: paired audit of unsupervised-state correction

Frozen after all packet-BO aggregate metrics were written, before calculating
any paired lift or benefit difference. Do not refit the clusters, change the
eight-state representation, repair the standalone state's excessive cadence,
or select another primary/state weight. Test exactly two comparisons on
identical 2025--2026 rows: aligned `cluster_shrunk_hit_lcb` versus its
`cluster_stale20_hit_lcb` control for waveform freshness, and
`primary75_cluster25` versus frozen `logit50_extra50` for incremental value.

Use paired four-week block-bootstrap lift and forward-benefit differences,
Holm-adjusted one-sided paired p-values across both hypotheses, a separate
fixed-comparison circular-date maximum-difference audit, and signal-overlap
subsets. Promotion of the blend requires lift >=1.30 and average rate 1--2 in
both years, minimum currency lift >=1.30, minimum quarter rate >=1.00, positive
combined lift and benefit gains, paired lift CI above zero, paired benefit CI
above zero, and Holm p<0.05. A waveform-freshness claim additionally requires
the aligned-versus-stale paired lift interval above zero. Preserve failed gates
instead of changing state count, shrinkage, blend weight or policy.

## Packet BQ: causal CNY shadow-rate nowcast

Frozen after packet BP and after an architecture-only screen on 2024, but
before reading either 2025 or 2026 result for this family. The economic anchor
is the log basis, in basis points, between the close of the last completed
CNYRUB_TOM session (`TRADEDATE < signal_date`) and the latest official CBR CNY
rate already available by that session. A positive basis is a transparent
market nowcast that the current target-currency rate is cheap relative to the
next common RUB move. No target label is fitted by the direct expert.

The 2024 screen compared the raw close and WAP basis, the minimum of
close/open/WAP, current cross-rate z-scores, and fixed +/- one-day and
minus-0.20 five-day target/CNY cross-return corrections. Preserve the complete
screen in the experiment protocol. Advance exactly the unmodified close basis
and the best screened correction `close_basis - 0.20 * cross_return_5`, plus
fixed causal-rank blends of the frozen primary with the close basis at 75/25
and 50/50, and with the corrected basis at 75/25. Keep the unchanged
22%/rolling-20 policy; do not fit a threshold or formula on 2025--2026.

The target/CNY cross at a signal date uses only target and CBR CNY values dated
at or before that signal. Physical corruption of every market or official-rate
observation after a cutoff must leave every earlier raw score bit-identical.
Report yearly, combined, currency and quarter diagnostics, block-bootstrap and
circular-shift multiplicity. Standalone feasibility requires annual lift >=1.30
and rate 1--2. A blend may be promoted only after a separately frozen paired
audit versus the primary; no result in this packet itself promotes a model.

## Packet BR: shadow-nowcast freshness and paired value

Frozen after all packet-BQ metrics were written and before constructing or
evaluating its stale control or any paired difference. Construct exactly one
new negative control by delaying the raw close-basis score by 20 target rows
within each currency; do not refit or otherwise transform it. Test exactly
three paired 2025--2026 hypotheses: fresh close basis versus stale20 close
basis, corrected versus uncorrected close basis, and the fixed 75/25
primary/close blend versus frozen primary.

Use identical-row four-week block-bootstrap lift and forward-benefit
differences, Holm-adjusted one-sided p-values across all three comparisons,
separate fixed-comparison circular-date shifts and overlap subsets. The direct
close basis is independently feasible if both annual lift values are >=1.30,
both annual rates are in [1,2], minimum-currency lift is >=1.30, and its
already-recorded standalone block CI excludes one with circular max-adjusted
p<0.05. Freshness additionally requires its paired lift CI over stale20 to be
above zero. Promotion of the 75/25 blend over primary additionally requires
positive paired lift and benefit lower bounds, Holm p<0.05, annual lift/rate
gates, minimum-currency lift >=1.30, and minimum-quarter rate >=0.95. Do not
repair any failed criterion in this packet.

## Packet BS: full-history transport of the shadow nowcast

Frozen after packet BR and before reading any pre-2024 shadow-basis metric.
Transport the BQ raw close-basis formula without fitted labels, coefficient
changes or policy changes across every complete year 2017--2026. Compare only
the original raw score and an operational availability-gated form that assigns
`-1e9` whenever the last completed CNY session is absent or older than seven
calendar days, ensuring such a day can never fire while retaining it in the
base population. The gate uses only the already-causal missing flag and is not
selected on outcomes.

Use each preceding calendar year solely as the score-threshold calibration
block and keep the 22%/rolling-20 policy. Report all years separately, pre-SVO
2017--2021, transition 2022--2023, post-2024, and full 2017--2026; report
market coverage, minimum annual lift/rate, minimum currency lift and quarterly
cadence. Use four-week block-bootstrap and circular-date multiplicity for both
fixed variants on the full period. Full-history transport requires every
annual lift >=1.30, every annual rate in [1,2], minimum-currency lift >=1.30,
and minimum-quarter rate >=0.75. Keep the post-2024 result separate even if
older market microstructure breaks transport.

## Packet BT: five-step survival and conditional-hazard logits

Frozen after packet BS and before fitting any model in this family. Recast the
future-only h=5 event as five nested survival events: today's rate must not be
beaten after each of the next one through five publications. At every quarterly
refit from 2023, use only post-24.02.2022 rows whose fifth-publication reach is
strictly before the refit date. Fit five independent cumulative L2 logistic
models and five conditional-hazard L2 logistic models. Hazard k is trained only
on rows that survived steps 1..k-1 and predicts survival of step k. Use
StandardScaler plus LogisticRegression C=0.03, maximum 3000 iterations, seed
20260905; do not tune class weights or probabilities.

The aligned matrix contains currency one-hots, causal target range/return/
volatility/calendar fields, all packet-AN CNY basis fields, and the packet-BE
50-field waveform. The stale control delays all basis and waveform fields by
20 target rows within currency while keeping current target/calendar fields.
Compare direct cumulative h5 probability, geometric mean and minimum of all
five cumulative probabilities, product of the five conditional hazards, its
stale20 twin, and a fixed 50/50 causal-rank blend of the hazard product with
the label-free BQ close-basis expert. Keep 22%/rolling-20.

No score may use an incompletely observed training path. Physically corrupting
all five-step outcome paths whose reach is after a cutoff must leave every
earlier aligned score bit-identical. Same-day/future feature corruption is
covered by the basis and waveform builders. Report 2024, both later years,
combined, currency/quarter breakdown, block bootstrap and circular
multiplicity. Fresh hazard information requires aligned hazard to beat stale20
in combined and minimum-currency lift; any promotion needs a separately frozen
paired audit.

## Packet BU: paired audit of the survival decomposition

Frozen after all packet-BT aggregate metrics were written and before
calculating a paired difference or overlap. Do not refit logits, alter the
five-step construction, choose another aggregation, or change cadence. Test
exactly three hypotheses on identical 2025--2026 rows: aligned hazard product
versus stale20 hazard product for market freshness; cumulative-geometric versus
direct h5 logit for value from the nested multi-horizon target; and
cumulative-geometric versus the label-free BQ close-basis expert for best
standalone lift.

Use paired four-week block-bootstrap lift and forward-benefit differences,
Holm-adjusted one-sided p-values across the three hypotheses, separate
fixed-comparison circular-date shifts, and signal-overlap subsets. The
cumulative-geometric model is operationally feasible if both annual lifts are
>=1.30, both annual rates lie in [1,2], minimum-currency lift is >=1.30,
minimum-quarter rate is >=1.00, its standalone lift CI is above one, and its
circular max-adjusted p is below 0.05. Claims of incremental lift require the
corresponding paired lower bound above zero and Holm p<0.05. Preserve a
negative benefit increment rather than changing the aggregation.

## Packet BV: survival-expert fixed consensus

Frozen after packet BU and an explicit 2024-only weight screen, before reading
any 2025--2026 metric for a new consensus. The screen compared survival/
shadow and survival/primary weights 25/75, 50/50 and 75/25, plus equal thirds
and primary50/survival25/shadow25. Advance exactly three fixed causal-rank
scores: primary75/survival25 (best 2024 lift and robustness), shadow75/
survival25 (best primary-free 2024 robustness), and primary50/survival25/
shadow25 as the predeclared three-expert diversity control. Inputs are frozen
packet outputs; do not refit or rescale a component.

Keep 22%/rolling-20 and report 2024, 2025, 2026, combined, currency/quarter,
block-bootstrap and circular multiplicity. No consensus is promoted from point
metrics. Any apparent primary increment requires a separately frozen paired
audit versus `logit50_extra50`; the primary-free consensus is compared with
both constituent standalone experts in that same later audit.

## Packet BW: paired audit of survival consensus

Frozen after all packet-BV metrics were written and before any paired
difference. Test exactly four 2025--2026 comparisons on identical rows:
primary75/survival25 versus primary; shadow75/survival25 versus shadow;
shadow75/survival25 versus survival; and the three-expert consensus versus
primary. Use paired four-week block-bootstrap lift and benefit differences,
Holm-adjusted one-sided p-values across all four, separate circular-date
fixed-comparison tests, and overlap subsets. A consensus increment requires
both paired lower bounds above zero and Holm p<0.05 in addition to annual
lift/rate and minimum-currency lift gates. Do not alter weights after failure.

## Packet BX: pooled discrete-time survival panel

Frozen after packet BW and before fitting a pooled hazard. At every quarterly
post-SVO refit, expand each fully resolved training episode into at most five
risk-set rows: step k is present only when the episode survived steps 1..k-1,
and its label says whether it survives step k. Fit one classifier across the
pooled panel and multiply its five predicted conditional survivals at test
time. Test rows never expose an actual intermediate future outcome.

Compare exactly: StandardScaler+L2 LogisticRegression C=0.03 with five step
one-hots; the same logit with a centered linear step interaction applied to
target range30/90/180, return1/5, CNY close basis, and latest waveform return;
HistGradientBoostingClassifier (200 iterations, learning rate .035, max seven
leaves, minimum leaf 100, L2=20, seed 20260905) on the non-interacted pooled
panel; the plain pooled logit on a 20-row-stale market matrix; the already
frozen separate-hazard product; and a fixed 50/50 causal-rank blend of the
pooled logit with the frozen cumulative-geometric expert. Use the unchanged
97-field packet-BT information set and 22%/rolling-20 policy.

All expanded labels must have h=5 reach strictly before each refit. Physical
corruption of all unresolved future paths must preserve every earlier pooled
score bit-identically. Report 2024, later years, combined, breakdown,
four-week bootstrap and circular multiplicity. Freshness requires aligned
pooled logit to beat stale20 in combined and minimum-currency lift; promotion
requires a later separately frozen paired audit.

## Packet BY: paired audit of pooled survival

Frozen after all packet-BX metrics were written and before paired inference.
Compare exactly three hypotheses on identical 2025--2026 rows: aligned pooled
logit versus its stale20 twin; pooled logit versus the five separately fitted
conditional hazards; and pooled logit versus cumulative-geometric survival.
Use paired four-week block-bootstrap lift and benefit differences, Holm-adjusted
one-sided p-values across all three, separate circular-date fixed-comparison
tests, and overlap subsets. Freshness and a pooling-efficiency claim each need
their paired lift lower bound above zero and Holm p<0.05. Do not change the
interaction set, learner or aggregation in response to this audit.

## Packet BZ: resolved-error router for two independent experts

Frozen after packet BY and before fitting a router. Use only the label-free
shadow close-basis expert and the independently trained cumulative-geometric
survival expert. Recover each row's own-year score, convert it to a same-
currency percentile against at most 250 strictly earlier scores (minimum 20),
and define the resolved router label as which expert's percentile had the lower
Brier loss. Ties select shadow. Router training starts in 2023 and each
quarterly refit admits only rows whose h=5 reach is strictly before the refit.

The gate matrix contains both ranks, signed/absolute disagreement, min/max/
mean, currency one-hots, target range30/90/180 and return1/5, close basis,
CNY waveform volatility5/20, last-return z-score and acceleration, annual and
weekday sine/cosine. Compare C=0.02 scaled logistic and a depth-two decision
tree with minimum leaf 150, each as soft and hard expert selectors; the soft
logit on the entire gate matrix delayed 20 rows within currency; and a fixed
50/50 causal-rank expert consensus. A soft selector uses its probability that
survival was better as the survival weight; a hard selector uses a 0.5 gate.

Physically changing every router label whose reach lies after a cutoff must
leave all earlier weights and scores bit-identical. Keep 22%/rolling-20.
Report 2024, 2025, 2026, combined, breakdown, block bootstrap and circular
multiplicity. Gate freshness requires aligned soft logit to beat stale20 in
combined and minimum-currency lift. No router is promoted before a separately
frozen paired audit against equal consensus and both experts.

## Packet CA: paired audit of the two-expert error router

Frozen after all packet-BZ metrics were written and before paired inference.
Test exactly four hypotheses on identical 2025--2026 rows: aligned soft-logit
router versus its stale20 twin for gate freshness; hard depth-two tree router
versus fixed 50/50 consensus; hard tree versus standalone shadow; and hard tree
versus cumulative-geometric survival. Use paired four-week block-bootstrap lift
and forward-benefit differences, Holm-adjusted one-sided p-values across all
four, separate circular-date fixed-comparison tests and overlap subsets.

Operational feasibility of the hard tree requires annual lift >=1.30 and rate
1--2 in both years, minimum-currency lift >=1.30, minimum-quarter rate >=1.00,
standalone lift CI above one and circular max-adjusted p<0.05. An incremental
claim versus an expert or equal consensus additionally requires paired lift CI
above zero and Holm p<0.05; a joint business improvement also requires benefit
CI above zero. Preserve every failed comparison without retuning the tree.

## Continued packet register CB--DB

Every entry below uses the same strict as-of convention: a local-market or
local-central-bank observation must be dated strictly before the signal date;
CBR reference observations may be dated no later than the signal date; no next
CBR target value is available. Model/formula choice was made on 2024 before
opening the protocol-controlled 2025--2026 table. Exact settings, hashes and
candidate lists are saved in each packet's
`results/research/round6/*/protocol.json`.

| Packet | Frozen question | Recorded outcome |
|---|---|---|
| CB/CC | Can NBT RUB/USD/CNY cross-rates improve TJS, including reversed orientations? | Raw local signals were weak; no promotion. |
| CD/CE | Can label-free expert geometry combine CNY experts more robustly? | Useful independent geometry; paired gain over the strongest parent unresolved. |
| CF/CH | Does the analogous CBA basis add fresh information? | CBA aligned basis beats stale20; `geometry75_cba_consensus_basis25` reaches pooled h5 lift 1.947, but incremental blend CI crosses zero. |
| CG | Recompute exactly the Q&A case lift and symmetric benefit at h=1/3/5/10/20. | CBA geometry blend min/mean lift 1.623/1.855, positive symmetric and future-only benefit at every horizon. |
| CI/CJ | Reweight experts for the five-horizon objective and audit multiplicity. | The 2024-selected three-way blend transfers worse; incumbent's 15 lift and 15 symmetric-benefit tests pass Holm. |
| CK | Train a separate pooled long-horizon committee. | Later min lift 1.456 standalone and at most 1.610 in a low-dose blend; incumbent retained. |
| CL/CM/CN | Test official Uzbek, Kazakh and Kyrgyz central-bank cross-rates separately. | Each source passed causality/unit checks; each 2024 finalist selection retained incumbent. |
| CO/CP | Route between incumbent/regime or geometry/CBA experts using only resolved errors. | Both frozen 2024 screens retained incumbent; no router promoted. |
| CQ | Test official NBG RUB/USD/CNY cross-rates. | Raw 2024 min lift 1.077 and later 0.974; all blends trail incumbent. |
| CR/CS | Test official NBRB daily RUB/USD/CNY and a dense low-dose blend grid. | A retrospective 10% blend has min lift 1.625, but the preselected 30% blend falls to 1.564; no promotion. |
| CT | Choose a separate local-CB formula for each target corridor, then blend globally. | Panel screen min lift 1.105; direct per-currency-expert hypothesis rejected. |
| CU | Replace rolling quantiles with causal exponentially weighted thresholds. | 2024 selected 30%/half-life40; later min/mean lift 1.594/1.819 and weak quarter cadence. |
| CV | Add a frozen pre-2024 weekday/month calendar prior. | 2.5% correction transfers to min lift 1.611 versus incumbent 1.623; rejected. |
| CW | Enforce at most two weekly alerts with confidence-dependent first/second slots. | Later min/mean lift 1.553/1.684 and one negative future-benefit year; rejected. |
| CX | Use the calendar gap since the previous CBR publication as a regime. | Screen gain reverses: later min lift 1.605 and full-lifecycle min 1.549 versus 1.553 control. |
| CY | Aggregate seven strictly lagged local-CB implied USD/RUB and CNY/RUB cross-rates into a robust shadow-RUB consensus. | The 2024-selected 30% negative-dispersion blend transfers at min/mean lift 1.544/1.773 versus incumbent 1.623/1.855; rejected. |
| CZ | Treat cross-bank dispersion as causal uncertainty: nonlinear penalties, confirmations and hard vetoes. | The 2024-selected low-dispersion confirmation transfers at min/mean lift 1.612/1.826; useful diagnostic, no promotion. |
| DB | Fit a quarterly joint logit/HistGB/ExtraTrees layer over target state, both incumbent ranks, CBA and cross-bank features using only resolved h5 labels. | Fresh logit and HistGB beat stale20 controls on 2024, but no joint learner beats the incumbent; the frozen selection remains incumbent. |
| DC | Difference each source against its own earlier cross-rate before aggregating revisions across banks. | No raw revision feature reaches screen min lift 1.01 with both benefits positive; the 2024 selector retains incumbent without using 2025--2026. |
| DD | Normalize every bank against its own strictly earlier 250-date history before estimating a latent cross-bank level. | Standalone screen min lift rises to 1.233, but the selected 10% blend transfers at min/mean 1.593/1.759 versus incumbent 1.623/1.855; rejected. |
| DE | Use one-sided alpha-beta local-linear target states, with full-history and post-SVO reset variants. | Selected 10% blend raises h10/h20 and every symmetric benefit, but lowers h1 to 1.591 and overall mean to 1.845; keep only as a long-horizon challenger. |
| DF | Combine incumbent and local target-state ranks with nonlinear label-free agreement geometries and a matched stale control. | Fresh state beats stale inside the selected formula, but transfers at min/mean 1.598/1.809 versus incumbent 1.623/1.855; rejected overall. |
| DG/DH | Archive lagged MOEX CNYRUBF/USDRUBF perpetual futures and fit quarterly classical learners with matched stale controls. | ExtraTrees transfers at min/mean five-horizon lift 1.534/1.648, annual h5 lift 1.770/1.645 and rate 1.18/1.26; stale20 falls to 1.064/1.123. Strong independent expert, not the leader. |
| DI/DJ | Combine incumbent and futures expert with label-free geometry, then audit all horizons with paired blocks and Holm. | Minimum geometry raises point worst-horizon lift 1.623 to 1.659, but paired CI [-0.183, 0.098] crosses zero and mean lift falls. Freshness versus stale is supported at every h; no promotion. |

The register deliberately keeps attractive failures. In particular, neither
the 10% Belarus point, the post-gap subgroup, nor an unselected joint learner
may be promoted after looking at 2025--2026; they are hypotheses for future
shadow data only.

## Packet DC: within-source cross-bank revision dynamics

Frozen after packet DB and before reading any packet-DC result. Packet CY may
confound a real common move with a change in which local bank is available on
a date. Build a paired panel for each of the same seven official banks first,
then compute each source's own change in implied USD/RUB and CNY/RUB basis over
1, 5 and 20 preceding signal dates. Only after differencing within source may
the packet aggregate across banks.

The fixed raw family contains, for each lag, the positive and negative median
consensus revision and the fractions of banks whose USD and CNY revisions are
jointly positive or jointly negative. Add exactly four multi-scale scores:
positive/negative short-versus-medium acceleration, negative revision
dispersion, and a level-reversion score equal to the negative product of the
cross-bank median level and one-date median revision. Require at least three
paired sources; otherwise emit the neutral score zero. Compare the 2024-selected
raw score with the incumbent at fixed weights 5%, 10%, 20% and 30%.

Select first by maximum worst case lift over h=1/3/5/10/20, then mean lift,
requiring positive symmetric and future-only benefit at every horizon. Open
2025--2026 only once after that selection. Every local observation remains
strictly earlier than the signal date, the CBR reference is no later than the
signal date, and a physical future-corruption check is mandatory. No next CBR
course, target outcome, later membership change, or packet-CY/DB later metric
may enter a packet-DC feature or selection.

## Packet DD: causally normalized cross-bank latent level

Frozen after packet DC and before reading any packet-DD result. Raw cross-bank
levels are not directly comparable because every country has a different
structural spread, regulation and domestic-currency market. For every source,
map the current implied RUB basis to a percentile and robust z-score against
at most 250 strictly earlier signal dates from that same source, with at least
60 finite earlier observations. Score the current value before adding it to
history. Then aggregate the comparable source states across banks.

Test exactly: positive and negative median/mean percentile; lower and upper
percentile quartiles; breadth below 0.25 and above 0.75; positive and negative
median/trimmed robust z-score; negative z-score dispersion; signed median-z to
dispersion; and high/low agreement between the Armenian source percentile and
the cross-bank median percentile. The raw family is label-free. Select one raw
orientation on 2024 and compare its fixed 5%, 10%, 20% and 30% causal-rank
blends with the incumbent. Use the same five-horizon objective, benefits,
rolling-20 22% policy, as-of boundary and one-time later opening as packet DC.
Physical future corruption must leave every earlier normalized feature exactly
unchanged. The 250/60 memory, feature list and blend grid may not be changed in
response to packet-DD results.

## Packet DE: one-sided local-linear target state

Frozen after packet DD and before reading any packet-DE result. Apply a causal
alpha-beta local-linear state filter separately to each of the five target
currency series in log space. This is not an ETS point-forecast contest: the
candidate scores are the standardized one-step forecast gap, negative current
innovation, positive filtered slope, and an equal rebound combination of the
last two. Standardize with the median absolute deviation of at most 120 earlier
innovations from the same currency, never including the current innovation
before scoring.

Use exactly three fixed gain pairs `(alpha,beta)`: (0.10,0.01), (0.20,0.03)
and (0.40,0.08). For each pair run a full-history filter and a mechanically
reset filter whose state and scale history restart at the first publication on
or after 24 February 2022. This gives 24 label-free raw scores. Select one on
2024 using the unchanged five-horizon/positive-benefit objective, then compare
fixed 5%, 10%, 20% and 30% causal-rank blends with the incumbent and open
2025--2026 once. A physical corruption of all target values after a cutoff must
leave every earlier state score bit-identical. The current published target
course is available; no next course or future outcome is used.

## Packet DF: nonlinear incumbent and target-state agreement

Frozen after packet DE and before reading any packet-DF result. Convert both
the incumbent score and packet-DE's selected raw negative-innovation score to
same-currency percentiles against at most 250 strictly earlier scores, minimum
20. Compare exactly ten label-free geometries: minimum, geometric mean,
harmonic mean, 75/25 and 90/10 lower/upper mixtures; incumbent minus 10% or 20%
of a state rank shortfall below 0.5; incumbent plus 10% or 20% of a state rank
excess above 0.5; and incumbent minus 20% absolute rank disagreement.

Select an aligned geometry on 2024 with the unchanged five-horizon and benefit
objective. For each formula also construct a predeclared stale control by
delaying only the state rank 20 rows within currency; after selection compare
the chosen aligned formula, its matching stale control and incumbent on the
one-time 2025--2026 opening. The current score enters rank history only after
its percentile is computed. No target labels choose weights or gates, and no
formula may be changed after later results are visible.

## Packet DG: lagged MOEX perpetual FX futures archive

Frozen before downloading the archive. Fetch `CNYRUBF` and `USDRUBF` from the
official public MOEX ISS history endpoint from 1 January 2022 through the
existing historical cutoff 3 September 2026. Preserve every requested page,
URL, schema, retrieval timestamp and a canonical SHA-256 digest. The contracts
are daily cash-settled futures with automatic extension, avoiding a hand-built
quarterly rollover. Only rows with positive close, settlement and trade count
are price observations.

Every model feature must use `TRADEDATE < signal_date`; same-day futures close,
settlement, volume, funding or open interest are forbidden. CBR USD/CNY
references may be dated no later than the signal date because the current CBR
publication is part of the current information set. Build per-contract lagged
returns (1/2/5/10/20), volatility (5/20), candle shape, close/settlement and
close/VWAP gaps, log volume/trades/open-interest, funding, open-interest changes
(1/5), close/settlement basis to the current matching CBR reference, age and
missingness. Add cross-contract close/settlement CNY-via-USD basis, return
divergence (1/5) and funding spread. Physically changing every futures row on
or after a cutoff must leave all features on or before that cutoff unchanged.

## Packet DH: quarterly perpetual-futures learners

Frozen after packet-DG data validation and before fitting or reading any
packet-DH metric. The fixed matrix contains the 47 packet-DG features, six
target-state fields (`pct_range_30/90/180`, `ret_1/5/20`) and the five currency
one-hots. Training begins 1 May 2022. At every quarterly refit from 2024 onward,
admit only h5 labels whose target-reach date is strictly before the refit.

Fit exactly three classifiers: StandardScaler plus L2 logistic regression
`C=0.025`; HistGradientBoosting with 180 iterations, learning rate 0.03, five
leaves, minimum leaf 100 and L2 30; ExtraTrees with 400 trees, depth six,
minimum leaf 45 and max-features 0.65. Seed all learners with 20260905. For each
learner also fit an otherwise identical control whose 47 futures columns are
delayed 20 rows within currency while target/currency fields remain aligned.

The aligned 2024 candidate set is the three raw learners, the incumbent, and
fixed 10% and 25% causal-rank additions of each learner to the incumbent. Select
by maximum worst official lift over h=1/3/5/10/20, then mean lift, requiring
positive symmetric and future-only benefit at every horizon. Open 2025--2026
once for the selected candidate, incumbent and its exactly matched stale
control. Physical corruption of all outcomes unresolved at a cutoff must leave
earlier prequential scores identical. No next CBR rate, same-day futures value,
later-period metric or post-selection weight adjustment is allowed.

## Packet DI: incumbent and perpetual-futures expert geometry

Frozen after packet DH and before reading any packet-DI metric. Convert the
incumbent and selected fresh futures-ExtraTrees scores to same-currency
percentiles against at most 250 strictly earlier scores, minimum 20, adding the
current value only after its rank is computed. Test exactly ten label-free
geometries: minimum, geometric mean, harmonic mean, arithmetic mean, 75/25 and
90/10 lower/upper mixtures, maximum, high-agreement gates at mean rank 0.65 and
0.75 (maximum above the gate, minimum otherwise), and arithmetic mean minus
25% absolute disagreement.

Select on 2024 by the unchanged five-horizon/positive-benefit objective. For
every formula construct a matched control using the already fitted stale20
futures expert while leaving the incumbent aligned. Open 2025--2026 once for
incumbent, the selected geometry and its exact stale control. Do not tune the
rank memory, formula family, gates or weights after any later result is read.

## Packet DJ: paired multi-horizon audit of futures agreement

Frozen after all packet-DI point estimates were written and before paired
inference. On identical 2025--2026 rows compare the selected minimum geometry
against incumbent and against its matched stale20 geometry at each of
h=1/3/5/10/20. Use the same four-week moving-block draws across all horizons
so every draw also yields a paired difference in the minimum and mean lift over
the five horizons. Keep all currencies and corridor-year base rates inside
each sampled week.

Report paired lift, symmetric-benefit and future-benefit differences with 95%
intervals and one-sided p-values. Holm-adjust each metric family across the ten
predeclared horizon comparisons (five geometry-versus-incumbent and five
fresh-versus-stale). A robust-minimum promotion requires: selected minimum
lift above incumbent in point estimate, paired minimum-lift lower bound above
zero, both annual h5 lifts at least 1.30, both annual rates 1--2, minimum
currency h5 lift at least 1.30, and all five selected point benefits positive.
Freshness is a separate claim and requires its matched paired lift lower bound
above zero after Holm. Do not alter packet-DI after this audit.

## Packet DM: noon-Moscow hourly perpetual-futures archive

Frozen after the 05.09 case-owner Q&A explicitly permitted MOEX as an
intraday indicator, before downloading or evaluating the hourly archive.
Download every public 60-minute candle for `CNYRUBF` and `USDRUBF` from the
official MOEX ISS endpoint, 1 January 2022 through the frozen historical cutoff
3 September 2026. Preserve schema, requested page URLs, retrieval timestamp,
row counts and a canonical SHA-256 digest. Reject duplicate or unsorted candle
start times and non-positive OHLC values.

Define one operational decision time rather than tune it: **12:00 Europe/Moscow**.
For a signal on date T, admit only candles whose `end` timestamp is strictly
before T 12:00; this normally exposes completed candles through 11:59:59 and
forbids the noon candle and the rest of the session. The next CBR fixing is
never loaded as an input. This packet is therefore a distinct, explicitly
timed intraday product variant rather than a rewrite of the previous-day
incumbent.

Build per-contract cutoff features for last price; overnight return from the
previous completed session; open-to-cutoff, last-one-hour and last-two-hour
returns; cutoff high-low range and realised volatility; log cutoff volume;
number of completed candles; log-price slope; last-price position within the
cutoff range; and last-price basis to the current available CBR reference.
Add CNY-via-USD cross basis plus one-hour and open-to-cutoff return divergence.
Every missing/age state is explicit. Physically changing all candles at or
after a row's noon cutoff must leave that row and all earlier rows bit-identical.

## Packet DN: quarterly learners on the fixed noon-Moscow state

Frozen after packet-DM integrity checks and before any packet-DN metric. Reuse
the six target state fields and five currency one-hots from packet DH. Fit two
fixed feature views: the new noon block alone, and the noon block appended to
the 47 previous-session perpetual-futures fields. For each view fit the same
three regularised quarterly classifiers as packet DH (logit C 0.025; HistGB
180 iterations/five leaves/minimum leaf 100/L2 30; ExtraTrees 400 trees/depth
six/minimum leaf 45/max-features 0.65), starting 1 May 2022 and admitting an h5
training label only when its reach date is strictly before the refit quarter.

For each learner fit an exact stale-20 control that delays only the noon block
within currency; static target fields and previous-session daily futures stay
aligned. Screen the six aligned raw learners, incumbent, and fixed 10%/25%
causal-rank incumbent blends on 2024. Select by the maximum worst official lift
over h=1/3/5/10/20, then mean lift, requiring positive symmetric and
future-only benefit at every horizon. Open 2025--2026 once for the selected
candidate, incumbent and its matched stale system. No cutoff, feature,
hyperparameter or blend change is permitted after later-period results appear.

## Packet DO: label-free three-view futures consensus

Frozen after packet DN selected `noon_hist` on 2024 and before evaluating any
new consensus metric. Keep three scores fixed: the incumbent, packet-DH daily
futures ExtraTrees, and packet-DN noon HistGB. Convert each to a same-currency
percentile against at most 250 strictly earlier scores, minimum 20. Evaluate
exactly five two-view formulas on incumbent/noon (minimum, geometric mean,
harmonic mean, arithmetic mean, and 75% minimum plus 25% maximum) and seven
three-view formulas (minimum, geometric mean, harmonic mean, arithmetic mean,
median, lower quartile, and 75% three-view minimum plus 25% three-view median).

Select on 2024 with the unchanged five-horizon/positive-benefit objective and
fixed rolling policy. Construct an exact stale system for every formula by
replacing both market experts with their own pre-fitted stale-20 controls while
leaving the incumbent aligned. Open 2025--2026 once for incumbent, the selected
fresh formula and its matched stale formula. The formula family is label-free;
no target loss chooses the row-level view or a later-period weight.

## Packet DP: paired audit of three-view consensus

Frozen after packet DO point estimates and before paired inference. Compare
the selected consensus with the incumbent and with its matched double-stale
system on the same four-week moving-block draws at h=1/3/5/10/20. Holm-adjust
lift, symmetric-benefit and future-benefit families over all ten horizon
comparisons; also report the draw-wise minimum and mean lift. Promotion needs
a positive lower 95% bound for minimum-lift improvement over incumbent plus
the established annual lift/rate, minimum-currency and all-horizon benefit
gates. Freshness remains a separate claim against the matched stale system.

## Packet DQ: fixed short/long-horizon state balance

Frozen after packet DP and before reading any packet-DQ metric. Treat packet
DO's selected noon/incumbent arithmetic consensus as the fixed short/medium
horizon expert and packet DE's raw selected target state-space score as the
fixed long-horizon expert. Compare only the two raw experts, the incumbent,
and three causal-rank mixtures with state-space weights 10%, 25%, and 40%.
Every percentile rank uses at most 250 strictly earlier same-currency scores
with a minimum history of 20; the current row and all future rows are excluded.

Select once on 2024 by maximum worst official lift over h=1/3/5/10/20, then
mean lift, requiring positive symmetric and future-only benefit at every
horizon. Open 2025--2026 only for that selected candidate and the frozen
comparators. Because later-period diagnostics motivated this coarse family,
label it a retrospective causal challenger rather than a pristine confirmatory
result. If the 2024 screen retains an existing score, stop without narrowing
the grid; no later-period weight tuning is allowed.

## Packet DR: shared official-horizon noon learner

Frozen after packet DQ retained the noon consensus and before fitting or
evaluating packet DR. Use the packet-DM noon state at 12:00 Moscow together
with the same six current-target and five currency fields as packet DN. Expand
each training row into the official horizons h=1/3/5/10/20, append a horizon
one-hot and normalized log-horizon, and fit one shared classifier. A label may
enter a quarterly refit only when that horizon's own reach date is strictly
before the quarter. Compare only the fixed packet-DN HistGB and ExtraTrees
regularizations; do not tune either learner.

At prediction time convert each of the five horizon probabilities into a
same-currency causal percentile using at most 250 strictly earlier values and
minimum history 20. Predeclare four horizon aggregations: minimum, geometric
mean, arithmetic mean, and mean minus one-half standard deviation. For each
raw aggregate also compare fixed 25% and 50% causal-rank mixtures with the
incumbent. Include packet DO's frozen noon/incumbent arithmetic consensus as a
do-nothing comparator. Repeat every new candidate with an exact 20-target-row
delay of only the noon block. Select once on 2024 by maximum worst official lift, then mean lift,
requiring positive symmetric and future-only benefit at all horizons. Open
2025--2026 only for the selected candidate and its matched stale system. This
is a retrospective causal challenger; no later-period hyperparameter,
aggregation, cutoff, or weight changes are permitted.

## Packet DS: noon-Moscow spot-FX archive

Frozen after confirming that the official MOEX ISS hourly endpoint exposes
`CNYRUB_TOM` and `USD000UTSTOM`, before downloading the historical archive or
reading any target metric. Download all public 60-minute candles from the MOEX
currency/SELT market from 1 January 2022 through the frozen 3 September 2026
cutoff. Preserve every requested URL, schema, timestamp, row count and a
canonical payload digest; reject duplicate/unsorted timestamps and non-positive
OHLC. Missing public volume/value fields remain explicit and are not imputed as
observed turnover.

Keep the already frozen operational decision time **12:00 Europe/Moscow**. A
row on date T may use only candles whose `end` is strictly before T 12:00. Build
the same price-state block as packet DM except turnover: last price, overnight,
open-to-cutoff, last-one/two-hour returns, range, realised volatility, candle
count, slope, range position, basis to the current available CBR fixing, age
and missing flags. Add CNY/USD cross-basis and return divergences, plus aligned
spot-minus-perpetual bases and return divergences. Physically corrupting the
noon candle and every future candle must leave all earlier rows bit-identical.

## Packet DT: quarterly learners on pre-noon spot FX

Frozen with packet DS before any historical target score is computed. Reuse
packet DN's six current-target fields and five currency one-hots. Fit the same
fixed logit, HistGB and ExtraTrees regularizations to two views: spot features
alone and spot plus the already frozen noon perpetual-futures block. Train on
`fav_h5` from 1 May 2022 with quarterly refits and admit a label only when its
reach date is strictly before the quarter. For every model fit a matched
20-target-row control delaying only the new spot block; any perpetual block
stays aligned.

For each raw learner compare fixed 25% and 50% causal-rank blends with (a) the
CBA geometry incumbent and (b) packet DO's noon-futures point leader. Screen
the raw models, both frozen comparators, and all fixed blends on 2024 using the
maximum worst lift over h=1/3/5/10/20, then mean lift, with positive symmetric
and future-only benefit at every horizon. Open 2025--2026 once for the selected
candidate and its exact stale control. This is a retrospective causal
challenger; do not tune the instruments, cutoff, feature set, model settings or
weights after the later block is opened.

## Packet DU: signed partial-fixing spot nowcast

Frozen after packet DT retained the noon-futures consensus and before scoring
any hand-built spot formula. Use only the two packet-DS noon bases to the
current available CBR references. The economic sign is fixed in advance:
positive spot-minus-current-CBR basis means ruble weakness in the observable
partial session and therefore a potentially favourable current official rate.
Test exactly six label-free scores: USD basis, CNY basis, their arithmetic
mean, minimum, maximum, and lower quartile. No sign search is permitted.

For every raw score compare 10%, 25%, and 40% causal-rank additions to both the
CBA geometry incumbent and packet DO's noon-futures consensus. Construct an
exact control by delaying only the new spot score 20 target rows within each
currency. Screen on 2024 with the unchanged five-horizon/positive-benefit
objective and open 2025--2026 once for the selected formula and matched stale
control. This is a retrospective mechanistic challenger; do not tune signs,
weights, rank memory or formula after later results are visible.

## Packet DV: delayed online spot-regime weighting

Frozen after packet DU showed a strong 2024 signed-spot screen and weaker
2025--2026 transport, before computing any adaptive score. Combine packet DO's
frozen noon-futures consensus with packet DU's selected signed-spot score. Use
the exact already-audited packet-DK online rule `global/window=250/eta=5` rather
than screen a new grid. Both inputs are converted to same-currency causal ranks
using window 250/minimum 20. Every h=1/3/5/10/20 outcome contributes equal
Brier loss only after its own reach date is strictly before the current signal
date. Also report the two components and static equal blend.

Build the matched control by replacing only the signed-spot input with its
packet-DU stale-20 twin. Physically flipping every unresolved future outcome
must leave all earlier weights and scores identical. Because the component
regime change is already known, this packet is an explicitly retrospective
mechanism test, not a fresh holdout; do not tune window, eta, loss, ranks, or
reset after reading its path.

## Packet DW: paired audit of online spot-regime weighting

Frozen after all packet-DV point estimates were written and before paired
inference. On identical 2025--2026 rows compare the fixed online score against
packet DO's noon consensus and against the exact stale-20 spot system at every
h=1/3/5/10/20. Reuse the four-week moving-block scheme with the same sampled
weeks across horizons so each draw yields paired horizon, minimum and mean
lift differences. Keep corridor-year base rates inside every sampled week.

Report lift, symmetric-benefit and future-benefit differences with 95%
intervals and one-sided p-values, Holm-adjusting each family over ten horizon
comparisons. A scorecard promotion requires positive lower confidence bounds
for both minimum- and mean-lift gain over packet DO, plus the existing annual
lift/rate, minimum-currency and all-positive-benefit gates. Freshness is a
separate online-versus-stale claim. Do not alter packet DV after this audit.
