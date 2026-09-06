# AP10-E: matched effect of knowing the next fixing

2026-09-06, registered BEFORE fitting this packet. User corrected the intended
task: use the TODAY-EFFECTIVE CBR as reference in both before/after publication
scenarios. Changing only the evaluation reference of AP3/AP4 without retraining
does not test the value of new information. Prior price-shape AP10 was paused
before fit; archived volume/value fields are all null, no volume experiment ran.

## Estimand and information sets

Same publication-event dates, latest fixing received under calendar assumption,
18:30 clock, h1/3/5/10/20 next observations counted from current_index. Hit iff
currentprice <= each next h prices. Announced_index=current_index+1 in this panel.
Hence effective h1 is ALREADY KNOWN after publication: not unknown-future skill.
If announcedprice<currentprice, all five hit labels are deterministically zero.
Higher h still have unknown remaining steps. Never call this banking execution.

One target, same train/query dates and same hyperparameters for these information
sets: (1) CBR current-effective prefixes only for EVERY currency, no future
announced value/return/peer statistic; (2) same feature schema, already announced
prefixes as before; (3) announced CBR plus the old133 market features. The third
set is additional market value, not part of the pure CBR-information ablation.
Past-only zeros unavailable known_change/known_change_z/next_effective_gap,
rebuilds announced_* from effective prefix and all peer/reference changes/ages
from effective prefixes. Effective history, currency identity/calendar unchanged.

The rows are own announcement-event dates shared with old packets; conditional
event-support comparison, not a deployable all-day strategy or certified18:00.
Without-new-fixing is an information-withholding ablation at the same time,
not a claim that an earlier clock has exactly the same market environment.

## Fixed training and policies

Three information sets times direct HistGB, standardized logistic, and coarse
survival HistGB. Same17quarter origins2022Q3 onward, train2022+; use the previous
FULL PUBLICATION-h20 maturity cap <origin-2days for BOTH information sets.
This conservative one-extra-observation wait keeps identical training masks;
effective label receipts are independently checked earlier than that cap.
Hist160/.05/15leaves/minleaf40/L2=5/noearlystopping; logistic C=.1/maxiter1500.
No extra trees, tuning, cross-validation reuse, or future-derived regime choices.

Four raw outputs per information set: Hist h5, logit h5, survival h5, survival
mean. For announced and market add the SAME four scores with a deterministic
no-signal gate when announcedprice<currentprice; no fitted threshold/price rule.
Keep past63 CDF40warmup and existing urgent-cap2 timing, at most2/week, >=2days.
Two simple controls: known_change_z with same gate/controller and announced
not-lower sign with3day cooldown. AP3 frozen signals diagnostic only. 23 policies.

Rank/select on2023 only with all five h, minlift>=1.3, everycurrency rate1..2,
positive sym benefit bootstrap lower bound, max2/week, futurebenefit ratio>=.8
of known-sign-cd3 control. Among feasible maxminlift, meanlift tie break;
fallback known-sign-cd3 if none feasible. Frozen AP3 is not a new winner.
Write selection before opening later results. Repeatedly inspected2024-26
remain retrospective; no fresh holdout and no search-adjusted CIs.

## Checks and prespecified comparisons

Independent effective target/reference indexing; past-only future-announcement
corruption for every currency; actual direct/logit/survival training future
feature/immaturelabel corruption; same quarterly masks; known-down gate never
fires, h1 of gated policies100% on events with receipt assumption satisfied.
Compare announced vs past for each fixed model/output; market vs announced;
gate vs ungated same model; gated learned vs known-sign simple. Allh utility,
cadence, year/currency, paired20dateblock uncertainty, selected50 sensitivity.
Preserve original publication-reference figures separately and the initial
fixed-signal re-score. Future search optimizes effective reference per user;
publication reference remains a diagnostic of genuinely unknown next steps.
