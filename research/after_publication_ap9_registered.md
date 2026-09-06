# AP9: available partial feedback, frozen before fitting

2026-09-06. AP8 was progress, pushedd8373c7. No confirmed h5 superiority; a
simple19:30 alternative is preserved, not silently used at18:30. This packet
is ONLY18:30, the original133features,20min marketdelay, latestpublishedCBR
reference, calendar-assumed receipts, identical5755events and early/later masks.

## Availability and censoring

For each fit origin, use only receipt dates strictly before origin minus2days.
Reconstruct each historical event's followup from that available price prefix:
count observed next observations up to20; first strictly cheaper observed step,
or0 if none yet. Equality is survival. Do not inspect values beyond the prefix.
No future realized gaps, observation counts, followup or failure IDs are features.
Administrative censoring in observation time may still vary with calendar and
regime; causal availability alone does not prove non-informative censoring.

Directh5 controls require fullobserved20; fresh directh5 requires observed5.
Do NOT include early known failures in directh5 beforeobserved5: that would
asymmetrically select labels. Coarse survival endpoints1/3/5/10/20: control
requires observed20, fresh uses only fully observed intervals, at risk at their
start. A failure inside an incompletely observed coarse interval is deliberately
not added yet, just like a survivor in that interval. Fine survival has20
one-observation steps, so every available step contributes a Bernoulli term
until first failure or censoring. Censored future steps are not negative labels.

## Twelve fitted variants,34policies

Six models at quarterly and monthly refresh, from July2022, expanding since2022:
direct_full20, direct_mature5; coarse_full20, coarse_partial;
fine_full20, fine_partial. Same HistGB parameters asAP2/AP4, no early stopping,
no hyperparameter search. Hazard covariates are original-event features plus
one-hot interval identity (5or20). Equal weight per observed at-risk interval.
Prediction does not condition on a query's future surviving to any interval.
Survival is product(1-hazard), evaluated at five target h. Both coarse/fine
full20 controls isolate the impact of censoring from changing interval size.
Quarterly versus monthly full20 controls isolate refresh frequency.

Direct models produce p5 and50/50 priorCDF(CNY)/priorCDF(p5) policies.
Hazard models produce p5, mean probability acrossfiveh, and CNY50/p5CDF50.
4direct*2 +8hazard*3 +2exactnamedAP3/AP4controls =34policies.
Unchanged prior63CDF/warmup40 and urgent_cap2 controller; no futureweekranking.
Quarterly direct_full20 and coarse_full20 must reproduceAP8/AP4 exactly.
Preserve raw AP2 and CSVroundtrip AP4 feature representations respectively.

Same AP4 early2023 joint selector: allhpointlift>=1.3, rates1..2, max2/week,
allhsymmetriclowerCI>0, futurebenefit>=.8AP3eachh. Write selection before late
scorecard.2023 and2024-2026 repeatedly viewed, no freshholdout or search-adjusted
CIs. Keep all negative results; no selecting the nicest late metric.

## Checks and artifacts

Synthetic likelihood equivalence and known hazard recovery under independent
administrative censoring; coarse full20 identity with oldperson_period; ties,
zero/partial exposure, completedinterval rule, no sample afterfirstfailure,
prefix/futureprice and futurefeature corruption, exactmaturityboundary.
Record everyfit's row/interval/label hashes, eventcounts, recentadditions,
failure/riskcounts, maxlabelreceipt, origins and all issued probabilities.
Preserve per-origin observed counts and observed firstfailure to audit every
training record without treating final outcomes as alwaysknown. Allhprobability
and signal checks, sameoutcomes/support, paired20/50datebootstrap, yearcurrency
and benefit/gap diagnostics. Tests/PDF/summary/checkpoint and scopedcheckedpush.
Goal stays active after this bounded packet.
