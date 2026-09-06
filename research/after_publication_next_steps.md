# Active research checkpoint: already published next CBR fixing

Updated 2026-09-06, T8B complete. The user made after-publication research the
primary indefinite task. The hourly heartbeat `automation` was DELETED at the
user's explicit request. An ACTIVE TARGET drives continuous work in the same
thread and ivan-experiments. Do not restart round7 as the main task, wait for a
schedule, stop at another audit, or mark the goal complete after this checkpoint.

## LATEST COMPLETED CHECKPOINT: T8B ANY-TIME ROUTER

The main-branch intermediate presentation has no literal app mockup, but slide
12 defines an on-screen indicator, a level notification and recipient
selection. T8B now supplies the first two through a single causal contract:
temperature 0--100, probabilities and expected future-only CBR basis points for
h=1/3/5/10/20, timestamps, freshness, phase, confidence, source and a separate
sparse push flag. The persisted 2024--2026 artifact contains 45,990 unique
currency/event snapshots and supports an admissible query on every calendar day
for all five currencies. The interface mapping is frozen in
`research/interface_prediction_contract.md`; the query function is
`ml.transfer_temperature.score_snapshot_as_of`.

Phase quality must be communicated honestly. T4 premarket history-only is only
a limited-confidence fallback: opened 2025--2026 AUC is about .594/.604/.596 on
h1/3/5 and it does not beat the prior on all longer horizons. T5 uses completed
market candles and improves mean all-horizon AUC from .7568 at 10:30 to .7852 at
15:30 while mean Brier falls .15622 -> .14767. T3 provides post-window 16:30 and
17:30 calibration without tomorrow's CBR; T6 expected-benefit MAE improves from
the 135.26 bp prior to 120.45 at 15:30 and 120.17 at 16:30, but the premarket
benefit head is slightly worse than prior and should not be trusted for magnitude.

AP50/AP51 handle the saved after-publication decision. AP50 overall calibrated
Brier on h3/5/10/20 is .16479/.16298/.16374/.13292. AP51 MAE is
56.65/85.91/132.68/197.08 bp versus prior 93.80/113.06/148.78/206.96. T7 was
invalidated after T8 exposed that the base timestamp was already 18:30 with a
20-minute market delay, so its supposed post-receipt candles overlapped the base
information. Corrected T7B adds only newly eligible candles: at 19:00 opened
mean Brier improves .150707 -> .149202, mean AUC .712446 -> .716130 and benefit
MAE 123.55 -> 122.74. The 20:00 update lowers Brier similarly but harms AUC.

T8B audit verifies source hashes, source_at<=valid_from, finite/bounded heads,
unique snapshots, every-day queryability, future-snapshot prefix invariance,
weekend stale behavior and push only on the after-decision snapshot. Push counts
AMD/KGS/KZT/TJS/UZS are 142/139/138/139/137 across the full open period. The
artifact still does NOT certify historical CBR receipt timestamps or executable
bank economics. Production must replace calendar-assumed receipt with observed
events. Next accuracy focus: strengthen morning discrimination, keep the prior
when morning magnitude regression is worse, validate receipt/event timestamps,
and test phase-specific ensembles without retuning on the same open years.
Full suite after T8B: 320 tests passed.

## LATEST PRODUCT GOAL: continuous transfer temperature plus sparse push

The user explicitly added a second output on 2026-09-06. Keep the sparse push
policy optimized for the strongest decisions and ТЗ lift, but also produce a
continuous per-currency "transfer temperature" for a user who opens the app at
any later moment. The widget must expose a calibrated 0--100 score, horizon,
data timestamp/freshness, decision phase and uncertainty; it must not present a
raw rank as probability or claim bank savings without executable quotes/fees.

The two products share causal features but have different evaluations. Push:
lift, 1--2 signals/week, benefit and future-only diagnostics. Widget: OOS Brier/
log-loss, reliability/calibration error, rank discrimination, expected
future-only benefit calibration, stability and stale-data behavior. On daily
CBR events alone the after-publication score is necessarily piecewise constant
within a day. A genuinely changing intraday widget requires timestamped market
or bank quotes available at each `as_of`; never backfill end-of-day values into
earlier queries. Detailed acceptance contract:
`research/after_publication_temperature_goal.md`.

AP49 already supplies four continuous causal OOS head scores for h=3/5/10/20;
the next widget stage must calibrate them using mature prior labels only and
provide a latest-valid-snapshot API before claiming any intraday validation.

## LATEST USER CLARIFICATION: TODAY-EFFECTIVE REFERENCE, not announced reference

During AP10 user explicitly said lift should start from CURRENT EFFECTIVE CBR,
even after receipt of tomorrow's fixing, and expected a strong benefit from
knowing it. Follow this clarification over the older goal wording about latest
announced reference. The overarching after-publication/causality/allmetrics goal
continues; publication-reference remains a separate diagnostic, not main target.
Do NOT go back to optimizing AP3's announced-price target as the main task.
Do NOT claim that rescore of old AP3 signals proves knowledge is harmful.
Models must be retrained to the SAME effective target in with/without ablations.
TЗ note updated: initial announced-reference choice was our conservative team
interpretation, not an organizer ruling. Bank execution still NOT validated.

## Latest completed AP40-E/AP45-E: weekly stopping and nonlinear meta quality

AP40 registered one weekly optimal-stopping target before scoring. For each
eligible AP26/AP23 opportunity, take-now=1 iff mean(y3,y5,y10,y20) is no worse
than every later opportunity of the same currency/ISO week; target maturity is
the latest publication-h20 maturity among all compared opportunities. Twenty
current-only features feed quarterly standardized LogisticRegression C=.1,
balanced, half-life730, mature-before-origin-minus2d. AP37 frequency guards and
fallback remain causal. It passed early but missed late cadence. Late
h3/5/10/20=2.453733/2.540882/2.500001/2.511429,minlift2.453733,
meanlift2.501511,minrate.947598,h5=649 signals/sym78.3044/fwd133.9364. Versus
AP37 it keeps648,removes47,adds1. Lift deltas +.0251/+.0317/+.0207/-.0136 all
20/50-date CIs cross0; symmetric bp deltas on h3/h5/h10 have positive20- and
50-date CIs. Preserve as accuracy-mode, not strict leader.

AP41 froze rate floor1.10 and restored32 AP40 rejections, but late minrate=.985808
and point metrics largely reverted: 2.430505/2.507634/2.469783/2.507969. AP42
used AP38 quality plus AP23 binary substitution; early failed, late minrate
.962882 and h3/h5=2.408154/2.469831 despite h20=2.569100. AP43 used AP23
continuous pre-controller pace>.55/reserve>.70: minrate1.016009 but h3/h5 fell
to2.359622/2.374719 and one empty month; paired20/50-date h3 and h5 deltas to
AP37 are significantly negative. Do not use broad external-rank substitution.

AP44 registered a new 28-feature quarterly CatBoostClassifier for mature y20:
240 trees,depth5,half-life730,balanced classes, OOS quarter fits. Added currency
one-hot, annual sin/cos and fixed post-2022 flag to AP40 features. Late
h3/5/10/20=2.419814/2.511389/2.485529/2.560249,minrate.947598. All lift deltas
to AP37 cross0. Annual sin/cos account for about43% mean importance; treat as
transport risk, not proof. AP45 required both AP38 and AP44 to reject and used
rate floor1.20. It passes point gates at2.428794/2.505628/2.476403/2.522759,
minrate1.016376, but removes0 AP37 signals and adds2; every delta CI crosses0.
AP37 remains the simpler practical strict leader.

AP40 independent audit rebuilds weekly labels/latest maturity,20 features,17
quarterly fits,predictions and router. AP42-AP45 combined audit rebuilds all
inputs/models/routers. Monday-boundary corruption of every future feature,
label,rank and upstream decision leaves all prior predictions/signals exact.
Full suite:299 tests. Current reports:
output/pdf/ivan_after_publication_ap45_effective.pdf and
output/pdf/ivan_after_publication_best_simple_explained.pdf. Both were rendered
page-by-page and visually checked. Next honest classes: horizon-specific models
only if product semantics permit, leave-one-year-out seasonal residual, actual
receipt timestamps, executable bank quotes, or new prospective shadow data.

## Latest completed AP37-E/AP39-E: mature precision and guarded regimes

AP37 registered exactly one disagreement-tail policy before late scoring. Three
frozen causal OOS ranks (AP26 y20,AP34 residual,AP35 distributional) define
support count0..3 at the existing top30 boundary. Expanding precision uses strict
min(y3,y5,y10,y20), only publication-h20 mature history with2day embargo, and
hierarchical shrinkage40 overall->stratum->currency. AP26 core stays exact;
AP23 late-week fallback requires local stratum precision>=causal overall, while
silence10 rescue remains unconditional.

AP37 passed early and all late strict point gates. Late h3/5/10/20=
2.428674/2.509188/2.479316/2.524987,minlift2.428674,meanlift2.485541,
minrate1.008734,zeroempty,max2/week. H5 has695 decisions,rate1.038420,
currency1.02348..1.06083,sym74.5514,fwd133.7337. It keeps693 AP33 decisions,
removes11 and adds2; reasons673 core/1 precision lateweek/21 silence10. Lift
deltas vsAP33=.0082/.0192/.0084/.0087; every20/50date CI crosses0. AP37 is the
new frozen strict point leader, not an independent holdout winner.

AP38 registered one core-regime veto using a separately matured core precision
pool; low-quality core may be skipped only when current trailing365 rate>=1.
It failed early cadence and late minrate=.955240, but is a useful accuracy upper
bound: h3/5/10/20=2.431555/2.520964/2.485514/2.597393. H20 delta vsAP37=.0724
has20date CI[.0065,.1543], while50date crosses0. Do not promote.

AP39 registered one product-derived repair, not a grid: weak core veto only when
n/(elapsedweeks+1)>=1. It still failed early and late cadence. Late h3/5/10/20=
2.434597/2.508938/2.478471/2.572246,average h5rate=.999573 but mincurrencyh5
=.978655 and all-h minrate=.970524. Do not tune2/3/4week runway on opened data.

All AP37-AP39 audits rebuild targets, three ranks, both maturity pools, support,
counts/shrinkage, rate/silence/reasons/veto/runway/cap, early selection and
future expert/label/source corruption prefix invariance. Full suite:289 tests.
Current report: output/pdf/ivan_after_publication_ap39_effective.pdf.

## Latest completed AP34-E/AP36-E: residual survival, distributional CDF, Hedge

AP34 registered one continuous-residual architecture before fit. The target is
effective_floor20-known_change. Seventeen quarterly OOS standardized Ridge fits
use only publication-h20-mature rows, a two-day embargo and train-only 1/99%
winsorization. A causal per-currency error state and train-only residual CDF turn
the estimate into survival probability. It passed early and late strict point
gates: h3/5/10/20=2.411500/2.468417/2.427785/2.432848,minlift2.411500,
minrate1.046943,zeroempty,max2/week. H5 has718 decisions,sym72.9311,fwd130.3284.
It is below AP33 at every horizon; all paired lift delta CIs cross0.

AP35 registered one monotone distributional CatBoost. Every mature train row is
expanded across five fixed anchor offsets -200/-100/0/+100/+200bp; a positive
anchor constraint learns a coherent residual-floor CDF. It passed early but late
h3 failed: 2.391461/2.449391/2.414259/2.402068,minrate1.039301. H20 is
significantly worse than AP33: delta-.114,20-date CI[-.252,-.021]. Do not repeat
synthetic-anchor distributional smoothing as a broad pace expert.

AP36 registered a causal mature-only Brier Hedge across frozen AP26,AP34,AP35
scores. Same-currency ranks exclude the current row; trailing730 losses use only
y20 outcomes mature at least two days earlier, with global/currency shrinkage and
fixed softmax temperature25. Average late weights=.5883/.3208/.0909. It passes
strict point gates at 2.401057/2.448897/2.436977/2.417616,minrate1.031659,
zeroempty,max2/week, but is significantly worse than AP33 on h5 and h20:
CI[-.0978,-.0039] and[-.2069,-.0258]. Average Brier competence is not rare-tail
decision competence.

All AP34-AP36 audits reconstruct 17 fits, targets, maturity, clipping, CDF,
monotonicity probes, ranks, mature losses/counts/weights, nested controller state
and future-corruption prefix invariance. Full suite:284 tests. Current report:
output/pdf/ivan_after_publication_ap36_effective.pdf. AP33 remains frozen leader.

## Latest leader AP33-E: calendar-aware decision fallback

AP33 froze exactly one policy before scoring. AP26 y20-shrink200 decisions keep
priority. AP23 fallback requires the combined stream's prior trailing365 rate<1,
84-day warmup, and either Thursday/Friday with no earlier current-week decision
or >=10 calendar days since the previous combined decision. A new sequential
max2/week follows. No scores, outcomes or future-week state are read.

The candidate passed early2023 joint gates and all late strict gates. Late
h3/5/10/20=2.420521/2.489967/2.470952/2.516300,minlift2.420521,
meanlift2.474435,minrate1.008734,zeroempty,max2/week. H5 has704 decisions,
rate1.051868,currency1.02348..1.06830,sym74.5964,fwd132.9394; reasons673 core,
18 late-week fallback,13 silence10 fallback. Versus AP32 lift deltas are
+.0174/+.0211/+.0153/+.0271 and all20/50-date CIs cross0. Future-only deltas
on h3/h5 are positive under both block sizes; h10 is positive only at50-date.

Early AP33,AP32,AP26 and several controls are identical, so early pass validates
the gate but does not independently validate calendar-filter superiority. AP33
is the new best strict point and frozen challenger, not an independent holdout
winner. Audit rebuilt inputs,targets,rate,warmup,silence,week,reasons,cap and
future-source prefix invariance. At the AP33 checkpoint the suite had279 tests;
its report remains output/pdf/ivan_after_publication_ap33_effective.pdf.

## Latest completed AP28-E/AP32-E: hierarchy, survival and decision routing

AP28 tested exactly one hierarchical pooled y20 CatBoost: all mature eligible
rows weight1, hard-pool rows weight4, plus causal rolling/reserve ranks and
hard-pool flag. Common prior removed cold start but diluted the sparse signal.
No early pass; late h3/5/10/20=2.408359/2.457002/2.411398/2.377662,
minlift2.377662,minrate.978166,h5=2.457002,693 signals,sym74.1154,fwd131.2186.

AP29 used AP23 soft730 mature-competence only as a backstop to the AP26
y20-shrink200 core, with the AP27 r60 state. It passed early, but late
h3/5/10/20=2.397969/2.456970/2.444863/2.463840,minlift2.397969,
minrate1.001092: strict h3 failed. AP30 fixed a causal rank blend
.75*AP26+.25*AP23 under the exact AP21 policy. It did not pass early and late
minlift=2.397866,h5=2.441546,minrate1.008734. Fixed blending dilutes the core.

AP31 tested a new factorized survival CatBoost score P(y3)*P(y20|y3=1), with
34 quarterly OOS mature-only fits. It passed early selection, but late
h3/5/10/20=2.386219/2.433696/2.388902/2.333349,minlift2.333349,
minrate.993450. The two stages compound calibration errors; direct AP26 y20 is
stronger.

AP32 froze exactly one decision-level router, without mixing scores. AP26
y20-shrink200 decisions have priority. Only when core does not fire and the
meta-router's own trailing365 rate is below1 may an AP23 decision fire; a new
sequential max2/week is then applied. It passed early selection and all late
strict gates. Late h3/5/10/20=2.403143/2.468871/2.455698/2.489157,
minlift2.403143,meanlift2.454217,minrate1.039301,zeroempty,max2/week. H5 has
720 signals,rate1.075774,currency1.05336..1.09072,sym73.4633,fwd130.1380;
reasons673 AP26 core/47 AP23 fallback. Deltas vs AP23 are
-.00349/+.00282/-.00881/+.01710 and all20/50-date CIs cross0. AP23 remains best
strict minimum; AP26 remains accuracy frontier with rate failure; AP32 is the
new early-selected strict Pareto challenger, not a statistically proven winner.

All AP28-AP32 audits independently rebuild fits/features, maturity, policy state,
reasons, cap and future-prefix corruption invariance. Full suite:277 tests.
Current report: output/pdf/ivan_after_publication_ap32_effective.pdf.

## Latest completed AP23-E/AP27-E: adaptive pace and low-data specialists

AP23 estimated expert competence only from matured past top30 decisions on
mean(y3,y5,y10,y20). Global precision shrinks to .5 with strength40; currency
precision shrinks to global with strength40. Updating the primary expert hurt.
The useful design updates pace only: `soft730_pace_competence_dual_month24_cap2`
late h3/5/10/20=2.406634/2.466054/2.464507/2.472062,minlift2.406634,
minrate1.039301,zeroempty,max2/week,h5sym73.8195,fwd131.0136. It did not pass
early cadence (.9847), so preserve as best strict late point, not selected/fresh
winner. Deltas vs AP21 are +.0045/+.0198/+.0355/+.0610; all 20/50-date CIs cross0.

AP24 tested five direct grouped CatBoostRankers with85 quarterly OOS mature-only
fits. The early-selected ranker-primary transported badly: late minlift2.232731,
h5=2.289234. YetiRank pace recovered min2.397212,h5=2.463666 but remained below
AP23. Do not promote or repeat direct ranker as primary without regime transport.

AP25 trained five models only on an outcome-free hard cadence pool. Best y20
specialist reached minlift2.403196,h5=2.473910,fwd136.4492 but minrate.939956.
The pool is genuinely sparse: 0 rows until2023Q2,74 by2024-01,409 by2026-07.
AP26 fixed cold start by pre-specified n/(n+k) shrinkage to global CatBoost.
`y20_shrink200` gives h3/5/10/20=2.417356/2.483314/2.445996/2.495087,
minlift2.417356,h5rate1.007044,sym74.4532,fwd135.3075, but all-h min currency
rate=.970524. Preserve as the best accuracy anchor, not a strict product policy.

AP27 added CatBoost backstop only after specialist pace fails and causal
rate/silence deficit is present. Early-selected r70 late minrate=.985808 fails.
The sole strict late policy `s200_cat95_r60_backstop_month24_cap2` gives
h3/5/10/20=2.405426/2.464633/2.444496/2.467493,minlift2.405426,
minrate1.001092,zeroempty,max2/week. H5 has691 signals,rate1.032444,sym73.6733,
fwd131.7880; reasons613 primary/46 pace/26 backstop/6 month. Deltas vs AP21 are
+.0033/+.0184/+.0155/+.0564 and all paired lift CIs cross0. H5 symmetric vs AP17
is +4.9258bp with positive20-date CI, but lift/future are not significant.

All AP23-AP27 audits independently reconstruct maturity, scores/counts/ranks,
state/reasons, policies and future-prefix corruption. Full suite:271 tests.
Current report: output/pdf/ivan_after_publication_ap27_effective.pdf.

## Latest completed AP18-E/AP22-E: CatBoost and dual-expert strict challenger

AP18 froze six full/recent score blends under the exact AP17 controller. The
early-selected 50/50 full/recent candidate reached late h3/5/10/20=
2.387024/2.447501/2.409138/2.468434, minrate1.016376,zeroempty,max2/week.
Its h5 delta vs AP17 is +.010078 CI[-.030982,.054304], not significant.

AP19 added truly new multi-horizon ExtraTrees and CatBoost scores plus residual
Hist/Ridge/PairLogit. CatBoost mean utility was the useful new expert:
h3/5/10/20=2.369258/2.463342/2.506324/2.466066,sym73.0199,fwd127.8248.
The residual stacks were worse. AP20 fixed raw and causal-rank AP18/Cat blends;
the selected 25/75 candidate reached min2.363420 and did not improve the frontier.

AP21's productive architecture separates roles. Rolling ExtraTrees is primary;
CatBoost utility is used only for deficit pace decisions when trailing365
currency rate<1, with AP17 p55/r70/month24/veto/cap2 state unchanged. The
registered late diagnostic `roll_cat_dual_pace_month24_cap2` gives h3/5/10/20=
2.402124/2.446225/2.428982/2.411100,minunknown2.402124,average rates
1.042173/1.044397/1.037204/1.031659,min currency rates
1.021299/1.023479/1.016129/1.008734,zeroempty,max2/week. h5 sym72.9005 and
future130.2379. This is the first registered strict point result above2.4 across
all unknown horizons, but it was NOT early-selected and is not a fresh winner.
All h-wise deltas vs AP17 have 20-date CIs crossing zero. Freeze it only as the
next prospective challenger. Local-Cat dual pace reaches h5=2.498576 and
minlift2.431360 but minrate.970524, so it is accuracy upper-bound only.

AP22 tested eight causal rolling/local rank-consensus primary scores with Cat/AP12
pace. No fresh candidate passed early joint gates; best fresh late min2.338042.
Do not repeat outcome-free rank consensus. All AP18-AP22 independent refit and
future-prefix corruption audits passed. Full suite:257 tests. Current report:
output/pdf/ivan_after_publication_ap22_effective.pdf.

## Latest completed AP15-E/AP17-E: strict cadence with deficit pacing

AP15 froze5 Extra policies: top31.25,narrow adaptive,silence14+month and their
month variants. All passed early. Selected silence14_month24 late h5=2.475700,
rate1.005550,zero empty months,mincurrency.986126. No AP15 fresh policy passes
late strict h5/all-h: month rescue fixes gaps,not corridor rate. Negative saved.

AP16 froze5 new pacing controllers. Early selected pace365_p55_r70: primary
top30; after84days and only while trailing365 decision rate<1, allow Extra
rank>.55 jointly with reserve rank>.70. Late h3/5/10/20=
2.383619/2.446617/2.403340/2.480249,minlift2.383619; minrate1.031659,h5rate
1.063821,currency1.045891..1.075774. Sym68.7693,fwd130.5701. It beats strict
top35 +.071788 CI[.020143,.121321] and reserve7 +.106379 CI[.008863,.200504],
but leaves2 empty currency-months.

AP17 protocol then froze EXACTLY ONE candidate before scoring: selected AP16
pacing plus month24/reserve70 when current currency-month is empty. It passes
early and late. Late h3/5/10/20=2.375650/2.437423/2.392312/2.465753;
minunknownlift2.375650,minrate1.031659,h5rate1.065315,currency
1.045891..1.075774,zero empty months,max2/week. Sym68.7475 CI[55.4998,84.1933],
future130.1724 CI[106.0713,152.9061]. h5 delta vsAP12=-.037805
CI[-.113234,.022645], no proven loss; vs top35 +.062594 CI[.010720,.110466];
vs reserve7 +.097185 CI[.006393,.186485]. 50date keeps positive signs.

AP17 changes AP16 by2 month rescues/1 displaced pace/net+1. It closes all strict
gates without global threshold widening. Full audits rebuildtargets,score,
ranks,365rate,reasons,month-first,veto,weeklycap,prefix corruption,selection and
paired20/50date uncertainty. Report output/pdf/ivan_after_publication_ap17_effective.pdf.
Full242tests passed.

## Latest completed AP14-E: light cadence repair preserves 2.48

Protocol frozen before all AP14 scorecards. Same5755 effective-reference rows,
known-down veto,early2023 selection/opened2024-2026 later diagnostic. Reuse
frozen AP13 roll2/local and AP12 Extra scores; add outcome-free50/50 roll/local.
Five controllers each: top32.5,top35,silence14/r80,silence21/r70,adaptive105.
20fresh policies. All prior250 causal ranks,warmup40,max2/week; adaptive uses
only trailing decisions, no labels.14 pass early joint gates.

Selected BEFORE late: extra_roll2_silence21_r70_cap2. Late h3/5/10/20=
2.425379/2.485115/2.423254/2.431420; h5rate.968196,mincurrency.933831,
sym75.9683,fwd137.0053. Accuracy delta vsAP12 fullExtra=.009887,
CI[-.075784,.096772], but late cadence fails. Preserve as honest selected result,
not product winner.

Most useful late diagnostic: extra_ap12_silence14_r80_cap2. Against frozen AP12
it adds28/removes5 decisions, net+23. Late h3/5/10/20=
2.395377/2.480451/2.360775/2.417480,minunknown2.360775. h5rate1.004055,
currency.986126..1.023479,sym67.2550CI[54.1800,83.4382],fwd133.0959
CI[110.0200,156.4562]. Delta vsAP12 +.005223 CI[-.027293,.036963]: cadence
is nearly free. Delta vs AP13reserve7 +.140213 CI[.038489,.259375]. One empty
full month and two currencies.986 mean near-cadence, NOT literal strict pass.

Literal strict diagnostic: extra_ap12_top35_cap2. Late h3/5/10/20=
2.326256/2.374829/2.305872/2.369384,min2.305872. Rate across all unknown h and
currencies>=1.016, h5currency1.031..1.113,zero empty months,max2/week. It loses
significantly vsAP12: -.100399 CI[-.171546,-.037851], but beats AP10 and AP11.
Adaptive105 h5=2.433364,rate1.030950,mincurrencyh5=1.008538 but rare empty
months. Fixed50/50 blend did not improve frontier.

Audit regenerated targets,frozen scores,exact blend,all20 policies/controller
state,early selection,known-down/cap and all-policy future-corruption prefix
invariance. Paired20/50date CIs complete.229tests passed. PDF
output/pdf/ivan_after_publication_ap14_effective.pdf is the current report.

## Latest completed AP13-E: recent/local models and cadence Pareto-front

Protocol was frozen before fits. Same5755 rows,17quarterly origins,
early2023 selection/later2024-2026 diagnostic, publication-h20 maturity cap and
2-day embargo. Six new causal scores: rolling730/1095-day ExtraTrees,
decay730-weighted ExtraTrees, per-currency localExtra shrunk global, frozen
AP12 Extra/localHist meta-router and same-date mature-only Brier365 router.
Three policies per score: primary top30 prior250 cap2; reserve7 after silence;
month24 rescue.18fresh policies +8controls. Select only fresh on 2023,
h3/5/10/20,rate1..2,symCI>0,max2/week,fwd>=.8knownsign,no empty full months.

Selected BEFORE late: router_extra_local_month24_cap2. Late h3/5/10/20=
2.324480/2.348520/2.315348/2.324251; h5rate1.027962,
currency.978655..1.083240,sym69.7836,fwd125.527,maxgap45,noemptyfullmonths.
It beats AP10 and AP11hazard, but is significantly WORSE than AP12 fullExtra:
20-date deltaCI[-.250426,-.008052],50-date[-.249022,-.005452]. Do not replace
AP12 with this router. Meta lateBrier.225916 and Brier-router.218253 are worse
than AP12.211861; complex routing did not generalize.

Late point diagnostics found stronger new candidates. localExtra primary h5=
2.531074,rate.905443,hit.745875; rolling2 primary h3/5/10/20=
2.435721/2.517586/2.470410/2.468674 and has best all-unknown-h minimum2.435721.
Neither increment over AP12 fullExtra2.475228 is proven: local deltaCI
[-.022661,.149765],rolling2[-.038610,.125435]. Both prove improvement over
AP1cap2: local[.052128,.388392],rolling2[.021824,.389225]. These are strong
next-period challengers, not fresh-holdout records.

Cadence repair is useful but expensive. rolling2 reserve7 h5=2.340238,
rate1.154963,currency1.1206..1.1953,hit.6895,sym73.9568,fwd121.996,maxgap29,
noemptyfullmonths. rolling3 reserve7 h5=2.352265,rate1.177375,maxgap27.
Reserve adds roughly150 events and rolling2 loses significantly vs AP12:
deltaCI[-.260273,-.024701], though it proves gains over AP10[.119026,.453165]
and AP11hazard[.059059,.354456]. Best AP13 probability accuracy is decay730
Brier.210891, but its lift does not prove improvement over AP12.

Latest rolling2 train-only impurity leaders: annual_sin.0713,
market_cny_basis_post_z.0520,known_change_z.0487,market_cny_basis_last_z.0390,
announced_ret1.0350,market_cny_basis_mean_z.0349,TJS_ret1.0303,
known_change.0301,effective_vol20.0254,USD_ret1.0222. Treat as correlated-tree
importance, not causal attribution. Audit rebuilt hashes,targets,17masks,
102fit/router logs,Brier weights,scores/signals,veto/cap/prefix.224tests passed.
PDF output/pdf/ivan_after_publication_ap13_effective.pdf rendered4pages/checked.

## Latest completed AP11-E/AP12-E: conditional remainder and strong ExtraTrees

AP11 protocol/model/audit/results now preserved locally. Same5755 rows,
17quarterly origins, early2023/later2024-2026, publication-h20 shared maturity
cap, today-effective target,18:30/20min market delay. Train conditional models
only when announced>=current; known-down veto; h1known and excluded from learned
unknown labels. Eight model families/25policies. Early selector mistakenly but
pre-registered ranked minallh includingknownh1; selected margin-ridge-local-mean
late h5=2.066759. Do not retrospectively replace it.

AP11 conditional hazard Hist h5 late h1/3/5/10/20=
1.938584/2.109131/2.135019/2.179477/2.192754,rate1.328282,sym60.146795,
fwd90.379656. h5delta vsAP10 knownz CI[.024340,.144939], allunknown-h mean
deltaCI[.024549,.118186]. This is a preserved significant point challenger,
not AP11-selected winner. AP1 exact matched h5=2.259851,maxweekly4; AP1cap2
h5=2.300704,rate.998079 but earlyrate.834862 and benefitLBnegative.

AP12 protocol frozen before fits and corrects selection horizons to h3/5/10/20;
h1 validity only. Five fresh scores: full ExtraTrees400/depth8/leaf25/maxfeat.6;
compact59-feature Hist and ExtraTrees; per-currency compactHist shrunk global
n/(n+150); multiclass first failure.16fresh policies prior250 strict rank30,
warmup40,known-down veto,max2/ISOweek plusgap variants; eight frozen controls.
85fit logs=17x5. Early3pass; selected BEFORE late:
local_hist_h5_r30_nogap_cap2. Late h3/5/10/20=
2.249813/2.268181/2.305287/2.270683; h5rate1.020491,currency.933831..1.075774,
sym67.713089,fwd115.912842. DeltaAP10 h5CI[.063612,.388060],50date
[.048893,.387239]; vsAP11hazardCI[-.018206,.290788], not proven.

Full ExtraTrees policy was early-pass but not early-selected. Late h1/3/5/10/20=
1.929562/2.385630/2.475228/2.350489/2.418814. h5=649/3260,hit.734977,
base.294479,rate.969691,currency.948773..993597,sym66.581498,fwd134.674142.
H5 liftCI[2.130341,2.839284]; delta vsAP10 [.205703,.636313], vsAP11hazard
[.138520,.545055], vsAP1exact[.051474,.370064], vsAP1cap2[-.010171,.335440].
50date deltaAP10[.188165,.680708]. Best late Brier .211861. Max2/week,
maxgap43days, oneemptyfullmonth. Strongest new next-period challenger, NOT a
freshholdoutselectedrecord; lateperiod repeatedly opened and CI search-unadjusted.

Known70/hazard30 r275 late h3/5/10/20=2.320842/2.382949/2.364561/2.395898,
rate1.177375h5 but early rates.77-.81 so fails early gate. It is a useful
practical diagnostic, not winner. Long causal-rank controller alone moves
known-z to2.318213h5, showing controller contributes strongly.

AP12audit rebuilt source hashes,5755targets,17masks,85logs,compactsubset,
local/class counts, all scores/signals, veto/prefix/weeklycap. PDF
output/pdf/ivan_after_publication_ap12_effective.pdf rendered4pages and checked.

## NEXT bounded AP40-E: causal weekly optimal stopping

1. Freeze AP37 as strict shadow leader and AP38 as accuracy-only diagnostic. Do
   not tune support=.70, shrinkage40, silence10 or runway on opened2024-2026.
2. Register exactly one classical optimal-stopping model before scoring. Train
   only on fully mature past weeks to estimate whether accepting the current
   opportunity dominates preserving one weekly slot for a later day. Use weekday,
   causal expert ranks, known change and current within-week state; no future row.
3. Preserve AP26 core unless the stopping model explicitly predicts wait and a
   causal cadence bank exists. Friday and silence rescue stay hard. Compare to
   frozen AP37 under identical today-effective target and availability.
4. Preserve chronological OOS fits, publication-h20 maturity cap,2day embargo,
   known-down veto, causal state and sequential max2/week. One architecture and
   one policy only; no weekday/threshold/window grid.
5. Acceptance remains minlift>2.4 on h3/5/10/20,min currency rate>=1,zeroempty,
   max2/week with paired uncertainty. Final product validation additionally needs
   actual receipt timestamps and executable bank price,not only official CBR.

## Latest completed AP10-E: strong matched information gain, simple rule selected

Previous pushed AP9 b4d9b5b. First AP10volume source scan found ALLvolume/value
null in173455mainbars(CNY74442+10directarchives) and183frozenCETSchecks; fresh
scopedCETS2026-09-02 response55bars metadata bothundefined/allnull. AP9probe
checked columns only; correct availability inference. InitialAP10OHLCshape
protocol/module PAUSED BEFORE FIT by usertarget clarification; no shape/volume
model evaluated. Preserve unexecuted draft, do not treat it as a negative model.

Actuallyfit AP10-E: 4information sets pastCBR,announcedCBR,pastCBR+market,
announcedCBR+market; HistGB/logit/coarseSurvivalHist,12variants/27policies.
Original3sets9models/23policies saved ap10_effective. Then separately registered
strongmarket control,3models/4policies added ap10_effective_extended, prior
predictions EXACTunchanged. Registered docs effective_information_registered.md
and effective_market_control_registered.md. Main scripts correspondingnames.

Same5755events18:30/20delay, current_index reference, announced=current+1.
Past-only independently rebuilds ALLcurrencies' features from effectiveprefix,
including CNY/USD/EUR; knownchange/gapzero. Strongpastmarket rebaseseachlocal/CNY
priceandvol to effectiveprefix, no algebraicfuture input. Same rawmarketbars.
This is same18:30 information-withholding, NOT actual15:30 prepublication.
Same17quarterorigins2022Q3..2026Q3,train2022+,keepOLDpublicationfullh20<origin−2
cap to matchrows across4sets; effective targets mature evenearlier. 68logrows,
each3fits. No hypersearch. Hist160/.05/15leaves/min40/L2=5/noES; logitC.1.

Target current<=minnext h observations. h1knownafterreceipt, knownnextdown
makesallhfailure. Gatedpoliciesveto exactlynextdown; ungatedsameoutputscontrols.
Samepast63CDF40warmup urgentcap2. Simpleknown-signcd3; zknownchangegated; oldAP3
frozen diagnosticonly. Select2023 jointminlift1.3/rate1..2/symLB>0/cap2/fwd.8
relativeknown-sign; 15pass. Selected known_change_z_urgent_cap2 in bothpackets.
Earlyminlift1.755432,benefitLB2.883796,fwd-ratiomin1.136258, rates1.177..1.327.
Late h1/3/5/10/20=1.935727/2.071153/2.054037/2.077536/2.075095.
h5=927/3260,2024-01-09..2026-08-25,rate1.385059,sym55.313108,fwd87.565409.
h5CI[1.860043,2.263515],50[1.832863,2.305466]. AllpooledliftLB>1.3 AND sym/fwdLB>0
at20/50dates. Fwdh20=86.739589 CI[28.041345,143.282781],50[11.943804,158.971125].
75year/currency/h points min1.650022, not75simultaneousCIclaims. Maxgap21,max2,
noemptyfullmonths,4.3–9.4%emptyweeks. h1notunknownfutureforecast! Receiptsstill
CALENDAR-ASSUMED,late/earlyopenedrepeatedly,nofreshholdout/searchadjustedCI.

SameCBRHist 0.997804->1.975266 withnewfixing deltaCI[.784993,1.211539].
SAME MARKET Hist1.717072->2.049456 deltaCI[.184434,.472143], gate->2.072437
vsoldCI[.202896,.496957]. MarketSurvh5 1.760820->2.041189,gate2.083183;
gated vswithoutCI[.180986,.462866]. Logitmarket1.613942->1.984409,gate2.083477.
Complex2.083477 vsselectedsimple2.054037 CI[-.040035,.098267], no superiority.
Signcd3=1.949810,knownz2.054037 deltaCI[.015262,.195382], improvedsimple.
OldAP1publication change_z_r25 matchedcurrenth5=2.259851 stillstrongerpoint;
differentcontroller/longhistory/cadence, MUST keep ascontrol next, notnewrecord.

All206tests passed, collectcount206 (11new). Finalindependentaudit terminal:
5755currentindices/targets,all68masks/riskcounts,allprioroutputs exact;
27555marketrebasings maxerror1.0214e−13. Tests all-currencyfuturepricecorruption,
futurebars,announcedmetadataignore,actualHist/logit/surv future/immaturelabels,
gates/policyprefix. Audit initially includedunscoredNaNtailinh1assert; fixedmask
only, reran successfully, no forecasts/metricschanges. Allprocessesterminal.

Also ran after_publication_reference_comparison.py: ALL510policyinstances in11
AP1..AP9packs re-scored bothreferences,10200rows,matchedsupportperh/samefired.
sourcehashes preserved, originalsunchanged,no newwinner. AP3old1.630->current
1.031 is targetmismatch, not evidenceagainst information. Alloutputs in
results/research/after_publication/reference_comparison. ThreepagePDF
output/pdf/ivan_after_publication_effective_information.pdf rendered/allchecked.
Goalactive; nohourlyautomation. Current effective main; pubdiagnostic retained.

## NEXT bounded AP11-E: unknown remainder, known buffer, strong old-rule control

1. Evaluate old AP1 knownchange_z rank25(window250) on EXACTsame currenttarget,
   dates and cadence constraints before claiming newrecord. Save unmodifiedold
   signals and causal cap2/cooldown adaptations separately; earlyselection only.
   New2.054 is feasible cap2 anchor, oldmatched2.260 notforgotten. Do not report
   bestlate policy as a selectedwinner; inspectallh/cadence/benefit andslices.
2. Exploit exact structure: announced<current impliesallhzero, h1known. For
   announced>=current learn remaining h−1 unknown observations conditionalon
   normalized known buffer log(announced/current). Never learn alreadyknownh1
   as if newforecastskill. Fix simple knownz and identicalcurrentmarket controls.
3. Bounded genuinelydifferent approaches: conditional survival (excludeknown
   first interval), localcurrency simple logistic/Ridge anchor plus global OOS
   residual trainedmatureonly, barrier probabilities from available pathscenario
   distributions relativeknownmargin (AP7 distributionmethod adapted to current
   reference, notreusezero-barrierpubscores). Need correcth−1 trajectoryindices.
4. Keep train-only statistics/empiricalresidual distributions, h20 maturity
   and2dayembargo; initialcontrolsharedcap, exploreeffectivematurityfreshness
   separately ifworth. Neverlate2026regimeroutefit. Ifscenariosresampled dates,
   dependence handledbydateblocks; zero/knownfuture contributionseparated.
5. Freezeprotocol/testsfirst, allh/pairedCI/cadence/PDF/summary/checkpushownbranch.
   Newround currenteffective primary; originalpub scorecardseparatediagnostic.
   Userexpectsstronger model but do notpromise/exaggerate; aimtoprovesuperiority
   overalreadystrong simpleknownfixing andold2.26policy, notweakCBR-onlybaseline.

## Historical AP9: resolved partial feedback did not improve lift

Previous pushed commit d8373c7 (AP8). AP9 actually fit 12 variants, 34 policies:
quarterly17/monthly51 origins, direct full20/mature5, coarse5-bin full20/partial,
fine20-step full20/partial. Same133 features,18:30/20min delay,5755 events and
old early/later target/support. Calendar-assumed CBR receipts, NOT certified.
RawX for direct, AP4 CSVroundtripX for hazards: quarterly full controls EXACT.
Hist160/.05/15leaves/minleaf40/L2=5, noearlystopping, originalX+intervalonehot.

Observed prefix uses only receipts effective_date-1calday <origin-2days.
Directh5 waits5 even afterknownfailure. Coarse incompletebin omitted for BOTH
failures and survivors, completedbins only. Fine each observedstep until first
strictlycheaper rate or censoring. No unknownzeros/futureexposure inputs. Full20
controls require20known but riskrows stop atfailure. Independent censoring is
not proved by temporal correctness; read Suresh/Severn/Ghosh2022 primarypaper.

11earlypasses,4oldcontrol/copyrows,7new. Selected
cny50_quarter_fine_full20_urgent_cap2: earlyminlift1.349995,benefitLB11.806261,
fwd-ratiomin.999124. h1/3/5/10/20=1.478110/1.511114/1.612490/1.566265/1.571702.
h5=908/3260,2024-01-09..2026-08-25,rate1.356670,sym36.818448,fwd55.765918.
Maxgap18,max2/week,noemptyfullmonth,7.2–10.9%emptyweeks. KZT2026h1=1.226804.
Lift deltaAP3 CI[-.077732,.031199],AP4[-.065825,.027325], NO confirmedliftgain.
SymdeltaAP3+3.424830 CI[.270638,6.545539],50date[.482541,6.148716]; forwarddelta
.858725 CI[-3.382141,5.073589]. Does not establish superiorityoverAP4sym.
AllpooledliftLB>1.3/symLB>0 at20/50blocks; fwdh20=53.942428 CI[-12.991694,
113.060969],50[-21.504837,131.412069]. No freshholdout/searchadjustedintervals.

Rawh5 full20->fresh: quarterdirect1.657377->1.602630,coarse1.615691->1.617500,
fine1.628703->1.631006; monthdirect1.632152->1.605050,coarse1.607110->1.647068,
fine1.643376->1.672724. AllpairedCIs cross0. Quarter->month full20 tooallcross0.
NO computedh5pair lowerCI>0, allablations. Late monthfinepartial1.672724 fails
earlyminlift1.194231/fwd-ratio.787236, notpromoted. Brier improved monthfine
.189396->.181083,quarterfine.191501->.185162, notbetterrankingproof.

Jan2025 training full20=3610,lastinitialDec2; mature5=3685,lastDec23;
observed>=1=3705,lastDec27. 95extra currencydayrows,83observedfailures/12censored
survivors; not95independentdates. Fine risk23671->23933/fail2871->2954;
coarse9125->9275/fail2871->2948. LastnewlabelreceiptDec28<cutoffDec30.
All195tests passed (9new), collectcount195. Independentaudit408fitrecords,
51prefixsnapshots, ALLriskrowhashes/counts/maxreceipts reconstructed. Tests
futureprice/featurecorruption,coarsefull20compatibility,likelihoodproduct,
100k independentcensoring geom.2 syntheticrecovery, immaturefailureinclusion.
Results results/research/after_publication/ap9; 4pagePDF allpagesrendered and
visually checked, verification.json records finalhashes. No live processes.
AP3 main/AP4 alternative retained; goal ACTIVE, no hourlyautomation.

## Paused AP10 starting plan: volume unavailable, target clarified before shape fit

1. Sourceprobe without targets saved inAP9/next_source_probe.json. Frozen
   data/moex_spot_fx_10min_2022_2026.json CNYRUB_TOM74442 bars contain
   open/close/high/low/value/volume/begin/end. Current round6 CNYloader drops
   value/volume; AP2–AP9 features ignorethem. Direct KZT55357 hasboth too.
   These are available unused raw fields, not evidence of predictivegain.
2. Verify primary ISS unit/normalization docs and zero/invalid handling first,
   sourcehashes, price/value/volume relationships with facevalues. Do NOT call
   volume-weightedclose actualVWAP without validation. USD2026 excluded asbefore.
3. Separate fullrawloader preserving olddefaults/results. Freeze compactpacket
   at18:30 with20mindelay, sameCBRreference/support and quarterlymature20.
   Completedbar volume/turnover weights, recentvolume shares, pastsession norms,
   pricepath reliability; distinguish price-only addedpathcontrol fromvolume
   addition. Priornormalizers onlyknownpast/samecutoff; nofuturewhole-daytotals.
   Price*volume is an activity/direction proxy, NOT observed aggressororderflow.
4. Compare simplevolumeweightedbasis/quality vsCNYlast and fixedHist/survival;
   avoid hugeweightgrid. Freeze earlyselector, nolateclock/regimeselection.
   Source/units/futurebarcorruption tests, independentdeadline/featureaudit,
   allh/cadence/sym/fwd/blockCI/PDF/summary/checkpush ownbranch. Goalactive.

## Historical AP8: later simple alternative but no superiority

Previous turn pushed AP7 as244f3b1. This turn fit fixed directHist/survivalHist
at18:10/18:30/18:50/19:30, same133 features and20min feed delay. Generalized
market_features optionalcutoff preserving default.68fits logs (4clocks*17origins,
two model families per log), mature20<quarterorigin-2d since2022; no hypergrid.
CBR source features are frozen publication snapshot, targets start from latest
announced fixing. CALENDAR-ASSUMED actualreceipt, NOT certified18:00 availability.
Earlier18:10 conditional on publication actuallyreceived. Laterdata cannot be
advertised as improvement at18:30. No bank execution savings assertion.

20policies=12 primary (4times*3 CNY/CNYHist50/CNYsurvival50) +6 frozen18:30
snapshots later1850/1930 +2 exactAP3/AP4 controls. Freeze ENTIRE snapshot/score,
inclage, no newdata; signals exact18:30. DirectHist preservesrawmarketfloats,
hazard preservesAP4 CSVroundtrip representation; exact18:30 model maxerrors0
forCNY/directHist/hazardallh. Saved raw+hX for everyclock.

13earlypassrows,7primarypasses (5newtiming variants). Selected
t1930_cny_urgent_cap2: earlyminlift1.392638,benefitLB1.362567,fwd-ratio1.078324.
Lateh1/3/5/10/20=1.506676/1.572046/1.652568/1.633637/1.586246.
h5 905/3260,2024-01-09..2026-08-25,rate1.352188,sym35.114453,fwd59.221593.
181signalspercurrency, identicaldates across5corridors inlateperiod: do NOT
call905independentdecisions. Block dates allcurrenciesjoint. Maxgap14days,
max2/week,noemptyfullmonths,9.4203%emptyweeks. MainAP3/AP4 retained at18:30;
AP8simple1930 is a distinct later-time alternative, notconfirmedimprovement.

h5deltaAP3 CI[-.093693,.150194],AP4[-.087410,.138163],symAP3deltaCI
[-7.422264,11.421188],fwd[-9.019021,17.418494]. AllpooledliftLB>1.3/symLB>0
with20/50dateblocks, buth20fwd58.182656 CI[-3.053233,113.490130],50date
[-11.860483,127.872995]. WeakKGS2025h20=1.172999,KZT2026h1=1.253314.
No newh5candidate lowerdeltaCI>0 vsAP4. Allopened early/later periods and
conditional/notsearchadjustedCIs; nofreshholdout.

Rawtimingh5 CNY/Histmix/survmix:
1810 1.503186/1.552996/1.583324
1830 1.638273/1.630058/1.630232
1850 1.564379/1.651329/1.661292
1930 1.652568/1.586931/1.605308.
1810vs1830simpleCI[-.217673,-.057628],Histmix[-.149432,-.010869],worse.
1930vs1830simpleCI[-.072194,.108710],1850Histmix[-.038982,.077205],no gain.
LateHistBrier1830 .188132->1850 .182669, notsameasliftimprovement.
CNYmeanbarcounts47/49/51/54.003, medianlasttradeage20.0167/20.0167/20.0167/
30.0167min. NewCNYbars to1850/1930on100%lateh5rows,19:30basisdiff99.2331%.
Directlocal1930new44.7546%,available71.5031%,medianage53.983min.

186tests passed9new. Reconstructed4*5755=23020snapshot events,all68trainmasks,
21063rawinstrument/date source-deadline rows, allscore/policy/target/support.
Reranfinalaudit after normalizingCSV None/NaN comparison. Same oldpredictions
exact. Tests featurefuturecorruption, actual+nominalboundary, overnightmissing,
frozen/policyprefix, actualHist/survivalfuture/immaturelabelcorruption.
3pagePDF output/pdf/ivan_after_publication_ap8.pdf allpagesrendered/checked;
verification.json recordschecks. No live model/test/audit remains.

## Completed AP9 starting plan: use already resolved partial outcomes

1. Current requirement fullmature20 for everytrainingevent is conservative but
   delays available short-horizon feedback. Investigate incremental outcome
   availability and right-censoring as a genuinely different information-use
   mechanism. Keep2day embargo and fullmature20 control; do NOT relax causality.
2. Read primary discrete-time survival/censoring sources, then freeze compact
   protocol beforefitting. Directh5 may use mature5. Hazard intervals1/2-3/4-5/
   6-10/11-20 can use only intervals whose outcome is already determined before
   origin-2d. Start from conservative completedintervals; if using already
   observed failures beforeintervalend, explicitly derive their availability
   from firstfailure receipt and at-risk eligibility, NOT ultimate fullpath.
3. Beware outcome-dependent inclusion/immature-negative bias: censored partial
   exposure must be treated correctly, not simply add earlyknownfailures while
   ignoring censoredsurvivors. Prefer per-observation discretehazards or a
   proper censoring likelihood with known exposure lengths. Validate analytically
   and test against fullmaturecontrol on synthetic censored trajectories.
4. Bounded quarterly/monthly refresh comparison isolates feedbackage from more
   modelsearch; fixeddirectHist/survivalHist and simpleCNY control. Main18:30
   and separate19:30 anchor are differentclockproducts. Start18:30, do not
   silently substitute laterinformation or pick clocks bylate2026 errors.
5. Sourceprefix/labelcorruption tests, counts exposure/risk/firstfailure/max
   receiptlogs, trainstatistics only, sameearlyjointcriterion/late support,
   allhbenefit/fwd/cadence, blockuncertainty, preservePDF/results/checkpush.
   Goalactive, hourlyautomationdeleted; continue actual experiments.

## Historical AP7: trajectory distributions did not improve

Previous turn pushed AP6 as71fcd0e. This turn implemented eleven path-distribution
families and70signal policies, fit/replayed all, 177 tests passed (9new), audited
and wrote4page PDF output/pdf/ivan_after_publication_ap7.pdf. All pages rendered
and visually checked. Active goal NOT complete; no live model/audit remains.

Same5755 publication events, AP2-D20 clock and old support. Paths are next20
cumulative log ratios / known own vol20 (floor1bp), multiplied by query known
vol. Only fullmature20<monthorigin-2d library, startsJan2022; unlike AP6 no base
OOS score required, library is legitimately larger. Queries July2022 onward.
Core7 (asinh CNY/fallback,knownchange,ret5z/ret20z,logvol,CNYlate,quality), wide24
source-only from AP6, no expert/disagreement scores. KNN train-only scaling.

Families global64/global128/wide128/local32/shrink nlocal/(nlocal+200), recent730
global128, unconditionaluniform, temporal-split32trees, conditionalRidge empirical
residuals, Gaussianresiduals, localRidge empirical. Structure first60%distinct
past library dates with own mature cutoff before boundary-2d; remaining40%
estimate leaf/residual distribution. No refit on calibration, no in-sample noise.
Gaussian256 fixedantithetic draws,.5cov+.5diag, no futures. Utilities computed
per scenario then averaged, NOT from meanpath. Numerical logguard20 neverused.

70policies=11*6 +4exactcontrols(AP3/AP4/AP5equal/AP6selected). p5,pmean,forward25,
symmetric25,CNY50,AP4anchor75. Existing urgent controller unchanged. Sixearly
joint passes,3new. Selected path_ridge_empirical_forward25_urgent_cap2:
earlyminlift1.349646,benefitLB.051056,forwardratio1.093564,rate1.177..1.284.
Later h1/3/5/10/20=1.393736/1.416979/1.496597/1.480243/1.401400.
h5=900/3260,2024-01-09..2026-08-25,rate1.344717,sym17.470702,fwd44.150352.
Gap28,max2/week,noemptyfullmonths. h5vsAP4 CI[-.276536,-.000990],vsAP3
[-.283005,.004014]; symdeltaAP3 -15.922917 CI[-24.393278,-8.479045].
h20liftCI[1.196815,1.602749],forward41.677513 CI[-27.070968,104.180656];
50dateh20lift[1.194735,1.611091],fwd[-41.662934,125.128986]. No all-h stability.
Weak cells2025AMD/KGS/TJS/UZS,2026KZT; TJS2025 allfiveh<1.3. AP3 main/AP4
alternative retained; candidate NOT promoted. No new h5CI superior toAP4.

Rawh5: global64 1.481954/global128 1.514931/wide1.337097/local1.493180/
shrink1.505898/recent1.534230/forest1.534410/Ridgeemp1.516829/Gauss1.506027/
localRidge1.495745. Widevscore7 CI[-.307840,-.059895] worse; recent/local/shrink/
forestvsglobal128 allCIs cross0. Unconditional.775899,rate.657417,gap238.
AP4+forest25 diagnostic1.661308,rate1.356670,sym34.599480,fwd54.414551;
vsAP4CI[-.032555,.087022], earlybenefitLB-7.407589, fails. No late selection.
Ridgeemp/Gauss Brierlate .188321/.209505, localRidge.187932, global128.189007.

Audit all paths/targets/knownpast, maxutilityreconstructionerror7.1e-12bps;
306fitlogs,51origins,56870scenariologs,10340 exact global128/local32 neighbors
andweights/predictions rebuilt,51 monthly forest examples rebuilt. Cached NPZ
arrays to avoid repeated decompression; reran final audit successfully. No
clipping; allweights valid, allsurvival monotone; futurepath/featurecorruption
tests passed, fourcontrols exact. Results results/research/after_publication/ap7.
Jan2025 library3610,lastprediction2024-12-02,lastmaturity2024-12-28;
split2023-10-05,structure2055(estmaturitymax2023-10-02),estimation1445.
All late+early dates repeatedly inspected, nofreshholdout/searchadjustedCIs.

## Completed AP8 starting plan: marginal value of evening market data

1. Several new model families failed. Re-focus on available information, whose
   earlier AP2 ablation was the largest gain. User explicitly wants AFTER the
   fixing publication; comparison of later decision clocks is in scope but
   MUST NOT be advertised as improvement at18:30 or executable bank savings.
2. Predeclare a small timing packet18:10/18:30/18:50/19:30, same announced CBR
   reference and20min marketdelay. Rebuild raw bars point-in-time at each clock;
   no futureclose, samebarend+delay<cutoff AND begin+10+delay<=cutoff. CBR actual
   receipt still unverified, so earliestclock conditional on actual receipt.
3. Read sourceprobe saved inAP7: CNY2026-09-02 has55bars09:50..18:59:59;
   KZT25bars10:00..18:59:59; CNY archive maxbeginhour2022=23,2023..25=18,2026=19.
   These are archive observations, not session rules. load_market_frames in
   after_publication_ap2_features.py has full-day bars, session_state accepts
   cutoff but market_features hardcodes CUTOFF18:30. Generalize optionalcutoff
   while preserving default and exact18:30 output; never mutate priorresults.
4. Compare fixed simpleCNY, fixed CNY+Hist50 and fixed CNY+survivalHist50 with
   causal quarterly training/mature20 and same urgentpolicy, maybe small fixed
   closing-range/late-momentum features. Separate addedinformation from larger
   search. Same date/target support, earlyjointselector frozen, later retrospective.
   Add stale/frozen18:30 feed control to identify gains actually due tonewbars.
5. Analyze information-age/coverage and source availability, not just lift.
   More time before prediction is a real product tradeoff, report waitingminutes
   explicitly, no phantom postclose updates. Testsforcutoff futurebars,default
   compatibility; audit/PDF/summary/tests/checkpushownbranch. Goal staysactive.

## Historical AP6: new feasible candidate but no superiority

Previous goal turn pushed AP5 c66f4b0. This turn actually trained sixteen OOS
meta models and evaluated 51 policies. 168 tests passed, nine new. Registered
protocol after_publication_ap6_registered.md was frozen before fitting. Same
5755 events, AP2-D20 clock and publication reference, exact early/later support.

Six raw h5 experts from AP3/AP4; 24 available context covariates +2 current
expert-disagreement features. Source-only controls omit both disagreement
features. Monthly origins July2022..Sept2026 (51), strict mature20<origin-2d,
train-only standardization. 969 fit-log rows. All masks, scalers and pairs were
rebuilt by audit. Optimizers succeeded; positive coefficients bounded at zero.

Models: positive logistic expanding/rolling730, ordinary global expert logit,
joint-context logit expanding/730, separate local logit, local/global shrink
n/(n+250), small HistGB expert/joint/joint730, multi-h mean regression, pairwise
same-currency positive/negative within60d up to8 nearest negatives, global
correction of AP5 issued equal probabilities, global correction of own local
meta OOS probabilities, source-only logit/Hist. 16*3 policies +3 old controls.
Same urgent controller; extra variants CNY50 and AP4 anchor75/metaCDF25.

8 early passes (6 new). Selected ap4_w25_stack_local_residual_urgent_cap2:
early minlift1.343717, benefitLB2.543740, forwardratio.942752. Later
h1/3/5/10/20=1.480635/1.507989/1.613434/1.543365/1.541725.
h5=900/3260,2024-01-09..2026-08-25,rate1.344717,sym33.594912,fwd54.666086.
Gap16, max2/week, noemptyfullmonths. Versus AP4 h5deltaCI[-.065564,.023436];
versus AP3[-.069507,.033499]. No new h5 policy CI superior to AP4.
AP3 stays main, AP4 alternative, not a record or replacement. All pooled lift
LB>1.3 and symLB>0 at20/50date blocks, but KZT2026 ALL h<1.3 (h5=1.257437).
Fwdh20=49.560763 CI[-16.853770,107.543957],50dates[-24.685692,128.291476].
At50date h3 lift delta vsAP3[-.085454,-.000267] barely negative. Conditional,
notsearch-adjusted, early2023 also repeatedly used, nofreshholdout.

Useful negatives: local meta1.645280 + globalOOS correction1.582839,
pairedCI[-.149828,.008082]. EqualAP5 1.664972 +correction1.560373, deltaCI
[-.213226,-.006224] negative. Global expertlogit1.642884 ->context1.575806;
Hist experts1.631272 ->context1.584234, no significantgain. Sourceonly Hist
1.528725 ->joint1.584234 CI[-.006845,.115482]. Pairwise1.289286,rate1.072785,
gap69,4emptyfullmonths maxpercurrency,sym45.915180/fwd25.248185.

Positive730 diagnostic1.656174,rate1.355176,Brier.178859 vsAP5equal.186302;
early minlift1.278336, minbenefitLB-20.209331, fails. Actualmeanpred.294717
vsrate.294479 later. Positive2023 fit coefficient means CNY.644027,Hist.283870,
localSurv.193918,globalSurv.002283,Extra/SurvHist0. Coefficients are not percent
weights or causal importance. 2023 descriptive sign regimes documented only,
notnew filters: CNYscore-/ownchange- n194 hit.113402, +/+ n406 hit.605911.

Jan2025 fit uses3025 rows, lastprediction2024-12-02,lastmaturity2024-12-28,
lastlocalorigin2024-12-01,8664pairs. Future corruption of labels/features and
residuals tested; current local training predictions never substituted for
their issued OOS values. All results results/research/after_publication/ap6.
PDF output/pdf/ivan_after_publication_ap6.pdf (4 pages), all pages rendered and
visually checked. verification.json records checks. No background job remains.

## Completed AP7 starting plan

1. Stop adding weight grids to the same six experts. Investigate a genuinely
   different forecast representation: complete normalized future paths of
   comparable past days, from which all-h survival and expected benefit are
   derived jointly. No retraining or choosing formulas on already opened2026.
2. Predeclare a bounded packet: simple nearest-neighbor analogs with available
   CNY basis, known own change, volatility and causal trend context; local
   same-currency versus global normalized-path transfer and shrinkage. Consider
   honest distributional forests or a small conditional Gaussian/path baseline
   as a different comparator, not only another point classifier.
3. Train/reference library includes ONLY fully mature20 paths before origin-2d.
   Scale each archived path by that event's available vol; unscale using current
   known vol. Distance standardization and weights use training only. Future
   realized calendar dates/gaps cannot be query features; only schedule facts
   actually known at decision are allowed. No test-neighbor outcomes.
4. Compare first-cheaper probabilities, expected future benefit and symmetric
   benefit reconstructed with knownpast, using unchanged signal policy and
   AP3/AP4 controls. Keep fixed earlyjoint/futureguard, later retrospective.
   Test library cutoff/future corruption, label/path consistency, same support.
5. Preserve all experiments, audit/summary/PDF/tests, checked push ownbranch.
   Historicalreceipt/prospective gaps remain limitations, not a reason to stop
   experiments or redo finished bank/legality audits. Goal stays active.

## Historical AP5: no new early-feasible policy

Previous goal turn AP4 pushedcb69c08. This turn actually fit monthly OOS
calibrators and replayed delayed weights:66policies,6experts,5calibrationmodes,
12adaptive weights, frozenweights, exactAP3/AP4controls. All159tests passed,
9newAP5. No newly selected solution: only oldAP3/AP4 pass earlyjointcriteria;
AP4 selected again. Do not describe its1.630232 as an AP5 improvement.

Expert order fixed in metadata: CNY,directHist,directExtra,survivalHist,
survivalGlobalLogit,survivalLocalLogit. Features arcsinhCNY orclippedlogit(basep).
Monthly2022July..2026Sept (51origins*5modes=255logs), using only priorbase OOS
withmatureh20<origin-2days. ExpandingpositivePlatt,rolling365,local365shrink
n/(n+250),expandingisotonic,frozenJan2023. Complete source support exactly same
asAP4. Labels reconstructed and source hashes checked. Monotone h projection.

Rolling calibration earlyBrier global/localLogit .327073/.343570->.243656/.247054;
later .202411/.231051->.198274/.208238. But lateh5lift1.396538/1.379569->
1.345907/1.296764, maxgap81/126days. Frozenmonotone CNY/global/localLogit
has ZERO changed later signals vsraw: calibration cannot invent ranking skill.

Delayedweights use the rolling365 ISSUED probabilities, five-h Brier. Eligibility
dates<T AND mature20<T-2days; equivalent feedbackavailablematureday+3. Exponential
decay of revealedrows byfeedbackage, half63/252caldays, eta2/10/30,10%uniformfloor.
Global orlocalmeanloss shrunkwith effective_mass/(mass+50). Same-day currencies
see same global state.67,210 rowweight logs (13*5170); arrays andlosses saved.
Frozenweights atJan2023, equal before, still uses evolving rollingcalibrated
experts afterward. It freezesweights, notthebaseforecasts/calibration.

Example2025-01-09:3025revealedrows, lastmaturity2024-12-28,lastprediction2024-12-02.
No nextfutureanswers used. Meanlate global252eta10weights:
.185947/.178149/.168658/.176140/.149318/.141789. Mostly modest equalweightadjustment.

Late h5 equalrolling1.664972,rate1.296905,sym33.767338,fwd58.427628;
global252eta10 1.673687,rate1.299893,sym33.804224,fwd58.039835;
frozenweights1.686736,rate1.298399,sym33.612597,fwd58.502363.
Adaptivevsequal deltaCI[-.004133,.027258],vsfrozen[-.032291,.001785]. All12
adaptivevsequal h5CIs cross0. Brier .186302(equal)/.185696(adaptive)/.185701(frozen).
Maxgap28days. Isotonicequal h5=1.703826,rate1.343223,notearlyselected.
Early equalrolling minlift1.295984,minsymLB-21.761257,minforwardratio.784625.
Newmodels fail someearlyjointgate. Do NOT promote via laterpretty numbers.

Files after_publication_ap5*.py; results/research/after_publication/ap5 include
all66scores/signals,5probabilitytensors,13weightarrays,67kweightlog,calibrationlog,
sourcehashes,probabilityaccuracy,pairedcalibration/rawcontrols,20/50blocks,
yearcurrency andclustering. AP4 choose_early only generalized outputdirectory;
default oldbehavior preserved, no priorresult overwritten. PDF
output/pdf/ivan_after_publication_ap5.pdf, source after_publication_ap5_report.md.
Verification.json has finalPDF/source/time/test evidence. Goal remains active.

## Completed AP6 starting plan

1. Preserve AP3/AP4 controls and all negatives. AP5 scalar Brier weighting did
   not beat equal/frozen; avoid anotherlarge eta/half-lifegrid. Its meanloss
   objective differs from selecting favorable points at1-2/week.
2. Predeclare a diverse OOS meta-learning packet: simple multi-expert logistic
   orconstrainedlinear stacker, globalHistGB stacker andshrunken/local variants,
   trained only on genuinely past expert predictions plusAVAILABLE regime
   features. Candidate regime features may include currency,knownCBRchange,
   CNYbasis/volatility, disagreement/marketavailability; neverfutureerrors.
3. Predict case h5/multihorizon orfirstpassage labels directly; residual/correction
   orconditional weighting is useful only if errors can be forecast fromknown
   context. Do not train meta on in-sample basepredictions. Matureallh20 and2day
   embargo. Compare simpler equal/handrank andsourceonlymodels.
4. Consider direct utility/ranking objective only with causal normalization and
   nofutureweeklytop-k. Keepjointcriteria/futureguard, early2023 selection fixed
   beforelater evaluation. Already-viewed2024-2026 stayretrospective evenwhen
   used as priortrainingrows bywalkforward learning.
5. Review early2023 calibration versus2022 marketregime withoutchoosing a
   UZS/KZT2026-specific rule. Testsforfuturecorruption,monthly/quarter origins,
   sourcehashes. Saveall/PDF. Historicalreceipts andprospective validation remain
   limitations, not an excuse tostop actualexperiments orredo bankaudits.

## Historical AP4: alternative not universal improvement

Previous goal turn pushed AP3 as203ae4a. This turn actually trained three
first-passage/hazard families (globalHistGB/globalLogit/localLogit), restricted
waiting-time regression, local hazard OOS residual corrections25/50%, ensemble
weights and soft knownpast/fullforecast/futureonly utility blends.46policies.
Frozen protocol after_publication_ap4_registered.md. Same5755events/133features,
AP2-D20 timing/source assumptions, exactAP3control and comparisondates reproduced.

Model construction: five at-risk intervals ending1/3/5/10/20. Failure is STRICTLY
cheaper future price; ties survive. Only intervals before/including first failure
entertrain. Prediction design repeats ORIGINAL-T features plusintervalonehot,
never realizedfutureprices. Curves are cumulative products, monotone byh.
Restrictedwait=min(firstcheaper,21), missing if incomplete20; regresseslog1p(wait).
17quarterorigins since2022Q3; matureh20 beforeorigin-2days, pastOOSresiduals only.

2023 selection adds future-only guard>=80% ofAP3 for eachh, toexistingjointgates.
Four earlyfeasible candidates; selected cny50_hazard_hist_h5_urgent_cap2.
Early minimumlift1.337504 vsAP3 1.309943. Selection saved beforelater scorecard.
Selected_simple field is compatibility alias for incumbent, NOT a claim that
AP3 is a non-ML model. No selected outcome used as a future signal input.

Later selected adjustedh1/3/5/10/20:
1.477640/1.520349/1.630232/1.566219/1.558629.
h5:903/3260,2024-01-09..2026-08-25,hit.480620,frequency1.349200.
Max2/week, maxgap16days,noemptyfullmonths. AP3 remains fixed maincontrol.
Lift h5 delta vsAP3 +.000175, paired20dateCI[-.046172,.050498].
Sym h5 +36.612139bp vs33.393618, deltaCI[.789986,5.840468].
Forward h5 +56.353744bp vs54.907193, deltaCI[-2.182189,5.073625].
50dateblocks: liftdeltaCI[-.038979,.040891], symdeltaCI[1.298464,5.130105],
forwarddeltaCI[-1.190069,3.848944]. h1 lift lower:50date deltaCI[-.053731,-.000051].
Do not promote as universal improvement or newrecord. All pooled lift CIs>1.3
and symCIs>0 at20/50blocks, but conditional/notmultiple-search-adjusted.
Yearh5=1.570134/1.826806/1.525618;currange1.568050..1.713972.
KZT2026h1/h3=1.236017/1.289747; forwardh20+54.065831CI[-11.547222,111.871453]
(50dates[-18.117823,129.039754]). Still no positivefutureh20 proof.

Negatives: hazardHist h5=1.615691 vsdirectHist1.657377; globalLogit1.396538,
localLogit1.379569; restrictedwait1.540239, pairedvsdirectHistCI[-.220897,-.022077].
Local meanhazard1.375065; residual25/50%1.353380/1.397037, no significantgain.
DirectHist Brierh5 .188132, hazardHist.188277, ExtraTrees.180566 onlater.
Global/localLogit EARLY Brier .327073/.343570, meanpred .112468/.156943 versus
early actual .389427: strong calibration/regime weakness, notproofuniquecause.

Softutilities25% knownpastall/fullforecastall/futuremean produce h5
1.541089/1.562938/1.607456. Knownpast50% h5=1.310456,sym49.018540 but
forward28.430112 (AP3forward54.907193). Symutility can be gamed by knownpast
geometry without leakage; forecast half vsknownpast at25% h5deltaCI[-.034639,.085815]
is NOT evidence of betterfutureprediction. Otherstaticensembleweights no bigboost.

Code after_publication_ap4*.py,8newtests, full150tests passed, allresults under
results/research/after_publication/ap4 including all46signals,3survivalcurves,
restrictedlabels,17trainlogs,sourcehashes,early/laterselection,20/50blockdiagnostics.
PDF output/pdf/ivan_after_publication_ap4.pdf, source after_publication_ap4_report.md.
See verification.json for final full-suite/PDF checks. No OnlineHedge fit inAP4.

## Completed AP5 starting plan (AP6 next above)

1. Read current AP4 outputs before acting. Keep AP3 and AP4 fixed controls;
   do not tune specifically to opened KZT/UZS2026. All later periods are repeated
   retrospective, nofreshholdout or historicalreceipt certification.
2. Check whether mature-label probability calibration (global/shrunkenlocal,
   rollingtail versusexpanding, simplelogistic/isotonic asappropriate) improves
   first-passage/direct probabilities and usefulpoints. Retain rawpast-rank
   controls: monotone calibration can improveBrier without improvingselection.
3. Predeclare a bounded delayed expert weighting experiment. Use equal/fixed
   weights ascontrols; update only from genuinely OOS predictions whose FULLh20
   outcome matured plus2day embargo BEFORE currentdecision. Allcurrencies ofa
   date must see the same prior information. Never update on future/currentquarter
   in-sample predictions or selectweeklytop-k. Save everyweight/cutoff/loss log.
4. Optimize joint TЗ lift/benefit/cadence, retain future-only guard. Diagnose
   regime differences and weaklocal calibration; no unconditional claim that
   more flexible regimes will improve. Diverse target families now exist.
5. Freeze selection beforelater scorecards; matchedsource/dates/baselines;
   20/50dateblocks, yearcurrency diagnostics, tests/PDF/preserve negatives.
   No need to repeatbank/TЗ audit before model work. Target remainsactive.

## Historical AP3 joint tradeoff (still main control)

Previous goal turn completed AP2-D20 and pushed0fc3d1a. This turn actually fit
AP3: six normalized local/global/residual scores, five future-mean models,
raw/rescaled controls, three causal controllers, three rank hybrids, six utility
gates.57 policies. Same5755events,133features,AP2-D20 source hashes/time/target.
Protocol after_publication_ap3_registered.md was frozen before packet evaluation.

The2023 mature-label selector chose cny_hist50_urgent_cap2, the only candidate
passing the early joint gates. No later winner replacement. Selected simple is
cny_cbr_w25_short_cap2. Latest-published reference,18:30MSK,20min market delay;
still CALENDAR-ASSUMED receipt, notcertified timestamps orfreshholdout.

Selected adjusted h1/3/5/10/20 =1.503262/1.547591/1.630058/1.534857/1.546588.
Later95%CI lift = [1.418,1.593]/[1.420,1.695]/[1.477,1.799]/[1.393,1.696]/[1.396,1.712].
Symmetric benefitbp =19.6775/30.0769/33.3936/37.6495/46.8714;
95%CI =[15.81,24.07]/[23.38,38.06]/[25.25,42.48]/[24.86,52.20]/[20.73,74.64].
All pooled lowerbounds meet1.3/0, but NOT multiplicity-adjusted certification.
h5 dates2024-01-09..2026-08-25,908/3260,hit.482379,frequency1.356670.
Currency h5 rates1.337..1.375; max2/week, maxgap15days,noemptyfullmonths.
9-12% of weeks are empty: notguaranteed1-2 EVERYweek.

Mechanism: CNY and AP2marketHistGB each mapped to strictly prior63 midrankCDF,
equal50/50blend, mixture ranked again against its prior63scores.40warmup.
Sequential urgency threshold max(.45,.80-.04*calendar_days_since_last_sent),
minimum2days gap,max2/ISOweek. No futureweek top-k. Initial age7. Neutral
midrank may pass relaxed threshold; no unconditional quota, no qualityguarantee.

VersusAP2mix1.6536 h5lift deltaCI[-.223,.193]: no superiority. Symmetric h5gain
+27.265bp pairedCI[14.973,39.117], maxgap34->15. Forward-only h5 declines
71.204->54.907bp, paired deltaCI[-31.490,-.635]; h20forward51.695CI[-13.977,109.438].
Do NOT call this universal predictive improvement. Symmetric utility improved
partly by choosing different past-window geometries; future-only matters too.

Year h5lift1.5454/1.9143/1.4882; pooledcurrency range1.5613..1.7217.
Narrow UZS2026 has h3/h10/h20 =1.2884/1.2279/1.2137,45-49signals. Cross-slices
selected_year_currency.csv are descriptive, not next-round selection holdouts.

Negative results: rawlocalRidgeh5=1.4731; normalized1.3829; normalizedlocal+
globalOOSresidual50%=1.2810. Gaps203/231/339days. Rescaling predictions alone
also fails tosolve gaps. Full normalizedresidual1.2359meanlift,15emptymonths.
Normalize-with-scale-as-feature paper evidence saved after_publication_ap3_literature.md.
Utility gates predict futuremean only, reconstruct knownpast+predictedfuture.
ExtraTrees gate_all h5=1.8329,sym69.8748bp butrate.64845,gap94,6emptymonths;
CNYgate_all1.7448,rate.69776. Not selected, too sparse. Softutility worth testing.

142 tests passed,7new AP3,17quarterorigins/maturity/OOSlogs verified.
Results results/research/after_publication/ap3 preserve all predictions/signals,
selection,early300 andlater1000paired20datebootstrap, benefitdiffs/cross-slices.
PDF output/pdf/ivan_after_publication_ap3.pdf; manuscript after_publication_ap3_report.md.
AP2audit generalized output/keys/pairs only; old numerical mechanics unchanged.

## Completed AP4 starting plan (AP5 next above)

1. Retain AP3 selected policy as fixed control. Predeclare diverse new candidates
   and early-only selection before running later scores. Never silently promote
   AP2ExtraTrees or hardutilitygate merely for higher opened-period lift.
2. Test CNY/HistGB/ExtraTree rank-blend weights and SOFT predicted utility rather
   than hardpositive gates. Compare knownpast-only utility versus fullforecast
   to identify whether gains come from actual future prediction or knowngeometry.
   Keep sequentialmax2controller; allh symbenefit andfuture-only diagnostics.
3. Consider genuinely delayed expert weighting/calibration, updates only when
   h20 resolves before currenttime plus embargo. Record each weight snapshot.
   No OnlineHedge updates on future labels or retrospective weekly top-k.
4. Try a distinct first-passage/survival or price-change-distribution target,
   not just tune tree depth. Protect against double-counting known firststep.
5. Assess year×currency/quarter and bootstrapblock sensitivity. UZS2026 and
   forwardh20 are weak descriptive slices, notfresh validation sets. Preserve
   all negatives. Actualreceipt/prospective checks remain future limitations;
   do not stall all actual model experiments waiting for new months.

Target active; hourlyautomation deleted. Continue untiluserstops. Scopedchecked
pushes authorized onlyivan-experiments. No bankactions ormain/forcepush.

## Historical AP2-D20 result, preserved below

Previous goal turn was PROGRESS: AP1/QA committed and pushed as1aeab71.
This goal turn completed55 policies with two full model/feature/policy replays.
5,755 events since2022,133 features. CNY + direct local bars relative to NEW
announced CBR. Main decision18:30, CBR assumed receipt18:00, market feed delay20min.
All dates are still calendar-assumed, delay not a measured historical SLA.

Source audit verified CNY FACEVALUE1 and183 candles on three dates against CETS.
USD2026 records are not necessarily corrupt: https://www.moex.com/s3933 documents
new ruble-settled non-deliverable contracts. Excluded from this bounded packet.
https://www.moex.com/a8531 confirms free ISS latency, hence a FULL20min delayed
replay, not merely dropping final signals. Zero-delay results preserved separately.

Frozen2023 selection again picked simple cny_cbr_w25_r25:75% CNY late basis/vol
plus25% known own announced change/vol, prior250 top25%,40warmup. Effective
future first step is NOT included: target starts at latest announced rate.
Adjusted h1/3/5/10/20 =1.5541 /1.6460 /1.6536 /1.5594 /1.5415.
h5 dates2024-01-09..2026-08-25;838/3260 signals/rows,hit.496420,rate1.252081.
h5 CI[1.439580,1.886183], paired gain versus AP1 change_z1.060 CI[.420874,.806010].
Year h5=1.649502/1.685281/1.637576;currency range1.539420..1.718508.
Symmetric h5+6.128208bp CI[-6.201625,19.077620], NOT statistically positive.
Maxgap34days,max5/week,noemptyfullmonths. h20 lift lower CI1.249631 below1.3.

Diagnostics, NOT replacement of early-selected winner based on later scores:
- cny_last_r25 h5=1.740952,rate1.225187,sym+23.629015 CI[10.998784,37.346592];
  h20 benefit CI still crosses0;maxgap41days,oneemptyfullmonth.
- market_hist_r25 h5=1.8053,rate1.0773,all-h sym CIspositive;maxgap73days,
  up to3emptyfullmonths and2025rate.936. Same HistGB CBR-only1.2281;
  h5 paired improvement CI[.2961,.9001], frequencies differ under same rule.
- market_extra_r25 h5=1.8396,rate1.1580;year h5=1.8151/1.9166/1.8106;
  h20 symCI[-.0311,78.5397] just crosses0;maxgap41days.
- direct quality-weighted fractions25/50/100% give1.7034/1.6191/1.5277 vsCNY1.7410.
- local Ridge h5floor1.4731; true earlier-quarter OOS global residual weights
  25/50/100% give1.4365/1.4146/1.3025. No improvement. Maxgap203days forlocal,
  502days forfullresidual. Raw regression scale/model transitions need diagnosis,
  not yet proven to be the cause. Do not silently convert these to classifiers.

Quarterly train2022Q3 onward,expanding since2022,all h20 mature beforeorigin-2days.
Residual targets use genuine past OOS local forecasts. Selection2023 only,
resolved before2024. Later2024-2026 repeatedly opened, NOT a newholdout.
All135tests passed (8newAP2). Code/protocols after_publication_ap2*. Results
results/research/after_publication/ap2 and ap2_delay20 retain predictions,
selections,traininglogs,coverage,pairedboot,benefit andclustering. Four-page PDF
output/pdf/ivan_after_publication_ap2.pdf. Main PDF emphasizes delayed scenario.

## Completed AP3 starting plan (results and NEXT AP4 above)

1. Read AP2-D20 saved diagnostic_breakdown.csv and clustering.csv. Investigate
   long no-signal runs: how raw regression scale, volatility and quarterly
   calibration shifts change the past-rank threshold. Use TRAIN/early statistics
   for choosing variants; opened later slices may diagnose but are not holdouts.
2. Test volatility-standardized local minimum target and OOS residuals, versus
   current raw-bp baseline. Normalize with available volatility at the row's T,
   never future volatility. Test a genuinely held-out rolling calibration tail
   for each refit, or causal recalibration from matured OOS labels.
3. Model joint favourable/benefit objectives and sequential communication budget:
   one score cannot be promoted on lift alone. Compare CNY anchor, market HistGB,
   ExtraTrees with predeclared causal benefit/regime gates and1-2/week policy.
   Do not post-hoc choose bestdays of a week or use immature h20 labels.
4. Preserve AP1/AP2 and both feed-delay scenarios. No need for another bank audit
   before actual model work. Historical receipt certification and prospective
   verification remain limitations; do not call current goal achieved.

## Read first

- after_publication_tz_decision.md: original authenticated case page rechecked;
  latest published reference is the conservative case-facing interpretation.
  Public CBR evaluation is permitted with an assumption about execution prices;
  a bank quote archive is NOT a prerequisite for the case's experiments.
- after_publication_protocol.md: target conventions, gates, selection rules.
- after_publication_ap1_registered.md: frozen AP1 packet; do not rewrite after
  seeing results. Data/protocol hashes saved in AP1 metadata.
- publication_applicability/report-source.md: Q&A/CBR/bank evidence. This older
  report predates the direct-page recheck; its bank-quote requirements apply to
  real monetary savings, not permission to evaluate the public CBR series.
- results/research/publication_applicability/calendar_alignment_summary.json:
  old2.459 replay and147/732 calendar-inconsistent signals. Preserve old artifacts.

## Completed AP0/AP1

AP0 added research/after_publication_clock.py with explicit received_at versus
effective_date, timezone-aware snapshots, current versus latest/next announced
fields, and mandatory evidence labels. Inferred18:00 timestamps are explicitly
calendar_assumed, never observed. New tests include the real TJS Saturday
failure pattern, UTC equivalence, just-before release, future corruption,
asynchronous/revised records and missing information. All9 targeted tests passed.
Verification saved in
results/research/after_publication/ap0_verification.json.

AP1 reconstructs 19,326 own-currency announcement events with 111 available-prefix
features and independent peer cutoffs. All times are CALENDAR-ASSUMED: effective
date minus one calendar day at assumed 18:00 MSK. No claim of verified historical
timestamps. Signals occur only on new own-currency announcements; no all-day or
weekend fallback evaluation yet. h is next observations; baseline uses identical
event dates. There is no live bank quote panel.

56 policies evaluated, 28 per target convention. Six model families plus simple
rules and stale controls. Annual training from 2016 with 7-year or 3-year windows;
all training labels matured to h20 plus two-calendar-day embargo. Selection uses
2017-2020 and 2022-2023 with all outcomes resolved before 2024. Selection JSON
written before later evaluation. 2024-2026 is already-viewed RETROSPECTIVE data,
not a new holdout. Frozen selector's effective h1 ceiling is a known limitation.

All 127 repository tests passed, including 9 clock and 6 AP1 tests. Model scores,
signals, outcomes, selections, training logs, year/currency breakdowns, paired
1000 date-block bootstrap and benefit/clustering audits are preserved in
results/research/after_publication/ap1/. Code: after_publication_panel.py,
after_publication_ap1.py and after_publication_ap1_audit.py. Run as modules with
PYTHONPATH=. .venv/bin/python -m research.after_publication_ap1, then the audit.

## Results to preserve, not silently reinterpret

Early-selected simple policy change_z_r25: known change / previous 20-return
volatility, above the 75th percentile of strictly previous 250 scores.

| Convention | Adjusted lift h1/3/5/10/20 | h5 cadence |
|---|---|---|
| Publication reference, all future steps unknown | 1.139 / 1.109 / 1.060 / 0.976 / 0.939 | 1.30 |
| Effective reference, first future step known | 1.917 / 2.274 / 2.265 / 2.227 / 2.220 | 1.30 |

Publication h5: 869 signals / 3260 rows, decisions 2024-01-09 through 2026-08-25;
hit .318757, symmetric benefit -36.942 bp. Stable >=1.3 / positive benefit NOT
achieved. Early selector picks this simple rule; HistGB r25's later h5 1.152510
cannot be promoted merely because it is retrospectively higher.

Effective h5: 869 / 3265, 2024-01-09 through 2026-08-26; hit .678941,
adjusted 2.264617 (pooled 2.309108), CI [1.992628, 2.571390]. By year
2.123 / 2.721 / 2.110; currency range 2.212-2.311. All-h symmetric-benefit CIs
positive; h5 +53.945 bp [40.620,70.471], future-only +121.692 bp. Max gap
43 calendar days, max 4 signals/week and one empty full month in some currencies.

Effective sign_cd3: h5 1.953315, cadence 1.43, max gap18 days, max2/week,
no empty full month. Effective early overall selector quantile7y_r25: h5
2.139092, cadence 1.52; delta versus simple2.265 CI [-.301761,.052598], no
ML superiority. z rule versus sign h5 paired delta CI [.142503,.477952], but
different allowable cadence means this is NOT an isolated matched-rate feature
ablation. Stale20 z control h5 1.075876.

Old after-publication 2.459 remains UNCONFIRMED: 147/732 old signals use a later
announcement under the date rule, mostly Saturdays. Preserve old artifacts;
the rebuilt panel does not rehabilitate that number or certify 18:00 timing.

New four-page PDF: output/pdf/ivan_after_publication_ap1.pdf; source manuscript
after_publication_ap1_report.md. README and EXPERIMENTS_SUMMARY now distinguish
these scenarios and keep the older 15:30 result in a historical section.

## Historical AP2 starting checks (completed; next work is AP3 above)

Initial archive inspection (no AP2 target scores evaluated): CNY has 74,442
candles, including 7,102 beginning during 17:00 and 7,091 during 18:00, so late
intraday slicing is feasible. USD archive has 46,547 rows and claims observations
through 2026-09-03; verify source history/market availability before relying on
it, rather than presuming either validity or unavailability. Existing CNY/USD
archive lacks per-instrument FACEVALUE metadata; verify normalization as well.

1. Inspect whether saved CNY/USD 10-minute archive and round7 direct-pair data
   cover post-fixing hours. Data: data/moex_spot_fx_10min_2022_2026.json and
   data/moex_direct_pairs/*.json. Respect SHA manifests, nominal units, source
   timestamps and full nominal candle completion, not merely last trade time.
2. Freeze a compact after-publication experiment before seeing scores: market
   movement after the fixing window relative to the NEW announced CBR reference,
   with fixed cutoff justified in protocol (18:00 is not guaranteed publication
   time). Start with simple common/local basis and mature-label calibrations,
   then local anchor + global residual or alternative targets. Respect missing
   sessions and currency availability; no outcome-based drops.
3. Main emphasis is publication-reference lift AND positive symmetric benefit.
   Use chronological early selection and label opened later years retrospective;
   keep same-information controls and matched target/date denominators.
4. Separately improve conditional effective-reference cadence: causal cooldown,
   sequential weekly budget/threshold adaptations from past data only. No
   post-hoc top days within a week. Show Pareto tradeoff with sign_cd3.

Record failures too; preserve frozen packets. Update PDF at meaningful completed
stages. Checked commit/push only to ivan-experiments under standing authorization.
No main/force-push, bank transfers, client notifications or bank contact.
