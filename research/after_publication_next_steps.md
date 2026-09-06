# Active research checkpoint: already published next CBR fixing

Updated 2026-09-06, AP8 complete. The user made after-publication research the
primary indefinite task. The hourly heartbeat `automation` was DELETED at the
user's explicit request. An ACTIVE TARGET drives continuous work in the same
thread and ivan-experiments. Do not restart round7 as the main task, wait for a
schedule, stop at another audit, or mark the goal complete after this checkpoint.

## Latest completed AP8: PROGRESS, later simple alternative but no superiority

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

## NEXT bounded AP9: use already resolved partial outcomes, not unknown futures

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
