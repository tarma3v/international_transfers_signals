# AP8: marginal information from later completed evening candles

2026-09-06, frozen before fitting. AP7 was progress (pushed244f3b1), but eleven
new distributions did not beat AP3/AP4. This packet changes information time,
not tree depth, the target convention, the signal budget or winner criteria.

## Four distinct decision products

18:10,18:30,18:50,19:30 MSK. Same own announcement events, latest announced
CBR reference, next1/3/5/10/20 observations unknown, same outcome/date support.
Every clock is CONDITIONAL on already receiving the fixing. Historical receipt
remains effective_date minus1 calendar day at assumed18:00, NOT verified.
CBR features are the frozen publication snapshot; only the market snapshot
changes. No extra CBR releases are assumed between these clocks.

Market information delay20minutes (15minutes public-feed assumption plus5minute
buffer, not historical SLA). A bar requires end+20<decision AND nominal
begin+10+20<=decision, begin>=10:00 that day. No overnight forwardfill. Missing
sessions use existing known-own-change fallback for CNY, missing flags for ML.
Keep the original10:00 start and all133 AP2 features for clean comparability;
do not add closing-range features in this packet. CNY and direct-pair source
archives and nominal normalization are unchanged and hash-checked.

18:10 is20minutes earlier than the control,18:50 is20later,19:30 is60later.
Later accuracy is NOT accuracy available at18:30; waiting may consume practical
bank execution opportunity. No bank quote data or bank savings claim.

## Fixed models and policies

At each clock: simple CNY last basis/knownvol; CNY50/HistGB50 prior-rank mix;
CNY50/first-cheaper survival HistGB50 prior-rank mix. Existing AP2 classifier
and AP4 hazard parameters, quarterly origins fromJuly2022, expanding since2022,
fullh20 mature strictly before origin minus2 calendar days. No early stopping.
Keep interval-risk design and all five-h survival curves. Separate fits at each
clock, never use a later-clock training snapshot to forecast the earlier clock.
Prior63 CDF with40warmup and unchanged urgent_cap2 sequential controller.

Four clocks*three policies=12, plus frozen18:30-information controls for18:50
and19:30 (three each), plus exact named AP3/AP4 controls=20 total. Frozen means
the ENTIRE18:30 feature snapshot, including information age/quality and issued
scores, is reused; the decision is delayed but no new data is assimilated.
These controls must produce IDENTICAL scores/signals to18:30. They cannot be
used at18:10. Duplicate controls do not add independent evidence.

Rebuild18:30 sources and scores as a compatibility check against saved AP2/AP4.
AP2 direct classifier originally used full-precision market values; AP4 hazard
read a CSV snapshot. Preserve these input representations across all clocks,
and record any numerical compatibility discrepancies before interpreting them.

## Selection and evidence

Same AP4 early2023 selector: allh pointlift>=1.3, per-currency rates1..2,
allh symmetric benefit lower CI>0, max2/week, future-only>=.8*AP3 eachh.
Write selection before opened-later scorecard.2023 and2024-2026 are repeatedly
studied retrospective periods, no fresh test or multiple-search adjusted CI.
Timing comparisons are paired on identical dates/currencies/target support,
with each policy's realized cadence disclosed. No choosing a later winner
merely from2024-2026 numbers. Report time-information Pareto tradeoff even
when the early selector retains18:30.

Save all market snapshots, features, scores, signals, targets, source hashes,
quarterly masks/counts/max maturity and information-age/coverage deltas.
Tests: optional cutoff backward compatibility, unavailable-bar corruption,
exact availability boundary, missing/weekend state, frozen snapshot equality,
training maturity/future corruption, policy prefix invariance.
Audit rebuilds raw snapshots and all training masks; check same CBR refs,
source deadlines and exact controls. Paired20/50-date uncertainty, allh,
year/currency slices, benefit and signal gaps; separate simple vs ML timing
changes. Preserve negative findings, PDF/summary/checkpoint, checked ownbranch
push; indefinite goal remains active after this packet.
