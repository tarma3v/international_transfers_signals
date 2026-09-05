# Round 7: direct currency pairs and a continuous usefulness indicator

Frozen before inspecting direct-pair target performance, 2026-09-05.

## Objective and boundaries

Improve push quality across h=1/3/5/10/20, retaining roughly 1-2 alerts per
currency/week. Incumbent: packet ED availability_route, frozen rolling 22%/20
policy. Latest known CBR rate for the effective date is allowed; tomorrow's
unpublished rate is not. Official target is fav_h, i.e. current normalized CBR
rate <= minimum of the next h observations. Benefits: both symmetric and
future-only. Never replace official target with an exchange quote silently.

## Data

CETS ten-minute TOM and TOD candles for KZT, AMD, KGS, TJS and UZS versus RUB,
2022-01-01 through 2026-09-03. Save raw public ISS responses, metadata, quote
units and hashes. Normalize by FACEVALUE; check historical scale on train.
10:00-15:30 local window for like-for-like comparisons; both nominal candle
completion and recorded end must be <=/before decision cutoff. Missing market
data never remove a target row. No inferred volume when ISS reports null.
TOD and TOM stay separate until availability is checked; TOM preferred.

## Candidate families, registered grid

1. Direct signed market/CBR basis, routed to incumbent when missing.
2. Direct last-price basis and geometric session mean; TOM-first TOD fallback.
3. Quality attenuation from completed candle count and last observation age.
   Hard gate: >=6 candles and <=60 minutes old. Soft quality:
   min(count/24,1) * exp(-age_minutes/120). These are density proxies, not volume.
4. Common/local causal rank mixtures with local weights 0,.10,.25,.50,.75,1.
   Compare global weight and individual currency weights. Select on 2024 only,
   with h20 outcomes resolved before 2025-01-01. Penalize unstable/sparse signals
   via cadence and all-horizon performance; tie-break to less local weight.
5. Quarterly logistic/HistGB/ExtraTrees on direct and common features, target
   history and currency identity. Train only when h5 outcomes have resolved.
   A separate simple per-currency logistic model checks local heterogeneity.
   Pooled learned correction of the incumbent is a separate candidate.

2025-2026 is already explored historical research data, NOT a pristine holdout.
Freeze choices on the 2024 screen before viewing later results. Report selected
model even if worse. Delayed20 local features, future candle corruption, and
paired date-block bootstrap versus incumbent are required. Show calendar and
currency breakdowns and do not optimize on those after viewing them.

## Widget

Design an as-of API for any time with quote timestamp, quality and valid-until.
Separate raw ranking 0-100 from a probability calibrated on resolved past
outcomes, and from actual transfer savings (requires executable provider price
and fees). A 15:30 model has not been validated at every intraday time. Repeated
intraday points must be split by date, not randomly. Missing/stale quotes must
remain visibly stale; no updating the time of a frozen score. Keep push policy
fixed in this round; do not select the best point of a future session.

## Second packet, registered after 7A and before running 7B

7A selected per-currency weights on 2024; those did not beat the incumbent on
the already explored later period. Test a different objective: predict the
minimum future h5 log-price change as CNY basis plus a learned residual. Use
global Ridge, HistGB mean, HistGB .25 quantile, ExtraTrees and local Ridge;
weights .10/.25/.50/1, with unchanged live availability routing. Select only
on purged 2024 again, record that the later data have already been seen, run a
delayed-local control and a future-outcome corruption check. No iterative
hyperparameter optimization against 2025-2026 is authorized by this protocol.

## Continuation

Maintain round7_next_steps.md and retain every experiment, including failures.
Only claim improvement after matched comparisons. Prospectively shadow the
frozen baseline and candidate together on dates after research cutoff.
