# AP1 registered packet - 2026-09-06, before evaluating rebuilt targets

Previous goal turn made progress: removed the unwanted heartbeat, created the
active target, wrote an explicit availability contract and passed9 unit tests.
AP1 now advances to actual comparative experiments.

## Fixed panel / assumptions

- Use normalized data/cbr_rates_2010_2026.json. Do not reuse shifted old features.
- One decision per new own-currency announcement, with inferred announcement
  date = effective_date-1 calendar day, assumed18:00 Moscow receipt. Label all
  results calendar-assumed, not verified historical18:00. Asynchronous actual
  receipts remain a prospective/historical-data gap, not a reason to insert
  values before publication.
- Current effective rate is the latest effective_date<=decision_date. Features
  contain only own/peer/reference prefixes whose inferred receipts<=decision.
  Calendar features describe decision date, not the next effective date.
- Two separate studies: effective-reference fav_h starts from current effective
  price; publication-reference fav_h starts from latest announced price. Same
  h=1/3/5/10/20 scorecard; separate baseline and no cross-definition winner.
- At non-announcement times this packet makes no new decision; deployment
  fallback is not evaluated here. Random baseline is matched announcement-event
  dates, not all calendar dates. Report that assumption prominently.

## Frozen search

Simple rules: nonnegative announced change with cooldown0/3/4 calendar days;
announced-change and change/20-step-volatility ranks, prior250 scores only,
top25/35/45%, no cooldown. Gate nonnegative change for effective-reference
target (a negative known first step proves fav_h=0), but do not apply this
logical gate to publication-reference ML where it is not logically necessary.

Classical models, annual OOS scores from2016 onward:
global standardized LogisticRegression C=.1; HistGB classifier 160iterations,
15leaves, minleaf40, l2=5; ExtraTrees 200trees, maxdepth8, minleaf30,
maxfeatures=.8; ExtraTrees short3y; per-currency standardized logit C=.1;
HistGB quantile .25 regressor of future h5 minimum log-price change, 160iters,
15leaves, minleaf40. Other models use7y rolling training. Random seed20260906;
tree workers2. All models predict one common score evaluated across allh.
Classifier training targeth5, quantile regression target min log-changeh5.
For effective-reference classifiers, train on nonnegative-known-change rows
and route known failures below all eligible scores. Regressor uses all rows.
No hyperparameter tuning on opened2024-2026.

Each learned score uses prior250-score top25/35/45% policies with40-observation
warm-up, chronological per currency. Model transitions keep causal prior OOS
scores, not retrospective re-ranking of the whole year. Scores are not labelled
calibrated probabilities in product language.

## Selection and checks

Annual train cutoff: origin minus2-calendar-day embargo; maximum h20 outcome
receipt must precede cutoff, even for h5 training. Latest source used by each
training row also precedes origin. Selection on2017-2020 and2022-2023 only;
all selection labels mature strictly before2024-01-01. Evaluate candidates on
the same date support. Per convention, select by minimum adjusted lift across
five horizons minus a2x cadence violation penalty, then mean lift, then simpler
candidate in a deterministic tie. Cadence violation=max(1-min_currency_rate,0)
+max(max_currency_rate-2,0). Preserve chosen simple and chosen overall separately.

Write selection.json before evaluating2024-2026. Later scores remain already
explored retrospective. Report all candidates for transparency, but do not use
their later ranking to replace the early-selected result. Report year/currency
breakdowns, complete future/symmetric benefits, headline matched uncertainty.
Controls: future-price corruption cannot change earlier feature rows; scores
strictly after cutoff cannot change prior policy decisions; stale20 known-change
rank; paired date-block bootstrap versus same-information sign/cooldown3 and
selected simple. Read uncertainty/negative controls before promotion.
