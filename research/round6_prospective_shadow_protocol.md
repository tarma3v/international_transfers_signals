# Prospective shadow protocol after round 6

Frozen on 2026-09-05. The historical data cutoff used during model research is
2026-09-03. Every signal date strictly later than that cutoff is prospective.

## Purpose

Compare two already fixed past-only policies on genuinely new dates without
changing features, model settings, blend weights or alert thresholds:

1. Primary `logit50_extra50`: 50% causal rank of the 19-feature global CNY
   logistic regression plus 50% causal rank of global CNY ExtraTrees.
2. Challenger `primary75_local_consensus25`: 75% causal rank of the primary
   plus 25% causal rank of the fixed 50/50 local-currency logit/ExtraTrees
   consensus.

Both use target rate 22%, a per-currency rolling window of the preceding 20
scores and no cooldown or quota. A decision is made before the current score is
inserted into threshold history.

## Information boundary

- A signal dated `t` may use only MOEX records with `TRADEDATE < t`.
- The next CBR rate, a same-day MOEX close and any later value are forbidden.
- A quarterly refit may use a target row only when its fifth future CBR
  publication date is strictly earlier than the refit timestamp.
- Both model families keep the hard training reset at 2022-02-24.
- A missing or more-than-seven-day-old CNY market record remains missing; no
  backward fill from a later date is allowed.

## Immutable settings

- Horizon: `fav_h5`, five CBR publications, not calendar days.
- Global logit: the fixed L2 pipeline and 19 fields recorded in packet AH.
- Global ExtraTrees: the fixed packet-AG architecture and fields.
- Local experts: minimum 140 resolved rows per currency; the same fixed logit
  and ExtraTrees architectures; five independent target currencies.
- All score combinations use causal per-currency percentile ranks.
- No IMOEX, RGBI, RUSFAR, gold, CNY/CBR basis or derived microstructure fields.
- No after-publication course value and no timestamp-dependent switching into
  the separate known-next-rate product.

The code/data hashes at freeze time are recorded in
`results/research/round6/prospective_freeze/manifest.json`. If any hashed input
changes, the new predictions belong to a new model version and must not be
pooled silently with this frozen run.

## Per-signal log

For every currency/date retain: model version, score of every component, causal
rank of every component, rolling threshold, primary/challenger decisions,
latest MOEX source date and age, refit quarter, latest resolved training-label
date, and the data/code manifest hash. The target and future-only benefit are
joined only after all five later CBR publications resolve.

## Reporting schedule and decision rule

- After 13 complete weeks: descriptive cadence and data-quality report only.
- After 26 complete weeks: first descriptive lift/benefit interval; no model
  replacement solely from this checkpoint.
- Primary replacement is eligible only after at least 52 complete weeks and
  250 resolved challenger signals, with annual rate in [1, 2], every currency
  lift above 1.30, paired four-week block-bootstrap lift-difference interval
  strictly above zero, and non-worse mean future benefit.
- All scheduled checkpoints must be reported, including failures and missing
  market days. No stopping at a favourable intermediate date.

Historical 2025--2026 results may be shown as prior evidence but must never be
pooled into the prospective confidence interval.
