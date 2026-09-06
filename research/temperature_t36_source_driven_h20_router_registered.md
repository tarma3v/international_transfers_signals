# T36 registration: source-driven h20 availability router

Registered after completing T35 and before running T36 or inspecting any T36
metric.

## Failure being addressed

T35 changed h20 by four wall-clock states before 10:30. The 10:15 state failed
because most 10:15 rows already had an admissible market prefix. Conversely,
weekends and missing-market days may remain on `cbr_history` after 10:30. A wall
clock is therefore the wrong boundary; the actual latest source is observable
at query time and is the causal routing state.

## Single frozen route

The only candidate is `source_driven_h20_shadow`:

1. Reconstruct the T19/T17 latest-valid snapshot for every T22 query.
2. If `snapshot_source_kind == cbr_history`, use the latest backward-joined T34
   `cold_identity_qstack_w125_r100` publication probability.
3. If the scenario contains a receipt-dependent source (`cbr_receipt`,
   `post_receipt_market`, or `post_receipt_perpetual`) use the already frozen
   T22 `rank_correction_selected` probability. In production this branch still
   requires an explicit verified same-day receipt event; the historical
   `calendar_assumed_replay` is an upper-bound replay only.
4. For every market, bridge, perpetual, stale-market, or other source, preserve
   the existing T19/T17 h20 probability exactly.
5. Do not change h1/h3/h5/h10, expected future-only bps, freshness, confidence,
   or `push_now`.

There is no clock list, threshold grid, alpha, blend, currency rule, or fit in
T36. The route is determined only by the source that was actually available.

## Evaluation and gates

The evaluation period is the already opened 2025--2026 retrospective. Report:

- overall and currency/year Brier, log-loss, ECE, AUC, and AP for every
  scenario/clock;
- route-source conditional metrics, deduplicated by
  scenario/date/currency/source to avoid counting an unchanged held score once
  per clock;
- reliability bins;
- paired 20/50-date bootstrap for every clock and for pooled scenarios, keeping
  all currencies and clocks of a date together.

The retrospective route passes only if:

- each pooled scenario improves Brier, log-loss, and AUC, with Brier upper CI
  below zero and AUC lower CI above zero for both block sizes;
- the deduplicated `t34_cbr_history` and `t22_after_receipt_replay` components
  each improve Brier/log-loss/AUC at point estimate and do not worsen ECE by
  more than 0.01;
- no scenario/clock has Brier delta above +0.001, log-loss delta above +0.003,
  ECE delta above +0.01, or AUC delta below -0.005;
- unchanged rows equal identity bit-for-bit.

`production_promoted` remains fixed to false because the component behavior and
the open interval have already been inspected. A pass would define a better
frozen prospective route, not a fresh independent winner.

## Required audit

- immutable source hashes and exact source-map reconstruction;
- unique complete query keys and source/query timestamp causality;
- T34 only on `cbr_history` with backward publication join;
- T22 only on receipt-dependent replay rows and never in no-receipt rows;
- every other row bitwise unchanged;
- target and future-source corruption invariance;
- complete metrics, reliability, source-conditional, and bootstrap grids;
- no runtime or push mutation.
