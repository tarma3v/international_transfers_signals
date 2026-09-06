# T37 registration: fixed 50% shrink for the history h20 route

Registered after the complete T36 result was inspected and before any T37
prediction or metric was computed.

## Failure being addressed

T36's source-driven route strongly improved pooled Brier, log-loss, and AUC in
both receipt scenarios, and both routed components passed their conditional
gates. It nevertheless failed the frozen route decision because the raw T34
history probability worsened ECE by more than 0.01 at six daytime clocks in
each scenario. The ranking signal is useful, but its probability displacement
is too confident in those mixed-source states.

## Single frozen candidate

The only candidate is `source_driven_h20_shrink50_shadow`:

1. Reconstruct the exact T36 source-driven route.
2. On `snapshot_source_kind == cbr_history`, replace the raw T34 probability
   with the equal-weight log-odds blend of the identity h20 probability and the
   T34 probability:

   `sigmoid(0.5 * logit(identity_h20) + 0.5 * logit(t34_probability))`.

3. Preserve T22 `rank_correction_selected` exactly on verified-receipt replay
   rows.
4. Preserve identity h20 bit-for-bit on every other source.
5. Do not change any other horizon, expected future-only bps, freshness,
   confidence, receipt policy, runtime route, or push decision.

The coefficient 0.5 is a single conventional equal-weight shrinkage constant,
not the winner of an alpha grid. No alternative alpha, currency rule, clock
rule, or threshold will be inspected in T37.

## Evaluation and frozen gates

Use the already opened 2025--2026 retrospective and the same complete
scenario/clock grid, deduplication, reliability bins, and paired 20/50-date
bootstrap as T36. T37 passes only if all T36 gates pass unchanged:

- each pooled scenario improves Brier, log-loss, and AUC, with the Brier upper
  CI below zero and AUC lower CI above zero for both bootstrap block sizes;
- both routed components improve Brier/log-loss/AUC at point estimate and do
  not worsen ECE by more than 0.01;
- no scenario/clock has Brier delta above +0.001, log-loss delta above +0.003,
  ECE delta above +0.01, or AUC delta below -0.005;
- unchanged rows equal identity bit-for-bit.

`production_promoted` remains false regardless of outcome because T37 was
motivated by an already inspected open-period calibration failure. A pass can
only nominate a frozen prospective rule for a fresh independent holdout.

## Required audit

- immutable source hashes and exact reconstruction;
- source/query timestamp causality and backward-only T34 join;
- the fixed log-odds formula only on `cbr_history`;
- T22 only on receipt-dependent replay and never on no-receipt rows;
- all other rows bitwise unchanged;
- target and future-T34 corruption invariance;
- complete metrics, reliability, component, and bootstrap grids;
- no runtime or push mutation.
