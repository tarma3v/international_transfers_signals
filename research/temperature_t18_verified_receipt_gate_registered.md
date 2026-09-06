# Temperature T18 frozen protocol — verified CBR receipt gate

Registered 2026-09-06 before T18 query examples, counts or audit outputs were
computed.

## Problem

The historical after-publication replay uses 18:30 Moscow as an explicit
calendar assumption because certified historical delivery timestamps are not
available. That convention is acceptable for an open retrospective comparison,
but it must never make a production-style query claim that the new CBR record
was already received. The default case CLI currently has no receipt gate and can
therefore activate a `cbr_receipt` or receipt-dependent market row at the
assumed clock.

## Frozen runtime rule

For a query on Moscow calendar day `D`, rows whose model requires the new CBR
record are admissible only after the caller supplies a timezone-aware,
same-day `verified_receipt_at <= as_of` event. Receipt-dependent rows are:

- `cbr_receipt`;
- `post_receipt_market`;
- `post_receipt_perpetual`;
- any row whose phase begins with `after_new_cbr`.

Without such an event, remove those rows for `D` before latest-valid lookup.
The router must hold the latest admissible pre-receipt state, including its true
source age and freshness. It must not synthesize a receipt from wall-clock time.

With a verified event, shift every same-day receipt-dependent row's
`valid_from` to `max(original_valid_from, verified_receipt_at)`. Its row-level,
horizon probability and expected-benefit source timestamps become at least the
verified receipt time, because the new fixing is a required input even if the
market candle itself completed earlier. Preserve all numeric outputs and push
decisions. Add explicit `receipt_verified` and `receipt_at` provenance.

The event must be rejected if it lacks a timezone, belongs to another Moscow
calendar date, or is later than `as_of`.

## Frozen CLI behaviour

`run_case_output.py` uses the verified-receipt gate by default. The caller may
pass `--verified-receipt-at`. The old calendar-assumed replay remains available
only through an explicit `--historical-calendar-assumption` research flag; the
two modes cannot be combined.

## Frozen audit gates

1. At 18:45 with no receipt event, all five corridors stay on a pre-receipt
   source.
2. At 18:45 with a verified 18:42 receipt, all five corridors may use the new
   CBR row and cite 18:42, not the assumed 18:30.
3. If receipt occurs after a planned 19:00 update, that update cannot become
   valid before the receipt and its source time cannot precede the receipt.
4. A future, timezone-free or wrong-day event is rejected.
5. Numeric snapshot payloads and AP37 decisions are not refit or selected from
   open-period outcomes.
6. Removing/corrupting rows after a query does not change the query result.
7. Historical receipts remain uncertified and executable bank prices remain
   unvalidated.

T18 is an information-availability safety repair, not a new holdout result or a
claim of improved predictive accuracy.
