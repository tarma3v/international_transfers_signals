# Temperature T17 frozen protocol — observed spot availability repair

Registered 2026-09-06 before T17 counts or metrics were computed.

## Problem

T12/T16 persists every planned 10:30--20:00 spot checkpoint for every CBR
effective-date row. The underlying score correctly becomes a missing-market
fallback when no same-day candle exists, but the router still labels that row
as a fresh `moex_prefix` and sets `source_at` equal to the planned clock. This
is not acceptable for an any-time product: on a weekend, exchange holiday or
feed gap, an absent candle must not masquerade as a newly observed market move.

T17 is a methodological availability repair, not a candidate selected for a
better retrospective metric.

## Frozen repair

For every T16 row whose `source_kind` is `moex_prefix`,
`post_window_market` or `post_receipt_market`, reconstruct physical CNYRUB_TOM
availability from the frozen 10-minute payload. A candle is admissible only if:

1. its `begin` belongs to the same Moscow calendar date as the snapshot;
2. its `begin` is no earlier than the 10:00 spot-session anchor;
3. its recorded `end` is strictly earlier than `valid_from`;
4. its values are already contained in the payload whose hash is frozen by the
   existing manifest.

If no admissible candle exists, delete the planned snapshot. Latest-valid
lookup must then hold the previous admissible state and compute its true age.
If a candle exists, preserve every probability, expected-bps value and push
decision exactly, but replace row-level and applicable horizon-specific spot
provenance with the actual last candle `end`. Never fabricate a cutoff-time
source timestamp.

The already availability-routed 10:00 `moex_early_prefix`, the 09:00
perpetual route, CBR/history rows, CBR receipt rows and 21:00--23:00 perpetual
rows are not candidates for deletion in this repair.

## Frozen reporting and gates

Report planned/observed/dropped counts by clock, year and weekday; verify that
dropped rows have zero admissible candles and kept rows have at least one.
All values on kept rows must be exactly equal to T16 except for provenance
fields. AP37 push counts must be exactly unchanged. Queries must remain
available for all five currencies on every sampled calendar day; a weekend or
feed-gap query must return a held source rather than a synthetic market update.

For diagnostics only, join T16 probabilities to effective-date targets and
compare old synthetic rows with the held fallback using Brier, log loss, ECE
and AUC. These open 2024--2026 diagnostics cannot veto the repair and cannot be
called a new holdout result.

Independently remove/corrupt all spot candles after a fixed boundary and
require every earlier repaired snapshot and query to remain exact. Historical
CBR receipt timestamps and executable bank prices remain uncertified.
