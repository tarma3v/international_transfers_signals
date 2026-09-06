# Temperature T16 frozen protocol - horizon/output-aware evening router

Registered 2026-09-06 after the audited T15 selection and before T16 outputs.

Start from the complete T14 snapshot stream. Do not change any T14 probability,
temperature or sparse push decision. At the existing 20:00 after-publication
snapshot, replace expected future-only benefit only for T15 screen-adopted,
physically usable horizon h=3. Add 21:00, 22:00 and 23:00 snapshots only when a
new completed dual CNYRUBF/USDRUBF prefix is physically available. At these new
snapshots carry every probability from the T7B 20:00 control; route expected
benefit through T15 only for the frozen screen-adopted h=3/h=5 combinations.
h=10/h=20 retain T7B, and h=1 remains exactly known from the current and newly
announced CBR rates.

The row-level timestamp records the newest source used by any output. Because a
single snapshot can carry an old probability but a newly updated benefit, add
separate per-horizon probability provenance and per-horizon benefit provenance:
source timestamp, source kind and availability evidence. The public lookup must
return both, assert that neither is later than `as_of`, and compute freshness
independently. This prevents a carried 20:00 probability from being described as
if it consumed the 22:59 futures candle.

The 20:00 base row may differ from T14 only in adopted expected-benefit fields
and the new provenance columns. All other T14 rows must remain exact. New rows
must have unique `(currency, valid_from)`, source timestamps not later than
their query clock, `push_now=False`, and exact fallback when the selected T15
benefit is unavailable. Future snapshot corruption must not change earlier
queries; weekend/holiday lookup must retain the latest state and become stale.
AP37 push counts must remain identical.

Historical receipt availability remains calendar-assumed. No field represents
an executable bank quote, fee-adjusted saving or guaranteed customer outcome.
