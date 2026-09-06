# Temperature T8 frozen protocol — unified any-time widget router

Registered 2026-09-06 before the unified replay artifact was built.

T8 introduces no new fitted model and performs no outcome-based selection. It
packages the audited phase experts into one timestamped snapshot table for
2024--2026 and exercises the production-style `score_snapshot_as_of` contract.

The fixed order is: T4 history-only snapshot at 00:00; T5 market snapshots at
10:30, 11:30, 12:30, 13:30, 14:30, 15:00, 15:20 and 15:30; T3 post-window
updates at 16:30 and 17:30; AP50/AP51 at the calendar-assumed CBR receipt; T7
market updates at 18:30 and 20:00. Production replaces the assumed receipt with
the actual observed event. Between events the latest snapshot is held and its
freshness worsens; weekends never fabricate an interpolation.

Every snapshot carries probabilities for h=1/3/5/10/20 and expected future-only
CBR basis points. Before receipt these come from T4/T5/T3 plus T6. After receipt,
h=1 is a known CBR comparison, while h=3/5/10/20 use AP50/AP51 or T7. The sparse
`push_now` flag is populated only on the AP37 after-receipt decision snapshot and
is false on later widget refreshes, preventing repeat notification.

Audit source hashes, timestamp monotonicity, uniqueness of currency/event rows,
all source times <= valid_from, and a future-snapshot corruption test. Query
examples must cover premarket, intraday, after receipt, overnight and weekend.
This artifact remains CBR/MOEX research, not executable bank pricing.
