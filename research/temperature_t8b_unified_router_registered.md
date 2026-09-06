# Temperature T8B frozen protocol — unified any-time widget router

Registered 2026-09-06 after the failed T8 timestamp assertion and before the T8B
artifact was built. T8 produced no artifact or metric.

T8B introduces no fitted model and performs no outcome-based selection. It
packages the audited phase experts into one timestamped snapshot table for
2024--2026.

The fixed sequence is T4 history-only at 00:00; T5 market snapshots from 10:30
through 15:30; T3 post-window updates at 16:30 and 17:30; the frozen AP50/AP51
decision at its actual saved calendar-assumed time 18:30; and genuinely new T7B
delayed-market updates at 19:00 and 20:00. Production replaces the assumed CBR
receipt relation with observed event timestamps. Between events the latest
snapshot is held and becomes aging/stale; weekends fabricate no values.

Every snapshot carries probabilities and expected future-only CBR basis points
for h=1/3/5/10/20. After 18:30, h=1 is a known comparison. The sparse `push_now`
flag appears only on the AP37 18:30 decision row and is false on later refreshes.

Require unique currency/valid_from rows, source_at <= valid_from, future-snapshot
corruption invariance and successful queries in premarket, intraday,
after-decision, overnight and weekend phases. No bank execution claim is made.
