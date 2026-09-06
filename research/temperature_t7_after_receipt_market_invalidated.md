# Temperature T7 invalidation

T7 is retained as a negative research record and must not feed the product
router. During T8 packaging, the event table exposed that AP49/AP50 is an 18:30
decision snapshot with a 20-minute market-feed delay. T7 had treated it as an
18:00 base and then added the 18:00--18:30 market move. That interval was not
strictly new relative to the underlying AP feature family, and the T7 18:30 row
collided with the true AP decision timestamp.

The causality audits inside T7 remain valid in the narrow sense that they did not
read future rows. The experiment is nevertheless semantically invalid because
the information sets overlap and the base timestamp is wrong. None of its small
reported improvements may be used. T7B restarts from the real saved 18:30 base,
preserves the 20-minute availability delay and evaluates only later clocks.
