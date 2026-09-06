# Prediction contract for the presentation interface

The main-branch intermediate deck does not contain a literal application
mockup. Slide 12 defines the intended product as three separate components:
an on-screen indicator, a level notification and a recipient-selection rule.
The continuous-temperature goal maps to that structure directly.

## On-screen indicator

For the selected corridor and horizon, show `temperature_0_100` and one cautious
label: “момент выглядит выгоднее обычного”, “скорее выгодный момент”,
“нейтральный момент” or “вероятно, лучше подождать”. Also show `last_source_at`,
`freshness`, `phase`, `confidence` and the information source. The colour reflects
the calibrated forward probability, not merely the rate's historical range.

The detail view exposes P(current effective CBR is no higher than the next h
publications) for h=1/3/5/10/20 and expected future-only CBR basis points. It
must not convert those basis points into customer rubles until executable bank
quotes, fees and limits are integrated.

## Notification

`push_now` is a separate sparse policy optimised for stable lift and 1–2 signals
per currency per week. A green or high-temperature widget state does not
automatically generate a push. When a user arrives later, the screen displays
the latest admissible snapshot rather than the old push payload.

## Any-day behaviour

The router selects by information state. Overnight it uses the weaker CBR-only
expert. During the market session it moves through completed-candle experts.
Between 15:30 and the actual CBR receipt it uses the frozen market anchor plus a
validated post-window correction. After receipt it can use the newly announced
rate. Weekends and missing feeds retain the latest score with explicit aging or
stale status.

Every snapshot has `valid_from` and optional `valid_until`. This prevents an
after-receipt forecast anchored to yesterday's effective rate from being shown
the next morning as if it described today's current rate. `source_at` must never
exceed the requested `as_of`.
