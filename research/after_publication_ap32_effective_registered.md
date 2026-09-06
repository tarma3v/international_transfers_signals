# AP32-E frozen protocol: decision-level AP26 core with AP23 deficit fallback

Registered 2026-09-06 before AP32 signals or scorecards. Score interpolation in
AP30 diluted the AP26 specialist. AP32 instead combines two already-causal
decision streams and tests exactly one meta-policy.

At each currency/date, take the frozen AP26 y20-shrink200 decision first. If it
does not fire, allow the frozen AP23 mature-competence decision only when the
meta-policy's own trailing365 selected rate is below1. Enforce a new sequential
max2/ISO-week cap on the combined stream. The rate uses only earlier combined
decisions; neither source signal is recomputed or changed. Source signals already
include their maturity-safe scores, known-down veto, month rescue and caps.

There is no score access, threshold grid, lookahead to later days in the week,
post-hoc replacement or use of outcomes. Reason1 means AP26 core; reason2 means
AP23 deficit fallback. AP21, AP23, AP26, AP27 strict, AP29, AP30 and AP31 are
frozen controls.

The sole fresh candidate is selected on early2023 with unchanged joint gates;
opened2024-2026 remains a retrospective diagnostic. Audit exact source signals,
trailing meta-rate, reason and cap state, future-source prefix invariance and
paired20/50-date uncertainty. Acceptance is min h3/h5/h10/h20 lift>2.4, min
currency rate>=1, zero empty complete months and max2/week. H1 is known after
receipt and excluded; publication timestamps remain calendar-assumed and bank
execution is unvalidated.
