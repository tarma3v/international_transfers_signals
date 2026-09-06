# AP48-E frozen protocol — same-week expert substitution after a veto

Registered 2026-09-06 before AP48 signals or scorecards were computed. AP46's
joint-survival model improves several accuracy/benefit diagnostics but removes
43 late AP37 decisions and misses cadence. AP47 showed that globally restoring
model-rejected core signals at a 1.20 pace buffer collapses back to the AP45
policy. AP48 tests one different architecture: replace a veto with an
independent expert's opportunity instead of undoing the veto.

Start from the frozen causal AP46 decision stream. Within each currency and ISO
week, remember whether AP46 has vetoed an AP26 core opportunity. If the frozen
AP23 fallback stream subsequently fires while that veto is pending, emit one
substitution signal and clear the pending flag. A fallback on the same row as
the veto is allowed because both inputs are available at that decision time.
Pending state resets at the next ISO week. Preserve AP46 signals in chronological
order and apply a new sequential cap of two signals per currency-week; a prior
substitution can therefore consume capacity that a later AP46 signal would have
used. This is a genuinely online trade-off and never looks ahead to later rows.

There are no new fitted models, outcome-derived thresholds, cadence floors, or
parameter grids. AP37 is the registered fallback if this sole policy does not
pass the unchanged 2023 early joint gate. Report all unknown horizons, cadence,
symmetric/future-only benefit, decision changes, paired 20/50-date bootstrap,
and a future-prefix corruption audit. `h=1` remains validity-only; receipt-time
and executable-bank-price limitations are unchanged.
