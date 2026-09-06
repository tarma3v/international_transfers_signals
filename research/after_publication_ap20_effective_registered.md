# AP20-E frozen protocol: causal geometry of AP18 and CatBoost utility

Registered 2026-09-06 before computing any AP20 policy or scorecard. AP18 has
the stronger short-horizon profile; AP19 CatBoost mean-survival regression has
the stronger h5/h10 and symmetric-benefit profile. AP20 tests fixed causal
ensemble geometries without fitting a new label-dependent model.

Use the same 5,755 rows, early-2023 selection, opened-2024-2026 diagnostic,
today-effective CBR reference, known first next fixing, eligibility veto,
publication-h20 maturity support and AP17 controller. Historical publication
receipt remains calendar-assumed and CBR benefit is not bank execution P&L.

Inputs are the frozen quarterly-OOS AP18 `full_recent50` score and frozen
quarterly-OOS AP19 `cat_mean_utility` score. Eight candidates are fixed:

1. raw AP18/CatBoost weights 75/25, 50/50 and 25/75;
2. past-250 per-currency causal-rank weights 75/25, 50/50 and 25/75;
3. minimum of the two causal ranks;
4. geometric mean of the two causal ranks.

No fitted blend weight and no late-year routing. Missing one expert falls back
to the available expert; both missing stays missing. Causal ranks use only prior
scores with warmup 40. Every combined score then enters the unchanged AP17
`pace365_p55_r70_month24_cap2` controller. This nested rank is deliberate: the
inner rank aligns expert scales using past data; the outer rank applies the
frozen product threshold and cadence state.

Selection gates and order are unchanged: h3/h5/h10/h20 minimum lift >=1.3,
1..2 signals per currency-week, maximum two in an ISO week, no empty complete
currency-month, positive symmetric-benefit CI and at least 80% of simple-sign
future benefit; rank by minimum unknown-h lift, then mean lift, then gap.

Audit exact source hashes, expert reconstruction, all eight combined scores,
all controller states, future-score prefix corruption and paired 20/50-date
bootstrap versus AP18, AP17, AP12 and AP19 CatBoost utility. The stretch target
is strict minimum unknown-h lift above 2.4 at the required cadence. The later
period is not a fresh holdout.
