# AP50-T frozen protocol — calibrated after-publication temperature

Registered 2026-09-06 before AP50 calibration predictions or metrics were
computed. This packet evaluates a widget output and does not change the sparse
push policy or claim a new lift.

For each h in 3/5/10/20, take the frozen AP49 quarterly OOS head probability.
Fit a quarterly Platt-style LogisticRegression on its clipped logit plus five
currency one-hot indicators. Use C=1, lbfgs, max_iter=1000, no class balancing,
and fixed 730-day half-life weights. A query quarter may train only on earlier
rows whose own AP49 score is OOS and whose publication-h label matured before
quarter origin minus two calendar days. If fewer than 200 rows or only one class
is available, emit the weighted train prior. Also save that prior as the honest
constant-probability baseline.

Evaluate frozen raw, calibrated, and train-prior probabilities on the already
opened later period using Brier score and log-loss. Report ROC AUC and average
precision as discrimination diagnostics, fixed-width reliability bins, ECE,
and currency/year slices. The widget temperature for each h is 100 times the
calibrated probability; the composite is the geometric mean of the four
calibrated probabilities. It remains a probability-like attractiveness score,
not bank savings.

Implement a latest-valid-snapshot lookup that requires timezone-aware `as_of`,
never selects a future receipt, returns source age/freshness/phase/evidence, and
keeps push state separate. Historical source times remain explicitly
`calendar_assumed`; no intraday or executable-bank validation is claimed.
