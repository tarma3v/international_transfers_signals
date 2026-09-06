# AP51-T frozen protocol — expected future-only CBR benefit

Registered 2026-09-06 before AP51 predictions or metrics were computed. This is
a widget magnitude model, not a change to push signals and not an estimate of
executable bank savings.

For each h=3/5/10/20, predict the existing future-only CBR benefit in basis
points. Use the 25 season-light AP49 current-only features plus the AP50
calibrated probability for that horizon. Fit one quarterly StandardScaler +
Ridge(alpha=10) model with a 730-day half-life. Within every fit, winsorize the
training target at its own 1st/99th percentiles; never use query-period target
quantiles. A label may enter only when its publication-h maturity is earlier
than quarter origin minus two calendar days. With fewer than 200 train rows,
emit the weighted train mean.

Evaluate the already opened later period with MAE and RMSE against a quarterly
train-mean baseline, Spearman correlation, sign accuracy, and fixed predicted
quintile calibration overall/by currency/year. Save expected basis points next
to AP50 probability and expose them from `score_as_of`. Keep the units labelled
`CBR future-only bps`; no conversion to rubles or fees is allowed.
