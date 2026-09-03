# Research references and decisions

Accessed 4 September 2026.  These sources informed the protocol; they are not
features unless a script explicitly says so.

- Case statement: <https://talenttrack.aitalenthub.ru/hackathon/cases/455>.
  We implement the future-only event separately from the symmetric benefit
  metric, evaluate h in 1/3/5/10/20, and report 1--2 signals per corridor/week.
- Bank of Russia daily rates: <https://www.cbr.ru/currency_base/daily/>.
  The page identifies rates by their effective date; the historical XML source
  is normalized for changes in currency nominal.
- Bank of Russia publication timing FAQ:
  <https://www.cbr.ru/Reception/TopicalMessage/Page/2661>.  The CBR says exact
  timing is not regulated, but the rates are normally posted by 18:00 Moscow.
  This is why the next-effective-rate experiment is conditional on an explicit
  post-publication product cutoff and isolated from ordinary causal features.
- Bank of Russia cross-border transfers database:
  <https://www.cbr.ru/hd_base/tg/?tab.current=t2>.  It exposes quarterly and
  annual outflow/inflow data.  Such data can describe demand regimes but must be
  lagged to its release timestamp before use in a daily signal.
- Bank of Russia, Financial Market Risk Review, October 2023:
  <https://www.cbr.ru/Collection/Collection/File/46563/ORFR_2023-10.pdf>.
  The review shows that 2023 transfer volumes and their destination/currency
  mix changed materially and not monotonically, so a blanket claim that demand
  simply rose after SWIFT restrictions is too strong.
- Hyndman et al., time-series cross-validation:
  <https://pkg.robjhyndman.com/forecast/reference/tsCV.html>.  Successive
  rolling origins motivate our purged chronological folds.
- Montero-Manso and Hyndman et al., Global Models for Time Series Forecasting:
  <https://arxiv.org/abs/2012.12485>.  Pooled regression/boosting can benefit
  related short series, motivating a global panel with currency identity.
- Rossi, "Are Exchange Rates Really Random Walks?":
  <https://doi.org/10.1017/S1365100506050085>.  Parameter instability can hide
  predictability, motivating expanding, rolling, decay-weighted, and explicitly
  post-shock comparisons.
- Beckmann et al., "Forecasting exchange rates under parameter and model
  uncertainty": <https://doi.org/10.1016/j.jimonfin.2015.07.001>.  Model
  averaging and shrinkage are useful when relevant predictors change.
- Ahmed, Liu and Valente, "Can currency-based risk factors help forecast
  exchange rates?": <https://doi.org/10.1016/j.ijforecast.2015.01.010>.
  Their broad negative out-of-sample result supports retaining naive and simple
  anchors instead of assuming a complex learner must win.
