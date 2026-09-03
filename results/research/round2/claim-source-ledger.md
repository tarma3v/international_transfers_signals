# Claim-source ledger

| Report claim | Evidence type | Source |
|---|---|---|
| The target is a five-publication current-minimum event and the operating range is 1-2 alerts per corridor per week | Case definition mirrored in repository | https://talenttrack.aitalenthub.ru/hackathon/cases/455 |
| Official CBR rates are the modeled series | Primary official data | https://www.cbr.ru/currency_base/dynamics/ |
| The next effective official rate is usually posted by 18:00 Moscow time, with no exact guaranteed minute | Primary official clarification | https://www.cbr.ru/Reception/TopicalMessage/Page/2661 |
| Pooling related series can help global models match or beat local models | Peer-reviewed forecasting research | https://doi.org/10.1016/j.ijforecast.2021.03.028 |
| Statistical model plus neural residual learning can outperform either component in large forecasting tasks | Peer-reviewed M4 hybrid paper | https://doi.org/10.1016/j.ijforecast.2019.03.017 |
| Exchange-rate predictability varies with predictors, horizon, sample, and evaluation method | Peer-reviewed survey | https://doi.org/10.1257/jel.51.4.1063 |
| Structural instability is a central reason forecast performance fails to transfer | Peer-reviewed paper | https://doi.org/10.1017/S1365100506050085 |
| Averaging forecasts over estimation windows can reduce break sensitivity | Primary working paper | https://www.ifo.de/en/cesifo/publications/2008/working-paper/forecasting-random-walks-under-drift-instability |
| Conditional predictive ability asks when one forecast should be preferred to another | Peer-reviewed paper/full text | https://doi.org/10.1111/j.1468-0262.2006.00718.x |
| Searching many models needs a data-snooping correction | Peer-reviewed papers | https://doi.org/10.1111/1468-0262.00152 ; https://doi.org/10.1198/073500105000000063 |
| A model confidence set should remain broad when the data are not informative | Peer-reviewed paper | https://doi.org/10.3982/ECTA5771 |
| RUONIA publication metadata and history come from the Bank of Russia | Primary official data | https://www.cbr.ru/hd_base/ruonia/dynamics/ |
| Key-rate history comes from the Bank of Russia | Primary official data | https://www.cbr.ru/hd_base/keyrate/ |
| Brent and broad-dollar series come from EIA/Federal Reserve via FRED | Primary publishers via official repository | https://fred.stlouisfed.org/series/DCOILBRENTEU ; https://fred.stlouisfed.org/series/DTWEXBGS |
| All numerical model results in the report | Reproducible local experiment outputs | `results/research/round2/*.csv`, `results/research/*.csv` |

