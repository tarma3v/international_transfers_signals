"""Phase-specific after-receipt market calibration helpers."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.after_publication_ap50_temperature_models import clipped_logit
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE


def delta_features(base_probability, delta_bps, available, currencies):
    currency_values = ('AMD', 'KGS', 'KZT', 'TJS', 'UZS')
    return np.column_stack([
        clipped_logit(base_probability), delta_bps / 100.,
        np.abs(delta_bps) / 100., np.asarray(available, dtype=float),
        *[(np.asarray(currencies) == value).astype(float)
          for value in currency_values],
    ])


def fit_quarterly_delta_calibrator(base_probability, delta_bps, available,
                                    target, maturity, dates, currencies):
    features = delta_features(
        base_probability, delta_bps, available, currencies)
    base_probability = np.asarray(base_probability, dtype=float)
    target = np.asarray(target, dtype=float)
    dates = np.asarray(dates, dtype=object)
    maturity = np.asarray(maturity, dtype=object)
    probability = np.full(len(dates), np.nan)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = ((dates >= origin) & (dates < end)
                 & np.isfinite(base_probability))
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        train = ((dates >= MIN_TRAIN_DATE) & (dates < origin)
                 & np.array([value < cutoff for value in maturity])
                 & np.isfinite(base_probability) & np.isfinite(target))
        ids = np.flatnonzero(train)
        n_train[query] = len(ids)
        if len(ids) < 200 or np.unique(target[train]).size < 2:
            weighted_prior = float(target[train].mean()) if len(ids) else .5
            probability[query] = weighted_prior
            coefficients = []
        else:
            age = np.array([(origin - dates[i]).days for i in ids], dtype=float)
            weight = np.exp2(-age / 730.)
            model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000)
            model.fit(features[train], target[train].astype(int),
                      sample_weight=weight)
            probability[query] = model.predict_proba(features[query])[:, 1]
            coefficients = model.coef_[0].tolist()
        logs.append({
            'origin': str(origin), 'n_train': int(len(ids)),
            'n_query': int(query.sum()), 'coefficients': coefficients,
            'last_target_maturity': str(max(maturity[train])) if len(ids) else '',
            'min_train_date': str(MIN_TRAIN_DATE),
        })
    return probability, n_train, logs
