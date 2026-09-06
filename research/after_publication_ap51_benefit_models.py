"""Quarterly mature-only Ridge estimates of future-only CBR benefit."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_quarterly_benefit(features, target, maturity, dates,
                          min_train_date=None):
    features = np.asarray(features, dtype=float).copy()
    features[~np.isfinite(features)] = np.nan
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    prediction = np.full(len(dates), np.nan)
    prior = np.full(len(dates), np.nan)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        mature = np.array([value < cutoff for value in maturity])
        train = (dates < origin) & mature & np.isfinite(target)
        if min_train_date is not None:
            train &= dates >= min_train_date
        ids = np.flatnonzero(train)
        median = np.zeros(features.shape[1])
        if len(ids):
            finite_columns = np.isfinite(features[train]).any(axis=0)
            median[finite_columns] = np.nanmedian(
                features[train][:, finite_columns], axis=0)
        X_train = np.where(np.isfinite(features[train]), features[train], median)
        X_query = np.where(np.isfinite(features[query]), features[query], median)
        age = np.array([(origin - dates[i]).days for i in ids], dtype=float)
        weight = np.exp2(-age / 730.) if len(ids) else np.array([])
        weighted_mean = (float(np.average(target[train], weights=weight))
                         if len(ids) else 0.)
        prior[query] = weighted_mean
        n_train[query] = len(ids)
        constant = len(ids) < 200
        if constant:
            prediction[query] = weighted_mean
            lo = hi = weighted_mean
            coefficients = []
        else:
            lo, hi = np.quantile(target[train], [.01, .99])
            clipped = np.clip(target[train], lo, hi)
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
            model.fit(X_train, clipped, ridge__sample_weight=weight)
            prediction[query] = model.predict(X_query)
            coefficients = model.named_steps['ridge'].coef_.tolist()
        logs.append({
            'origin': str(origin), 'n_train': int(len(ids)),
            'n_query': int(query.sum()), 'constant_prediction': bool(constant),
            'weighted_train_mean': weighted_mean,
            'target_clip_lo': float(lo), 'target_clip_hi': float(hi),
            'coefficients': coefficients,
            'last_target_maturity': str(max(maturity[train])) if len(ids) else '',
            'min_train_date': str(min_train_date) if min_train_date else '',
        })
    return prediction, prior, n_train, logs
