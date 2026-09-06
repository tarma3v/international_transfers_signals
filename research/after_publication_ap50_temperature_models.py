"""Causal quarterly probability calibration for the temperature widget."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data import CORRIDORS


def clipped_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1. - 1e-6)
    return np.log(probability / (1. - probability))


def calibration_features(probability, currencies):
    currencies = np.asarray(currencies)
    return np.column_stack([
        clipped_logit(probability),
        *[(currencies == value).astype(float) for value in CORRIDORS],
    ])


def fit_quarterly_calibrator(probability, target, maturity, dates, currencies,
                             min_train_date=None):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    features = calibration_features(probability, currencies)
    calibrated = np.full(len(dates), np.nan)
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
        query = (dates >= origin) & (dates < end) & np.isfinite(probability)
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        mature = np.array([value < cutoff for value in maturity])
        train = ((dates < origin) & mature & np.isfinite(probability)
                 & np.isfinite(target))
        if min_train_date is not None:
            train &= dates >= min_train_date
        ids = np.flatnonzero(train)
        age = np.array([(origin - dates[i]).days for i in ids], dtype=float)
        weight = np.exp2(-age / 730.) if len(ids) else np.array([])
        weighted_prior = (float(np.average(target[train], weights=weight))
                          if len(ids) else .5)
        prior[query] = weighted_prior
        n_train[query] = len(ids)
        constant = len(ids) < 200 or np.unique(target[train]).size < 2
        if constant:
            calibrated[query] = weighted_prior
            coefficient = []
            intercept = float(clipped_logit(weighted_prior))
        else:
            model = LogisticRegression(
                C=1., solver='lbfgs', max_iter=1000)
            model.fit(features[train], target[train].astype(int),
                      sample_weight=weight)
            calibrated[query] = model.predict_proba(features[query])[:, 1]
            coefficient = model.coef_[0].tolist()
            intercept = float(model.intercept_[0])
        logs.append({
            'origin': str(origin),
            'n_train': int(len(ids)),
            'n_positive': int(target[train].sum()) if len(ids) else 0,
            'weighted_prior': weighted_prior,
            'n_query': int(query.sum()),
            'constant_prediction': bool(constant),
            'intercept': intercept,
            'coefficients': coefficient,
            'last_target_maturity': str(max(maturity[train])) if len(ids) else '',
            'min_train_date': str(min_train_date) if min_train_date else '',
        })
    return calibrated, prior, n_train, logs


def expected_calibration_error(probability, target, bins=10):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(probability) & np.isfinite(target)
    probability, target = probability[valid], target[valid]
    edges = np.linspace(0., 1., bins + 1)
    ids = np.minimum(np.searchsorted(edges, probability, side='right') - 1,
                     bins - 1)
    total = max(len(probability), 1)
    return float(sum(
        (ids == i).sum() / total
        * abs(probability[ids == i].mean() - target[ids == i].mean())
        for i in range(bins) if (ids == i).any()
    ))
