"""Announced-anchor decayed residual survival model for AP34."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS


RIDGE_ALPHA = 100.0
LOCAL_SHRINK = 150.0
GLOBAL_HALF_LIFE = 730.0
LOCAL_HALF_LIFE = 365.0


def _decay_weights(dates, origin, half_life):
    age = np.array([(origin - day).days for day in dates], dtype=float)
    if np.any(age <= 0):
        raise ValueError('Residual state may only use dates before origin')
    return np.power(.5, age / half_life)


def _survival_probability(errors, weights, thresholds):
    errors = np.asarray(errors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if not len(errors) or not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.full(len(thresholds), .5)
    return ((errors[:, None] >= thresholds[None, :])
            * weights[:, None]).sum(axis=0) / weights.sum()


def fit_residual_survival(X, target, train, query, dates, currencies,
                          known_change, origin):
    """Fit one Ridge and train-only global/local empirical error survival."""
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)
    train = np.asarray(train, dtype=bool) & np.isfinite(target)
    query = np.asarray(query, dtype=bool)
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    known_change = np.asarray(known_change, dtype=float)
    train_ids = np.flatnonzero(train)
    query_ids = np.flatnonzero(query)
    if len(train_ids) < 100:
        value = float(np.mean(target[train])) if train.any() else 0.
        return np.repeat(.5, len(query_ids)), {
            'ridge_prediction': np.repeat(value, len(query_ids)),
            'local_bias': np.zeros(len(query_ids)),
            'local_weight': np.zeros(len(query_ids)),
            'global_probability': np.repeat(.5, len(query_ids)),
            'local_probability': np.repeat(.5, len(query_ids)),
        }, {'n_train': int(len(train_ids)), 'fallback': True}

    lo, hi = np.quantile(target[train_ids], [.01, .99])
    clipped = np.clip(target[train_ids], lo, hi)
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(X[train_ids], clipped)
    train_prediction = model.predict(X[train_ids])
    query_prediction = model.predict(X[query_ids])
    raw_error = clipped - train_prediction

    train_dates = dates[train_ids]
    train_currency = currencies[train_ids]
    query_currency = currencies[query_ids]
    global_weights = _decay_weights(train_dates, origin, GLOBAL_HALF_LIFE)
    correction = np.zeros(len(train_ids))
    query_bias = np.zeros(len(query_ids))
    query_weight = np.zeros(len(query_ids))
    state = {}
    for currency in CORRIDORS:
        local = train_currency == currency
        n_local = int(local.sum())
        weight = n_local / (n_local + LOCAL_SHRINK)
        if n_local:
            local_weights = _decay_weights(
                train_dates[local], origin, LOCAL_HALF_LIFE)
            bias = float(np.average(raw_error[local], weights=local_weights))
        else:
            bias = 0.
        correction[local] = weight * bias
        local_query = query_currency == currency
        query_bias[local_query] = bias
        query_weight[local_query] = weight
        state[currency] = {'count': n_local, 'bias': bias, 'weight': weight}

    centered_error = raw_error - correction
    margin = (known_change[query_ids] + query_prediction
              + query_weight * query_bias)
    threshold = -margin
    global_probability = _survival_probability(
        centered_error, global_weights, threshold)
    local_probability = global_probability.copy()
    for currency in CORRIDORS:
        local_train = train_currency == currency
        local_query = query_currency == currency
        if not local_query.any() or not local_train.any():
            continue
        weights = _decay_weights(
            train_dates[local_train], origin, LOCAL_HALF_LIFE)
        local_probability[local_query] = _survival_probability(
            centered_error[local_train], weights, threshold[local_query])
    probability = ((1 - query_weight) * global_probability
                   + query_weight * local_probability)
    detail = {
        'ridge_prediction': query_prediction,
        'local_bias': query_bias,
        'local_weight': query_weight,
        'global_probability': global_probability,
        'local_probability': local_probability,
    }
    stats = {
        'n_train': int(len(train_ids)),
        'fallback': False,
        'clip_lo': float(lo),
        'clip_hi': float(hi),
        'train_target_mean': float(np.mean(clipped)),
        'train_error_std': float(np.std(centered_error)),
        'local_state': state,
    }
    return np.clip(probability, 0, 1), detail, stats
