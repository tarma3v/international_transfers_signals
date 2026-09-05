"""Discrete first-cheaper-price hazards, trained only on fully mature events."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.targets import HORIZONS
from research.after_publication_ap1 import factory


def prediction_design(X):
    """Original-time covariates plus interval identity; no outcomes accepted."""
    n, k = len(X), len(HORIZONS)
    return np.column_stack((np.repeat(X, k, axis=0), np.tile(np.eye(k), (n, 1))))


def person_period(X, survival):
    """Exclude intervals after first failure; equality with current price survives."""
    y = np.asarray(survival)
    if y.shape != (len(X), len(HORIZONS)) or not np.isfinite(y).all():
        raise ValueError('Only complete mature outcomes are accepted')
    if not np.isin(y, (0, 1)).all() or (np.diff(y, axis=1) > 0).any():
        raise ValueError('Survival labels must be binary and non-increasing')
    risk = np.column_stack((np.ones(len(y), bool), y[:, :-1] == 1))
    rows = np.repeat(np.arange(len(X)), len(HORIZONS))[risk.ravel()]
    intervals = np.tile(np.arange(len(HORIZONS)), len(X))[risk.ravel()]
    return prediction_design(X)[risk.ravel()], (1 - y)[risk], rows, intervals


def survival_from_hazards(hazards):
    hazards = np.asarray(hazards)
    if not np.isfinite(hazards).all() or ((hazards < 0) | (hazards > 1)).any():
        raise ValueError('Hazards must be finite probabilities')
    return np.cumprod(1 - hazards, axis=1)


def fit_predict_hazard(X_train, y_train, X_test, kind, min_events):
    design, target, _, intervals = person_period(X_train, y_train)
    if len(X_train) < min_events or len(np.unique(target)) < 2:
        hazard = np.array([target[intervals == j].mean() if (intervals == j).any()
                           else 0. for j in range(len(HORIZONS))])
        predictions = np.tile(hazard, (len(X_test), 1))
    else:
        if kind == 'logit':
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500))
        elif kind == 'hist':
            model = factory('hist7y').set_params(early_stopping=False)
        else:
            raise ValueError(kind)
        model.fit(design, target)
        predictions = model.predict_proba(prediction_design(X_test))[:, 1].reshape(-1, len(HORIZONS))
    return survival_from_hazards(predictions), len(target), int(target.sum())


def restricted_wait_targets(series, panel):
    """1..20 for first strictly cheaper observation,21 for none; NaN if incomplete."""
    waits = np.full(len(panel), np.nan)
    for row, (currency, index) in enumerate(zip(panel.currency, panel.announced_index)):
        v = series[currency].values
        i = int(index)
        if i + 20 >= len(v):
            continue
        cheaper = np.flatnonzero(v[i + 1:i + 21] < v[i])
        waits[row] = int(cheaper[0] + 1) if len(cheaper) else 21
    return waits
