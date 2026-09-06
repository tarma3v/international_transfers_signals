"""Conditional models for steps 2..20 after the next fixing is already known."""
import datetime as dt

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from ml.targets import HORIZONS
from research.after_publication_ap1 import SEED, factory
from research.after_publication_ap4_survival import survival_from_hazards

UNKNOWN_H = np.array((3, 5, 10, 20))
FAMILIES = ('hist', 'recent_hist', 'logit', 'local_logit', 'hazard_hist',
            'hazard_logit', 'margin_q25', 'margin_ridge_local')


def eligible_next(panel):
    return panel.announced_price.to_numpy(float) >= panel.current_price.to_numpy(float)


def conditional_targets(outcomes):
    result = np.column_stack([outcomes['y' + str(h)] for h in UNKNOWN_H]).astype(float)
    if not np.isin(result[np.isfinite(result)], (0, 1)).all():
        raise ValueError('Conditional targets must be binary')
    if (np.diff(result[np.isfinite(result).all(axis=1)], axis=1) > 0).any():
        raise ValueError('Conditional survival must be non-increasing')
    return result


def margin_targets(series, panel, scale):
    """Min of unknown steps2..h relative current; training labels only."""
    result = np.full((len(panel), len(UNKNOWN_H)), np.nan)
    for row, event in enumerate(panel.itertuples()):
        values = series[event.currency].values
        i = int(event.current_index)
        for j, h in enumerate(UNKNOWN_H):
            if i + h < len(values):
                result[row, j] = 1e4 * np.log(values[i + 2:i + h + 1].min() / values[i]) / scale[row]
    return result


def effective_scale(announced_X, names):
    scale = announced_X[:, names.index('effective_vol20')]
    return np.maximum(scale.astype(float), 1.)


def person_period(X, survival):
    """Four unknown intervals only; no interval for the already known step1."""
    y = np.asarray(survival)
    if y.shape != (len(X), 4) or not np.isfinite(y).all():
        raise ValueError('Only complete eligible conditional outcomes')
    risk = np.column_stack([np.ones(len(y), dtype=bool), y[:, :-1] == 1])
    rows, intervals = np.where(risk)
    design = np.column_stack([X[rows], np.eye(4)[intervals]])
    target = 1 - y[rows, intervals]
    return design, target, rows, intervals


def prediction_design(X):
    n = len(X)
    return np.column_stack([np.repeat(X, 4, axis=0), np.tile(np.eye(4), (n, 1))])


def fit_classifier(X, y, train, query, kind, sample_weight=None):
    predictions = np.zeros((query.sum(), 4))
    for j in range(4):
        yy = y[:, j]
        tr = train & np.isfinite(yy)
        if tr.sum() < 60 or np.unique(yy[tr]).size < 2:
            predictions[:, j] = yy[tr].mean() if tr.any() else 0.
            continue
        model = (factory('hist7y').set_params(early_stopping=False) if kind == 'hist'
                 else make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500)))
        kwargs = {}
        if sample_weight is not None:
            if kind == 'hist':
                kwargs['sample_weight'] = sample_weight[tr]
            else:
                kwargs['logisticregression__sample_weight'] = sample_weight[tr]
        model.fit(X[tr], yy[tr], **kwargs)
        predictions[:, j] = model.predict_proba(X[query])[:, 1]
    return predictions


def fit_local_logit(X, y, train, query, currencies, global_prediction, shrink=150):
    result = global_prediction.copy()
    query_ids = np.flatnonzero(query)
    counts = {}
    for currency in CORRIDORS:
        tr = train & (currencies == currency)
        qlocal = currencies[query] == currency
        counts[currency] = int(tr.sum())
        if not qlocal.any() or tr.sum() < 60:
            continue
        for j in range(4):
            yy = y[:, j]
            usable = tr & np.isfinite(yy)
            if np.unique(yy[usable]).size < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500))
            model.fit(X[usable], yy[usable])
            local = model.predict_proba(X[query_ids[qlocal]])[:, 1]
            weight = usable.sum() / (usable.sum() + shrink)
            result[qlocal, j] = weight * local + (1 - weight) * global_prediction[qlocal, j]
    return result, counts


def fit_hazard(X, survival, train, query, kind):
    design, target, rows, intervals = person_period(X[train], survival[train])
    if len(np.unique(target)) < 2:
        hazard = np.array([target[intervals == j].mean() for j in range(4)])
        pred = np.tile(hazard, (query.sum(), 1))
    else:
        model = (factory('hist7y').set_params(early_stopping=False) if kind == 'hist'
                 else make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500)))
        model.fit(design, target)
        pred = model.predict_proba(prediction_design(X[query]))[:, 1].reshape(-1, 4)
    return survival_from_hazards(pred), len(target), int(target.sum()), rows, intervals


def fit_margin(X, margin, train, query, kind, currencies):
    result = np.zeros((query.sum(), 4))
    query_ids = np.flatnonzero(query)
    local_counts = {}
    for j in range(4):
        y = margin[:, j]
        tr = train & np.isfinite(y)
        if kind == 'q25':
            model = HistGradientBoostingRegressor(loss='quantile', quantile=.25, max_iter=160,
                learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=5., early_stopping=False, random_state=SEED)
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
        model.fit(X[tr], y[tr])
        global_prediction = model.predict(X[query])
        result[:, j] = global_prediction
        if kind != 'ridge_local':
            continue
        for currency in CORRIDORS:
            usable = tr & (currencies == currency)
            qlocal = currencies[query] == currency
            local_counts[currency] = int(usable.sum())
            if usable.sum() < 60 or not qlocal.any():
                continue
            local = make_pipeline(StandardScaler(), Ridge(alpha=100.))
            local.fit(X[usable], y[usable])
            prediction = local.predict(X[query_ids[qlocal]])
            weight = usable.sum() / (usable.sum() + 150)
            result[qlocal, j] = weight * prediction + (1 - weight) * global_prediction[qlocal]
    return result, local_counts


def full_curve(conditional, eligible):
    conditional = np.asarray(conditional)
    if conditional.shape != (len(eligible), 4):
        raise ValueError('Conditional predictions must match the query rows')
    result = np.zeros((len(eligible), 5))
    result[eligible, 0] = 1.
    result[eligible, 1:] = np.minimum.accumulate(conditional[eligible], axis=1)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError('Invalid conditional probability curve')
    return result


def margin_scores(prediction, known_buffer_z, eligible):
    """Combine predicted unknown minima with the exactly known first margin."""
    result = np.full((len(eligible), 5), -1e6)
    result[eligible, 0] = known_buffer_z[eligible]
    result[eligible, 1:] = np.minimum(np.asarray(prediction)[eligible], known_buffer_z[eligible, None])
    if not np.isfinite(result).all():
        raise ValueError('Invalid margin scores')
    return result
