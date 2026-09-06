"""AP18 residual, partial-pooling and stacking predictors."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from research.after_publication_ap1 import SEED


SCORE_NAMES = (
    'full_recent50',
    'resid_hist100',
    'resid_hist50',
    'resid_ridge',
    'local_resid_ridge',
    'stack_logit',
)


def fixed_blend(base, recent):
    return .5 * np.asarray(base, dtype=float) + .5 * np.asarray(recent, dtype=float)


def _usable(y, base, train):
    return np.asarray(train, dtype=bool) & np.isfinite(y) & np.isfinite(base)


def residual_hist(X, y, base, train, query):
    usable = _usable(y, base, train)
    if usable.sum() < 100 or np.unique(y[usable]).size < 2:
        return np.asarray(base[query], dtype=float), np.zeros(query.sum())
    model = HistGradientBoostingRegressor(
        max_iter=160, learning_rate=.04, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=10., early_stopping=False,
        random_state=SEED,
    )
    model.fit(X[usable], y[usable] - base[usable])
    correction = model.predict(X[query])
    return np.clip(base[query] + correction, 0, 1), correction


def residual_ridge(X, y, base, train, query):
    usable = _usable(y, base, train)
    if usable.sum() < 100 or np.unique(y[usable]).size < 2:
        return np.asarray(base[query], dtype=float), np.zeros(query.sum())
    model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
    model.fit(X[usable], y[usable] - base[usable])
    correction = model.predict(X[query])
    return np.clip(base[query] + correction, 0, 1), correction


def local_residual_ridge(X, y, base, train, query, currencies, shrink=200):
    global_score, global_correction = residual_ridge(X, y, base, train, query)
    result = global_score.copy()
    correction = global_correction.copy()
    query_ids = np.flatnonzero(query)
    counts = {}
    for currency in CORRIDORS:
        usable = _usable(y, base, train) & (currencies == currency)
        local_query = currencies[query] == currency
        counts[currency] = int(usable.sum())
        if usable.sum() < 100 or np.unique(y[usable]).size < 2 or not local_query.any():
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
        model.fit(X[usable], y[usable] - base[usable])
        local = model.predict(X[query_ids[local_query]])
        weight = usable.sum() / (usable.sum() + shrink)
        mixed = weight * local + (1 - weight) * global_correction[local_query]
        correction[local_query] = mixed
        result[local_query] = np.clip(base[query_ids[local_query]] + mixed, 0, 1)
    return result, correction, counts


def stack_logit(X, y, base, train, query):
    usable = _usable(y, base, train)
    if usable.sum() < 100 or np.unique(y[usable]).size < 2:
        return np.asarray(base[query], dtype=float)
    clipped = np.clip(base, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1 - clipped))[:, None]
    design = np.column_stack([X, logit])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.05, max_iter=2000, solver='lbfgs', random_state=SEED),
    )
    model.fit(design[usable], y[usable].astype(int))
    return model.predict_proba(design[query])[:, 1]
