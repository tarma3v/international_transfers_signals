"""AP19 multi-horizon ExtraTrees, CatBoost and pairwise ranking models."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

from research.after_publication_ap1 import SEED
from research.after_publication_ap12_effective_models import extra_factory


SCORE_NAMES = (
    'extra_multi_mean',
    'extra_multi_geom',
    'cat_h5',
    'cat_multi_mean',
    'cat_multi_geom',
    'cat_mean_utility',
    'cat_pairrank_h5',
    'base75_cat25',
    'base50_cat50',
    'base75_catmulti25',
)


def _valid(train, labels):
    return np.asarray(train, dtype=bool) & np.isfinite(labels).all(axis=1)


def _cat_common():
    return dict(
        iterations=320,
        depth=6,
        learning_rate=.035,
        l2_leaf_reg=10.,
        random_strength=.5,
        bootstrap_type='Bernoulli',
        subsample=.8,
        random_seed=SEED,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )


def extra_multi(X, labels, train, query):
    """Four independent conditional-survival ExtraTrees probabilities."""
    usable = _valid(train, labels)
    result = np.zeros((int(np.sum(query)), labels.shape[1]), dtype=float)
    for j in range(labels.shape[1]):
        target = labels[:, j]
        if usable.sum() < 100 or np.unique(target[usable]).size < 2:
            result[:, j] = float(np.mean(target[usable])) if usable.any() else 0.
            continue
        model = extra_factory()
        model.fit(X[usable], target[usable])
        result[:, j] = model.predict_proba(X[query])[:, 1]
    return np.clip(result, 0, 1)


def cat_multi(X, labels, train, query):
    """Four fixed CatBoost classifiers, one for each unknown horizon."""
    usable = _valid(train, labels)
    result = np.zeros((int(np.sum(query)), labels.shape[1]), dtype=float)
    for j in range(labels.shape[1]):
        target = labels[:, j]
        if usable.sum() < 100 or np.unique(target[usable]).size < 2:
            result[:, j] = float(np.mean(target[usable])) if usable.any() else 0.
            continue
        model = CatBoostClassifier(loss_function='Logloss', **_cat_common())
        model.fit(X[usable], target[usable].astype(int))
        result[:, j] = model.predict_proba(X[query])[:, 1]
    return np.clip(result, 0, 1)


def cat_mean_utility(X, labels, train, query):
    """Regression to the average survival across h3/h5/h10/h20."""
    usable = _valid(train, labels)
    target = labels.mean(axis=1)
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        value = float(np.mean(target[usable])) if usable.any() else 0.
        return np.repeat(value, int(np.sum(query)))
    model = CatBoostRegressor(loss_function='RMSE', **_cat_common())
    model.fit(X[usable], target[usable])
    return np.clip(model.predict(X[query]), 0, 1)


def cat_pairrank(X, labels, train, query, currencies):
    """PairLogit score learned only from within-currency h5 comparisons."""
    usable = _valid(train, labels)
    target = labels[:, 1]
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        value = float(np.mean(target[usable])) if usable.any() else 0.
        return np.repeat(value, int(np.sum(query)))
    indices = np.flatnonzero(usable)
    order = np.argsort(np.asarray(currencies)[indices], kind='stable')
    indices = indices[order]
    model = CatBoostRanker(loss_function='PairLogit', **_cat_common())
    model.fit(X[indices], target[indices], group_id=np.asarray(currencies)[indices])
    raw = np.clip(model.predict(X[query]), -30, 30)
    return 1. / (1. + np.exp(-raw))


def score_family(extra_curve, cat_curve, cat_utility, cat_rank, base):
    """Build the ten fixed AP19 scores from model outputs."""
    eps = 1e-6
    extra_mean = extra_curve.mean(axis=1)
    extra_geom = np.exp(np.log(np.clip(extra_curve, eps, 1)).mean(axis=1))
    cat_h5 = cat_curve[:, 1]
    cat_mean = cat_curve.mean(axis=1)
    cat_geom = np.exp(np.log(np.clip(cat_curve, eps, 1)).mean(axis=1))
    base = np.asarray(base, dtype=float)
    usable_base = np.isfinite(base)

    def blend(weight, component):
        result = np.asarray(component, dtype=float).copy()
        result[usable_base] = weight * base[usable_base] + (1 - weight) * result[usable_base]
        return np.clip(result, 0, 1)

    return {
        'extra_multi_mean': np.clip(extra_mean, 0, 1),
        'extra_multi_geom': np.clip(extra_geom, 0, 1),
        'cat_h5': np.clip(cat_h5, 0, 1),
        'cat_multi_mean': np.clip(cat_mean, 0, 1),
        'cat_multi_geom': np.clip(cat_geom, 0, 1),
        'cat_mean_utility': np.clip(cat_utility, 0, 1),
        'cat_pairrank_h5': np.clip(cat_rank, 0, 1),
        'base75_cat25': blend(.75, cat_h5),
        'base50_cat50': blend(.50, cat_h5),
        'base75_catmulti25': blend(.75, cat_mean),
    }
