"""Fresh AP12 conditional rankers and causal capped-rank controller."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from ml.data import CORRIDORS
from research.after_publication_ap1 import SEED, factory


MODEL_SCORES = (
    'extra_h5',
    'compact_hist_h5',
    'compact_extra_h5',
    'local_hist_h5',
    'first_failure_extra_h5',
)


def compact_feature_indices(names):
    """Fixed, lower-dimensional feature set registered before AP12 fitting."""
    exact = {
        'known_change', 'known_change_z', 'peer_change_mean', 'peer_change_std',
        'local_minus_common', 'effective_age_days', 'next_effective_gap',
        'dow_sin', 'dow_cos', 'annual_sin', 'annual_cos', 'pre_new_year14',
        'month_end', 'after2022',
    }
    prefixes = (
        'announced_ret', 'announced_vol', 'announced_range',
        'effective_ret', 'effective_vol', 'effective_range',
        'currency_',
    )
    suffixes = ('_source_age_days',)
    index = [i for i, name in enumerate(names)
             if name in exact or name.startswith(prefixes) or name.endswith(suffixes)]
    if not index or 'known_change_z' not in [names[i] for i in index]:
        raise ValueError('Compact AP12 feature subset is invalid')
    return np.array(index, dtype=int)


def extra_factory():
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=25,
        max_features=.6,
        n_jobs=2,
        random_state=SEED,
    )


def _fit_probability(model, X, y, train, query):
    usable = train & np.isfinite(y)
    if usable.sum() < 60 or np.unique(y[usable]).size < 2:
        value = float(np.nanmean(y[usable])) if usable.any() else 0.
        return np.repeat(value, query.sum())
    model.fit(X[usable], y[usable])
    return model.predict_proba(X[query])[:, list(model.classes_).index(1)]


def fit_extra(X, y, train, query):
    return _fit_probability(extra_factory(), X, y, train, query)


def fit_hist(X, y, train, query):
    model = factory('hist7y').set_params(early_stopping=False)
    return _fit_probability(model, X, y, train, query)


def fit_local_hist(X, y, train, query, currencies, global_prediction, shrink=150):
    """Currency models shrunk toward the identically trained global model."""
    result = np.asarray(global_prediction, dtype=float).copy()
    query_ids = np.flatnonzero(query)
    counts = {}
    for currency in CORRIDORS:
        usable = train & (currencies == currency) & np.isfinite(y)
        local_query = currencies[query] == currency
        counts[currency] = int(usable.sum())
        if usable.sum() < 100 or np.unique(y[usable]).size < 2 or not local_query.any():
            continue
        model = factory('hist7y').set_params(early_stopping=False)
        model.fit(X[usable], y[usable])
        local = model.predict_proba(X[query_ids[local_query]])[:, 1]
        weight = usable.sum() / (usable.sum() + shrink)
        result[local_query] = weight * local + (1 - weight) * result[local_query]
    return result, counts


def first_failure_class(conditional):
    """0..3 are first failing intervals; 4 survives through step20."""
    y = np.asarray(conditional, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4:
        raise ValueError('Expected conditional h3/h5/h10/h20 labels')
    result = np.full(len(y), np.nan)
    complete = np.isfinite(y).all(axis=1)
    for i in np.flatnonzero(complete):
        zero = np.flatnonzero(y[i] == 0)
        result[i] = int(zero[0]) if len(zero) else 4
    return result


def fit_first_failure_extra(X, conditional, train, query):
    target = first_failure_class(conditional)
    usable = train & np.isfinite(target)
    model = extra_factory()
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        return np.repeat(float(np.mean(target[usable] >= 2)), query.sum()), {}
    model.fit(X[usable], target[usable].astype(int))
    probability = model.predict_proba(X[query])
    classes = model.classes_.astype(int)
    h5 = probability[:, classes >= 2].sum(axis=1)
    return h5, {str(int(c)): int(np.sum(target[usable] == c)) for c in classes}


def causal_rank_cap2(values, dates, currencies, eligible, rate, minimum_gap=0,
                     window=250, warmup=40):
    """Strict trailing-rank threshold followed by an online ISO-week budget."""
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=bool)
    for currency in CORRIDORS:
        history = []
        week, used, last = None, 0, None
        for i in np.flatnonzero(currencies == currency):
            value = values[i]
            iso = dates[i].isocalendar()[:2]
            if iso != week:
                week, used = iso, 0
            opportunity = False
            if np.isfinite(value) and len(history) >= warmup:
                threshold = float(np.quantile(history[-window:], 1 - rate))
                opportunity = value > threshold
            far_enough = last is None or (dates[i] - last).days >= minimum_gap
            if opportunity and eligible[i] and used < 2 and far_enough:
                result[i] = True
                used += 1
                last = dates[i]
            if np.isfinite(value):
                history.append(value)
    return result


def causal_rank_percentile(values, currencies, window=250, warmup=40):
    """Past-only CDF used for the fixed known/model rank blend."""
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for currency in CORRIDORS:
        history = []
        for i in np.flatnonzero(currencies == currency):
            value = values[i]
            if np.isfinite(value) and len(history) >= warmup:
                ref = np.asarray(history[-window:])
                result[i] = float(np.mean(ref < value) + .5 * np.mean(ref == value))
            if np.isfinite(value):
                history.append(value)
    return result
