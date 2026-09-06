"""Specialist hard-pool pace models for AP25."""
from __future__ import annotations

import datetime as dt

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from research.after_publication_ap1 import SEED


SPECS = {
    'pace_cat_mean_gate_full': ('cat_reg', 'mean', None),
    'pace_cat_mean_gate_roll3': ('cat_reg', 'mean', 1095),
    'pace_extra_mean_gate_full': ('extra_reg', 'mean', None),
    'pace_cat_y20_gate_full': ('cat_class', 'y20', None),
    'pace_cat_future5_gate_full': ('cat_reg', 'future5', None),
}


def _cat_common():
    return dict(
        iterations=320, depth=6, learning_rate=.035, l2_leaf_reg=10.,
        random_strength=.5, bootstrap_type='Bernoulli', subsample=.8,
        random_seed=SEED, thread_count=2, verbose=False,
        allow_writing_files=False,
    )


def specialist_target(labels, future5, kind, train):
    labels = np.asarray(labels, dtype=float)
    if kind == 'mean':
        return labels.mean(axis=1), None
    if kind == 'y20':
        return labels[:, 3], None
    if kind == 'future5':
        raw = np.asarray(future5, dtype=float)
        usable = np.asarray(train, dtype=bool) & np.isfinite(raw)
        if usable.any():
            lower, upper = np.quantile(raw[usable], [.02, .98])
        else:
            lower, upper = -200., 200.
        return np.clip(raw, lower, upper) / 200., (float(lower), float(upper))
    raise ValueError(kind)


def fit_specialist(X, labels, future5, train_pool, query, dates, spec):
    family, target_kind, rolling_days = spec
    use = np.asarray(train_pool, dtype=bool).copy()
    if rolling_days is not None:
        origin = min(np.asarray(dates)[query])
        use &= np.asarray(dates) >= origin - dt.timedelta(days=rolling_days)
    target, clipping = specialist_target(labels, future5, target_kind, use)
    usable = use & np.isfinite(target)
    stats = {
        'family_kind': family, 'target_kind': target_kind,
        'rolling_days': rolling_days, 'n_specialist_train': int(usable.sum()),
        'clip_lo': None if clipping is None else clipping[0],
        'clip_hi': None if clipping is None else clipping[1],
    }
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        value = float(np.nanmean(target[usable])) if usable.any() else 0.
        stats['fallback'] = True
        return np.repeat(value, int(np.sum(query))), stats
    if family == 'cat_reg':
        model = CatBoostRegressor(loss_function='RMSE', **_cat_common())
        model.fit(X[usable], target[usable])
        score = model.predict(X[query])
    elif family == 'cat_class':
        model = CatBoostClassifier(loss_function='Logloss', **_cat_common())
        model.fit(X[usable], target[usable].astype(int))
        score = model.predict_proba(X[query])[:, 1]
    elif family == 'extra_reg':
        model = ExtraTreesRegressor(
            n_estimators=400, max_depth=8, min_samples_leaf=25,
            max_features=.6, n_jobs=2, random_state=SEED)
        model.fit(X[usable], target[usable])
        score = model.predict(X[query])
    else:
        raise ValueError(family)
    stats['fallback'] = False
    return np.asarray(score, dtype=float), stats
