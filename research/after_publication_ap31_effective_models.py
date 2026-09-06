"""Factorized short/long survival model for AP31."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

from research.after_publication_ap1 import SEED


SCORE = 'cat_factor_y3_y20'


def factorized_score(p3, p20_given3):
    return np.clip(np.asarray(p3, dtype=float), 0., 1.) * np.clip(
        np.asarray(p20_given3, dtype=float), 0., 1.)


def _fit_stage(X, target, train, query):
    target = np.asarray(target, dtype=float)
    usable = np.asarray(train, dtype=bool) & np.isfinite(target)
    stats = {'n_train': int(usable.sum())}
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        value = float(np.mean(target[usable])) if usable.any() else 0.
        stats['fallback'] = True
        return np.repeat(value, int(np.sum(query))), stats
    model = CatBoostClassifier(
        loss_function='Logloss', iterations=360, depth=6, learning_rate=.035,
        l2_leaf_reg=10., random_strength=.5, bootstrap_type='Bernoulli',
        subsample=.8, random_seed=SEED, thread_count=2, verbose=False,
        allow_writing_files=False,
    )
    model.fit(X[usable], target[usable].astype(int))
    stats['fallback'] = False
    return model.predict_proba(X[np.asarray(query, dtype=bool)])[:, 1], stats


def fit_factorized_survival(X, y3, y20, train, query):
    train = np.asarray(train, dtype=bool)
    p3, short_stats = _fit_stage(X, y3, train, query)
    continuation = train & np.isfinite(y3) & (np.asarray(y3) == 1)
    p20_given3, long_stats = _fit_stage(X, y20, continuation, query)
    return factorized_score(p3, p20_given3), p3, p20_given3, {
        'short_n_train': short_stats['n_train'],
        'short_fallback': short_stats['fallback'],
        'long_n_train': long_stats['n_train'],
        'long_fallback': long_stats['fallback'],
    }
