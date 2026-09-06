"""Hierarchically pooled low-data pace expert for AP28."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

from research.after_publication_ap1 import SEED


SCORE = 'hier_y20_weight4'
HARD_WEIGHT = 4.


def hierarchical_design(X, rolling_rank, reserve_rank, hard_pool):
    """Append only outcome-free regime coordinates to the shared feature set."""
    return np.column_stack([
        np.asarray(X, dtype=float),
        np.asarray(rolling_rank, dtype=float),
        np.asarray(reserve_rank, dtype=float),
        np.asarray(hard_pool, dtype=float),
    ])


def hierarchical_weights(train, hard_pool):
    train = np.asarray(train, dtype=bool)
    hard_pool = np.asarray(hard_pool, dtype=bool)
    weights = np.zeros(len(train), dtype=float)
    weights[train] = 1.
    weights[train & hard_pool] = HARD_WEIGHT
    return weights


def fit_hierarchical_y20(X, target, train, query, rolling_rank, reserve_rank,
                         hard_pool):
    design = hierarchical_design(X, rolling_rank, reserve_rank, hard_pool)
    target = np.asarray(target, dtype=float)
    usable = np.asarray(train, dtype=bool) & np.isfinite(target)
    weights = hierarchical_weights(usable, hard_pool)
    stats = {
        'n_train': int(usable.sum()),
        'n_hard_train': int((usable & np.asarray(hard_pool, dtype=bool)).sum()),
        'weight_sum': float(weights.sum()),
        'hard_weight': HARD_WEIGHT,
    }
    if usable.sum() < 100 or np.unique(target[usable]).size < 2:
        value = (float(np.average(target[usable], weights=weights[usable]))
                 if usable.any() else 0.)
        stats['fallback'] = True
        return np.repeat(value, int(np.sum(query))), stats
    model = CatBoostClassifier(
        loss_function='Logloss', iterations=400, depth=6, learning_rate=.035,
        l2_leaf_reg=12., random_strength=.5, bootstrap_type='Bernoulli',
        subsample=.8, random_seed=SEED, thread_count=2, verbose=False,
        allow_writing_files=False,
    )
    model.fit(design[usable], target[usable].astype(int),
              sample_weight=weights[usable])
    stats['fallback'] = False
    return model.predict_proba(design[np.asarray(query, dtype=bool)])[:, 1], stats
