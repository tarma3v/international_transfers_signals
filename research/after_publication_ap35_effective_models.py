"""Monotone distributional residual CatBoost for AP35."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

from research.after_publication_ap1 import SEED


ANCHOR_OFFSETS = np.array([-200., -100., 0., 100., 200.])


def fit_distributional_cat(X, target, known_change, train, query, dates, origin):
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)
    known_change = np.asarray(known_change, dtype=float)
    train = np.asarray(train, dtype=bool) & np.isfinite(target)
    query = np.asarray(query, dtype=bool)
    dates = np.asarray(dates)
    train_ids = np.flatnonzero(train)
    query_ids = np.flatnonzero(query)
    if len(train_ids) < 100:
        return np.repeat(.5, len(query_ids)), {
            'n_train': int(len(train_ids)),
            'n_augmented': int(len(train_ids) * len(ANCHOR_OFFSETS)),
            'positive_share': None,
            'weight_sum': None,
            'fallback': True,
            'monotonic_min_delta': 0.,
        }

    context = np.repeat(X[train_ids], len(ANCHOR_OFFSETS), axis=0)
    anchor = (np.repeat(known_change[train_ids], len(ANCHOR_OFFSETS))
              + np.tile(ANCHOR_OFFSETS, len(train_ids)))
    residual = np.repeat(target[train_ids], len(ANCHOR_OFFSETS))
    label = (residual >= -anchor).astype(int)
    expanded = np.column_stack([context, anchor])
    age = np.array([(origin - dates[i]).days for i in train_ids], dtype=float)
    if np.any(age <= 0):
        raise ValueError('Distributional fit may only use dates before origin')
    weight = np.repeat(np.power(.5, age / 730.), len(ANCHOR_OFFSETS))
    model = CatBoostClassifier(
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
        loss_function='Logloss',
        monotone_constraints={X.shape[1]: 1},
    )
    if np.unique(label).size < 2:
        value = float(label.mean())
        prediction = np.repeat(value, len(query_ids))
        fallback = True
        monotonic_min_delta = 0.
    else:
        model.fit(expanded, label, sample_weight=weight)
        query_X = np.column_stack([X[query_ids], known_change[query_ids]])
        prediction = model.predict_proba(query_X)[:, 1]
        fallback = False
        probe_ids = query_ids[:min(20, len(query_ids))]
        probe_anchor = np.array([-400., -200., 0., 200., 400.])
        probe = np.column_stack([
            np.repeat(X[probe_ids], len(probe_anchor), axis=0),
            np.tile(probe_anchor, len(probe_ids)),
        ])
        curve = model.predict_proba(probe)[:, 1].reshape(
            len(probe_ids), len(probe_anchor))
        monotonic_min_delta = float(np.min(np.diff(curve, axis=1)))
    return np.clip(prediction, 0, 1), {
        'n_train': int(len(train_ids)),
        'n_augmented': int(len(label)),
        'positive_share': float(label.mean()),
        'weight_sum': float(weight.sum()),
        'fallback': fallback,
        'monotonic_min_delta': monotonic_min_delta,
    }
