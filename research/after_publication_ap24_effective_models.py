"""Fixed grouped CatBoost ranking models for AP24."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRanker

from research.after_publication_ap1 import SEED


SPECS = {
    'rank_qtr_mean_full_pair': ('quarter', 'mean', None, 'PairLogit'),
    'rank_qtr_mean_roll3_pair': ('quarter', 'mean', 1095, 'PairLogit'),
    'rank_month_mean_full_pair': ('month', 'mean', None, 'PairLogit'),
    'rank_qtr_y20_full_pair': ('quarter', 'y20', None, 'PairLogit'),
    'rank_qtr_mean_full_yeti': ('quarter', 'mean', None, 'YetiRankPairwise'),
}


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


def ranking_target(labels, kind):
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError('Expected h3/h5/h10/h20 label matrix')
    if kind == 'mean':
        return labels.mean(axis=1)
    if kind == 'y20':
        return labels[:, 3]
    raise ValueError(kind)


def group_keys(dates, currencies, kind):
    if kind not in ('month', 'quarter'):
        raise ValueError(kind)
    result = []
    for day, currency in zip(dates, currencies):
        period = day.month if kind == 'month' else (day.month - 1) // 3 + 1
        result.append(f'{currency}-{day.year:04d}-{period:02d}')
    return np.asarray(result)


def grouped_training_rows(train, target, dates, currencies, group_kind):
    usable = np.asarray(train, dtype=bool) & np.isfinite(target)
    indices = np.flatnonzero(usable)
    keys = group_keys(np.asarray(dates)[indices], np.asarray(currencies)[indices],
                      group_kind)
    order = np.argsort(keys, kind='stable')
    indices, keys = indices[order], keys[order]
    keep = np.zeros(len(indices), dtype=bool)
    groups_kept = 0
    groups_total = 0
    for key in np.unique(keys):
        member = keys == key
        groups_total += 1
        if member.sum() >= 2 and np.unique(target[indices[member]]).size >= 2:
            keep[member] = True
            groups_kept += 1
    indices = indices[keep]
    keys = keys[keep]
    _, group_id = np.unique(keys, return_inverse=True)
    return indices, group_id.astype(np.int64), {
        'n_groups_total': groups_total,
        'n_groups_kept': groups_kept,
        'n_group_rows': int(len(indices)),
    }


def fit_ranker(X, labels, train, query, dates, currencies, spec):
    group_kind, target_kind, rolling_days, loss = spec
    target = ranking_target(labels, target_kind)
    use = np.asarray(train, dtype=bool).copy()
    if rolling_days is not None:
        origin = min(np.asarray(dates)[query])
        use &= np.asarray(dates) >= origin.__class__.fromordinal(
            origin.toordinal() - rolling_days)
    indices, group_id, stats = grouped_training_rows(
        use, target, dates, currencies, group_kind)
    stats.update({'target_kind': target_kind, 'group_kind': group_kind,
                  'rolling_days': rolling_days, 'loss': loss})
    if len(indices) < 100 or len(np.unique(group_id)) < 5:
        value = float(np.nanmean(target[use])) if use.any() else 0.
        stats['fallback'] = True
        return np.repeat(value, int(np.sum(query))), stats
    model = CatBoostRanker(loss_function=loss, **_cat_common())
    model.fit(X[indices], target[indices], group_id=group_id)
    raw = np.clip(model.predict(X[query]), -30, 30)
    score = 1. / (1. + np.exp(-raw))
    stats['fallback'] = False
    return score, stats
