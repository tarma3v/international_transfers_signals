"""Causal cold-start mixtures for AP26 pace specialists."""
from __future__ import annotations

import numpy as np

from research.after_publication_ap12_effective_models import causal_rank_percentile


SCORES = (
    'y20_hard100',
    'y20_shrink100',
    'y20_shrink200',
    'mean_shrink100',
    'future5_rank_shrink100',
)


def _weight(counts, strength):
    counts = np.maximum(np.asarray(counts, dtype=float), 0.)
    return counts / (counts + strength)


def _blend(specialist, global_score, weight):
    specialist = np.asarray(specialist, dtype=float)
    global_score = np.asarray(global_score, dtype=float)
    result = np.full(len(specialist), np.nan)
    both = np.isfinite(specialist) & np.isfinite(global_score)
    result[both] = (weight[both] * specialist[both]
                    + (1 - weight[both]) * global_score[both])
    result[np.isfinite(specialist) & ~np.isfinite(global_score)] = specialist[
        np.isfinite(specialist) & ~np.isfinite(global_score)]
    result[np.isfinite(global_score) & ~np.isfinite(specialist)] = global_score[
        np.isfinite(global_score) & ~np.isfinite(specialist)]
    return result


def cold_start_scores(cat, y20, mean, future5, y20_count, mean_count,
                      future_count, currencies):
    cat = np.asarray(cat, dtype=float)
    y20 = np.asarray(y20, dtype=float)
    hard_weight = (np.asarray(y20_count) >= 100).astype(float)
    w_y100 = _weight(y20_count, 100.)
    w_y200 = _weight(y20_count, 200.)
    w_mean100 = _weight(mean_count, 100.)
    w_future100 = _weight(future_count, 100.)
    cat_rank = causal_rank_percentile(cat, currencies)
    future_rank = causal_rank_percentile(future5, currencies)
    scores = {
        'y20_hard100': _blend(y20, cat, hard_weight),
        'y20_shrink100': _blend(y20, cat, w_y100),
        'y20_shrink200': _blend(y20, cat, w_y200),
        'mean_shrink100': _blend(mean, cat, w_mean100),
        'future5_rank_shrink100': _blend(
            future_rank, cat_rank, w_future100),
    }
    weights = {
        'y20_hard100': hard_weight,
        'y20_shrink100': w_y100,
        'y20_shrink200': w_y200,
        'mean_shrink100': w_mean100,
        'future5_rank_shrink100': w_future100,
    }
    return scores, weights, cat_rank, future_rank
