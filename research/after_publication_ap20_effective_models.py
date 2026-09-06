"""Fixed raw and causal-rank AP18/CatBoost expert geometries."""
from __future__ import annotations

import numpy as np

from research.after_publication_ap12_effective_models import causal_rank_percentile


SCORE_NAMES = (
    'raw_ap18_75_cat25',
    'raw_ap18_50_cat50',
    'raw_ap18_25_cat75',
    'rank_ap18_75_cat25',
    'rank_ap18_50_cat50',
    'rank_ap18_25_cat75',
    'rank_agreement_min',
    'rank_agreement_geom',
)


def _mix(left, right, left_weight):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    result = np.full(len(left), np.nan)
    both = np.isfinite(left) & np.isfinite(right)
    result[both] = left_weight * left[both] + (1 - left_weight) * right[both]
    result[np.isfinite(left) & ~np.isfinite(right)] = left[
        np.isfinite(left) & ~np.isfinite(right)]
    result[np.isfinite(right) & ~np.isfinite(left)] = right[
        np.isfinite(right) & ~np.isfinite(left)]
    return result


def expert_scores(ap18, cat_utility, currencies):
    ap18 = np.asarray(ap18, dtype=float)
    cat_utility = np.asarray(cat_utility, dtype=float)
    left_rank = causal_rank_percentile(ap18, currencies)
    right_rank = causal_rank_percentile(cat_utility, currencies)
    minimum = np.full(len(ap18), np.nan)
    geometric = np.full(len(ap18), np.nan)
    both = np.isfinite(left_rank) & np.isfinite(right_rank)
    minimum[both] = np.minimum(left_rank[both], right_rank[both])
    geometric[both] = np.sqrt(left_rank[both] * right_rank[both])
    only_left = np.isfinite(left_rank) & ~np.isfinite(right_rank)
    only_right = np.isfinite(right_rank) & ~np.isfinite(left_rank)
    minimum[only_left] = geometric[only_left] = left_rank[only_left]
    minimum[only_right] = geometric[only_right] = right_rank[only_right]
    return {
        'raw_ap18_75_cat25': _mix(ap18, cat_utility, .75),
        'raw_ap18_50_cat50': _mix(ap18, cat_utility, .50),
        'raw_ap18_25_cat75': _mix(ap18, cat_utility, .25),
        'rank_ap18_75_cat25': _mix(left_rank, right_rank, .75),
        'rank_ap18_50_cat50': _mix(left_rank, right_rank, .50),
        'rank_ap18_25_cat75': _mix(left_rank, right_rank, .25),
        'rank_agreement_min': minimum,
        'rank_agreement_geom': geometric,
    }, left_rank, right_rank
