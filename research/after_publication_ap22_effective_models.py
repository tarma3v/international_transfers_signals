"""Causal rolling/local primary consensus geometries for AP22."""
from __future__ import annotations

import numpy as np

from research.after_publication_ap12_effective_models import causal_rank_percentile


PAIRINGS = (
    'rank75_cat', 'rank50_cat', 'rank25_cat', 'rankmin_cat', 'rankgeom_cat',
    'rank75_ap12', 'rank50_ap12', 'rankmin_ap12',
)


def _mix(left, right, weight):
    result = np.full(len(left), np.nan)
    both = np.isfinite(left) & np.isfinite(right)
    result[both] = weight * left[both] + (1 - weight) * right[both]
    only_left = np.isfinite(left) & ~np.isfinite(right)
    only_right = np.isfinite(right) & ~np.isfinite(left)
    result[only_left] = left[only_left]
    result[only_right] = right[only_right]
    return result


def consensus_pairings(rolling, local, cat, ap12, currencies):
    rolling_rank = causal_rank_percentile(rolling, currencies)
    local_rank = causal_rank_percentile(local, currencies)
    rank75 = _mix(rolling_rank, local_rank, .75)
    rank50 = _mix(rolling_rank, local_rank, .50)
    rank25 = _mix(rolling_rank, local_rank, .25)
    rankmin = np.fmin(rolling_rank, local_rank)
    rankgeom = np.sqrt(np.clip(rolling_rank, 0, 1) * np.clip(local_rank, 0, 1))
    pairs = {
        'rank75_cat': (rank75, cat),
        'rank50_cat': (rank50, cat),
        'rank25_cat': (rank25, cat),
        'rankmin_cat': (rankmin, cat),
        'rankgeom_cat': (rankgeom, cat),
        'rank75_ap12': (rank75, ap12),
        'rank50_ap12': (rank50, ap12),
        'rankmin_ap12': (rankmin, ap12),
    }
    return pairs, rolling_rank, local_rank
