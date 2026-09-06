"""Mature-only online expert competence and fixed AP23 pairings."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


PAIRINGS = (
    'soft730_both',
    'soft365_both',
    'hard730_both',
    'soft730_primary',
    'soft730_pace',
)
EXPERTS = ('rolling', 'local', 'ap18', 'cat')


def unknown_utility(outcomes):
    """Mean survival across unknown h3/h5/h10/h20; h1 is excluded."""
    matrix = np.column_stack([outcomes[f'y{h}'] for h in (3, 5, 10, 20)])
    result = np.full(len(matrix), np.nan)
    complete = np.isfinite(matrix).all(axis=1)
    result[complete] = matrix[complete].mean(axis=1)
    return result


def causal_top_precision(rank, utility, mature20, dates, currencies, eligible,
                         window_days, prior_strength=40.):
    """Per-currency precision using only top-rank labels mature before today."""
    rank = np.asarray(rank, dtype=float)
    utility = np.asarray(utility, dtype=float)
    mature20 = np.asarray(mature20, dtype=object)
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    precision = np.full(len(rank), np.nan)
    local_count = np.zeros(len(rank), dtype=np.int32)
    global_count = np.zeros(len(rank), dtype=np.int32)
    base = (eligible & np.isfinite(rank) & (rank > .70)
            & np.isfinite(utility))
    for day in sorted(set(dates)):
        query = dates == day
        cutoff = day - dt.timedelta(days=2)
        start = day - dt.timedelta(days=window_days)
        past = base & (dates < day) & (dates >= start) & (mature20 < cutoff)
        n_global = int(past.sum())
        global_precision = ((float(utility[past].sum()) + prior_strength * .5)
                            / (n_global + prior_strength))
        for currency in CORRIDORS:
            target = query & (currencies == currency)
            if not target.any():
                continue
            local = past & (currencies == currency)
            n_local = int(local.sum())
            value = ((float(utility[local].sum())
                      + prior_strength * global_precision)
                     / (n_local + prior_strength))
            precision[target] = value
            local_count[target] = n_local
            global_count[target] = n_global
    return precision, local_count, global_count


def _routed_score(left, right, left_competence, right_competence, hard=False,
                  eta=25.):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    difference = np.asarray(left_competence) - np.asarray(right_competence)
    if hard:
        weight = (difference >= 0).astype(float)
    else:
        clipped = np.clip(eta * difference, -50, 50)
        weight = 1. / (1. + np.exp(-clipped))
    result = np.full(len(left), np.nan)
    both = np.isfinite(left) & np.isfinite(right)
    result[both] = weight[both] * left[both] + (1 - weight[both]) * right[both]
    result[np.isfinite(left) & ~np.isfinite(right)] = left[
        np.isfinite(left) & ~np.isfinite(right)]
    result[np.isfinite(right) & ~np.isfinite(left)] = right[
        np.isfinite(right) & ~np.isfinite(left)]
    return result, weight


def competence_pairings(experts, outcomes, mature20, dates, currencies, eligible):
    """Return five pre-registered pairings and full causal diagnostics."""
    if tuple(experts) != EXPERTS:
        raise ValueError(f'Expected experts in order {EXPERTS}')
    ranks = {name: causal_rank_percentile(experts[name], currencies)
             for name in EXPERTS}
    utility = unknown_utility(outcomes)
    competence, counts = {}, {}
    for window in (365, 730):
        for name in EXPERTS:
            values = causal_top_precision(
                ranks[name], utility, mature20, dates, currencies, eligible, window)
            competence[(window, name)] = values[0]
            counts[(window, name, 'local')] = values[1]
            counts[(window, name, 'global')] = values[2]

    roll_local_soft730, w_primary730 = _routed_score(
        ranks['rolling'], ranks['local'], competence[(730, 'rolling')],
        competence[(730, 'local')])
    ap18_cat_soft730, w_pace730 = _routed_score(
        ranks['ap18'], ranks['cat'], competence[(730, 'ap18')],
        competence[(730, 'cat')])
    roll_local_soft365, w_primary365 = _routed_score(
        ranks['rolling'], ranks['local'], competence[(365, 'rolling')],
        competence[(365, 'local')])
    ap18_cat_soft365, w_pace365 = _routed_score(
        ranks['ap18'], ranks['cat'], competence[(365, 'ap18')],
        competence[(365, 'cat')])
    roll_local_hard730, w_primary_hard = _routed_score(
        ranks['rolling'], ranks['local'], competence[(730, 'rolling')],
        competence[(730, 'local')], hard=True)
    ap18_cat_hard730, w_pace_hard = _routed_score(
        ranks['ap18'], ranks['cat'], competence[(730, 'ap18')],
        competence[(730, 'cat')], hard=True)

    pairs = {
        'soft730_both': (roll_local_soft730, ap18_cat_soft730),
        'soft365_both': (roll_local_soft365, ap18_cat_soft365),
        'hard730_both': (roll_local_hard730, ap18_cat_hard730),
        'soft730_primary': (roll_local_soft730, experts['cat']),
        'soft730_pace': (experts['rolling'], ap18_cat_soft730),
    }
    weights = {
        'primary_soft730': w_primary730,
        'pace_soft730': w_pace730,
        'primary_soft365': w_primary365,
        'pace_soft365': w_pace365,
        'primary_hard730': w_primary_hard,
        'pace_hard730': w_pace_hard,
    }
    return pairs, ranks, competence, counts, weights, utility
