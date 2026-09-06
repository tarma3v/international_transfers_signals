"""Mature-only local Brier Hedge for AP36."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


EXPERTS = ('ap26_y20', 'ridge_survival', 'distributional_cat')


def mature_brier_hedge(experts, y20, mature20, dates, currencies, eligible,
                       window_days=730, prior_strength=40., eta=25.):
    if tuple(experts) != EXPERTS:
        raise ValueError(f'Expected experts in order {EXPERTS}')
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    y20 = np.asarray(y20, dtype=float)
    mature20 = np.asarray(mature20, dtype=object)
    ranks = np.column_stack([
        causal_rank_percentile(experts[name], currencies) for name in EXPERTS
    ])
    weights = np.full_like(ranks, np.nan)
    local_loss = np.full_like(ranks, np.nan)
    local_count = np.zeros_like(ranks, dtype=np.int32)
    global_count = np.zeros_like(ranks, dtype=np.int32)
    score = np.full(len(dates), np.nan)
    for day in sorted(set(dates)):
        query_day = dates == day
        cutoff = day - dt.timedelta(days=2)
        start = day - dt.timedelta(days=window_days)
        for currency in CORRIDORS:
            query = query_day & (currencies == currency)
            if not query.any():
                continue
            losses = []
            for j in range(len(EXPERTS)):
                base = (eligible & np.isfinite(ranks[:, j]) & np.isfinite(y20)
                        & (dates < day) & (dates >= start) & (mature20 < cutoff))
                global_n = int(base.sum())
                global_value = ((float(np.square(ranks[base, j] - y20[base]).sum())
                                 + prior_strength * .25)
                                / (global_n + prior_strength))
                local = base & (currencies == currency)
                local_n = int(local.sum())
                value = ((float(np.square(ranks[local, j] - y20[local]).sum())
                          + prior_strength * global_value)
                         / (local_n + prior_strength))
                losses.append(value)
                local_loss[query, j] = value
                local_count[query, j] = local_n
                global_count[query, j] = global_n
            raw_weight = np.exp(-eta * (np.asarray(losses) - min(losses)))
            raw_weight /= raw_weight.sum()
            weights[query] = raw_weight
            for i in np.flatnonzero(query):
                finite = np.isfinite(ranks[i])
                if finite.any():
                    normalized = raw_weight[finite] / raw_weight[finite].sum()
                    score[i] = float(np.dot(normalized, ranks[i, finite]))
    return score, ranks, weights, local_loss, local_count, global_count
