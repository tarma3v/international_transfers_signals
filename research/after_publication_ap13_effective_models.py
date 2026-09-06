"""AP13 recent-window ExtraTrees, causal expert routers and reserve policies."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap1 import factory
from research.after_publication_ap12_effective_models import (
    causal_rank_percentile,
    extra_factory,
)


MODEL_SCORES = (
    'extra_roll2',
    'extra_roll3',
    'extra_decay730',
    'local_extra',
    'router_extra_local',
    'brier365_extra_local',
)
POLICY_KINDS = ('primary', 'reserve7', 'month24')


def fit_extra(X, y, train, query, sample_weight=None):
    usable = train & np.isfinite(y)
    if usable.sum() < 100 or np.unique(y[usable]).size < 2:
        return np.repeat(float(np.nanmean(y[usable])) if usable.any() else 0., query.sum())
    model = extra_factory()
    kwargs = {} if sample_weight is None else {'sample_weight': sample_weight[usable]}
    model.fit(X[usable], y[usable], **kwargs)
    return model.predict_proba(X[query])[:, 1]


def fit_local_extra(X, y, train, query, currencies, global_prediction, shrink=150):
    result = np.asarray(global_prediction, dtype=float).copy()
    query_ids = np.flatnonzero(query)
    counts = {}
    for currency in CORRIDORS:
        usable = train & (currencies == currency) & np.isfinite(y)
        local_query = currencies[query] == currency
        counts[currency] = int(usable.sum())
        if usable.sum() < 100 or np.unique(y[usable]).size < 2 or not local_query.any():
            continue
        model = extra_factory()
        model.fit(X[usable], y[usable])
        local = model.predict_proba(X[query_ids[local_query]])[:, 1]
        weight = usable.sum() / (usable.sum() + shrink)
        result[local_query] = weight * local + (1 - weight) * result[local_query]
    return result, counts


def fit_meta_router(X, y, train, query, extra, local, minimum=200):
    """Blend frozen OOS experts using a mature-only learned preference."""
    finite = np.isfinite(extra) & np.isfinite(local) & np.isfinite(y)
    usable = train & finite
    prefer_extra = (extra - y) ** 2 < (local - y) ** 2
    if usable.sum() < minimum or np.unique(prefer_extra[usable]).size < 2:
        weight = np.repeat(.5, query.sum())
    else:
        model = factory('hist7y').set_params(early_stopping=False)
        model.fit(X[usable], prefer_extra[usable].astype(int))
        weight = model.predict_proba(X[query])[:, 1]
    score = weight * extra[query] + (1 - weight) * local[query]
    return score, weight, int(usable.sum()), int(prefer_extra[usable].sum())


def brier365_router(dates, mature5, y, eligible, extra, local, minimum=100, eta=20.):
    """Same-date global weights from outcomes resolved strictly beforehand."""
    dates = np.asarray(dates)
    result = np.full(len(dates), np.nan)
    weights = np.full((len(dates), 2), np.nan)
    counts = np.zeros(len(dates), dtype=int)
    finite = np.isfinite(y) & np.isfinite(extra) & np.isfinite(local) & eligible
    for day in sorted(set(dates)):
        query = dates == day
        start = day - dt.timedelta(days=365)
        past = finite & (mature5 < day) & (dates >= start) & (dates < day)
        n = int(past.sum())
        if n < minimum:
            weight = np.array([.5, .5])
        else:
            loss = np.array([
                np.mean((extra[past] - y[past]) ** 2),
                np.mean((local[past] - y[past]) ** 2),
            ])
            raw = np.exp(-eta * (loss - loss.min()))
            weight = raw / raw.sum()
        result[query] = weight[0] * extra[query] + weight[1] * local[query]
        weights[query] = weight
        counts[query] = n
    return result, weights, counts


def reserve_policy(primary_score, reserve_score, dates, currencies, eligible, kind,
                   window=250, warmup=40):
    """Primary top30 plus a causal silence/month rescue under the same cap2."""
    if kind not in POLICY_KINDS:
        raise ValueError(kind)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    result = np.zeros(len(primary_score), dtype=bool)
    for currency in CORRIDORS:
        week, used, last, month, month_used = None, 0, None, None, 0
        for i in np.flatnonzero(currencies == currency):
            iso = dates[i].isocalendar()[:2]
            current_month = (dates[i].year, dates[i].month)
            if iso != week:
                week, used = iso, 0
            if current_month != month:
                month, month_used = current_month, 0
            primary = np.isfinite(primary_rank[i]) and primary_rank[i] > .70
            fallback = False
            if kind == 'reserve7':
                age = (dates[i] - last).days if last is not None else 999
                fallback = age >= 7 and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .65
            elif kind == 'month24':
                fallback = (dates[i].day >= 24 and month_used == 0
                            and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .50)
            if eligible[i] and used < 2 and (primary or fallback):
                result[i] = True
                used += 1
                month_used += 1
                last = dates[i]
    return result
