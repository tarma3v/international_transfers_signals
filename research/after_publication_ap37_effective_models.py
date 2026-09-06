"""Mature disagreement-tail precision gate and calendar router for AP37."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS


EXPERTS = ('ap26_y20', 'ridge_survival', 'distributional_cat')


def mature_pool_support_precision(expert_ranks, outcomes, mature20, dates,
                                  currencies, eligible, pool_mask,
                                  prior_strength=40.):
    if tuple(expert_ranks) != EXPERTS:
        raise ValueError(f'Expected experts in order {EXPERTS}')
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    pool_mask = np.asarray(pool_mask, dtype=bool)
    ranks = np.column_stack([np.asarray(expert_ranks[name], dtype=float)
                             for name in EXPERTS])
    finite_ranks = np.isfinite(ranks).all(axis=1)
    support = np.full(len(dates), -1, dtype=np.int8)
    support[finite_ranks] = (ranks[finite_ranks] >= .70).sum(axis=1)
    ys = np.column_stack([np.asarray(outcomes['y' + str(h)], dtype=float)
                          for h in (3, 5, 10, 20)])
    valid_y = np.isfinite(ys).all(axis=1)
    all_success = np.full(len(dates), np.nan)
    all_success[valid_y] = ys[valid_y].min(axis=1)
    mature20 = np.asarray(mature20, dtype=object)

    quality = np.zeros(len(dates), dtype=bool)
    overall_precision = np.full(len(dates), np.nan)
    global_stratum_precision = np.full(len(dates), np.nan)
    local_stratum_precision = np.full(len(dates), np.nan)
    overall_count = np.zeros(len(dates), dtype=np.int32)
    global_stratum_count = np.zeros(len(dates), dtype=np.int32)
    local_stratum_count = np.zeros(len(dates), dtype=np.int32)

    base_pool = eligible & pool_mask & finite_ranks & valid_y
    for day in sorted(set(dates)):
        query_day = dates == day
        cutoff = day - dt.timedelta(days=2)
        history = base_pool & (dates < day) & (mature20 < cutoff)
        n_all = int(history.sum())
        p_all = ((float(np.nansum(all_success[history])) + prior_strength * .5)
                 / (n_all + prior_strength))
        for currency in CORRIDORS:
            query = query_day & (currencies == currency)
            if not query.any():
                continue
            for i in np.flatnonzero(query):
                overall_precision[i] = p_all
                overall_count[i] = n_all
                if support[i] < 0:
                    continue
                stratum = history & (support == support[i])
                n_global = int(stratum.sum())
                p_global = ((float(np.nansum(all_success[stratum]))
                             + prior_strength * p_all)
                            / (n_global + prior_strength))
                local = stratum & (currencies == currency)
                n_local = int(local.sum())
                p_local = ((float(np.nansum(all_success[local]))
                            + prior_strength * p_global)
                           / (n_local + prior_strength))
                global_stratum_precision[i] = p_global
                local_stratum_precision[i] = p_local
                global_stratum_count[i] = n_global
                local_stratum_count[i] = n_local
                quality[i] = p_local >= p_all
    return (support, quality, overall_precision, global_stratum_precision,
            local_stratum_precision, overall_count, global_stratum_count,
            local_stratum_count)


def mature_support_precision(expert_ranks, outcomes, mature20, dates, currencies,
                             eligible, core, prior_strength=40.):
    return mature_pool_support_precision(
        expert_ranks, outcomes, mature20, dates, currencies, eligible,
        ~np.asarray(core, dtype=bool), prior_strength=prior_strength)


def precision_calendar_router(core, fallback, quality, dates, currencies,
                              eligible):
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    core = np.asarray(core, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
    quality = np.asarray(quality, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    signal = np.zeros(len(core), dtype=bool)
    trailing_rate = np.full(len(core), np.nan)
    days_since = np.full(len(core), np.nan)
    reason = np.zeros(len(core), dtype=np.int8)
    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        available = indices[eligible[indices]]
        first_eligible = dates[available[0]] if len(available) else dates[indices[0]]
        week, used = None, 0
        chosen_days = []
        for i in indices:
            iso = dates[i].isocalendar()[:2]
            if iso != week:
                week, used = iso, 0
            if dates[i] < first_eligible:
                continue
            start = max(first_eligible, dates[i] - dt.timedelta(days=365))
            weeks = max((dates[i] - start).days / 7., 1.)
            rate = sum(start <= chosen < dates[i]
                       for chosen in chosen_days) / weeks
            trailing_rate[i] = rate
            gap = ((dates[i] - chosen_days[-1]).days
                   if chosen_days else np.inf)
            days_since[i] = gap
            warmed = (dates[i] - first_eligible).days >= 84
            late_week = dates[i].isoweekday() >= 4 and used == 0
            silence = gap >= 10
            use_core = core[i]
            precision_late_week = late_week and quality[i]
            use_fallback = (not use_core and warmed and rate < 1.
                            and fallback[i]
                            and (silence or precision_late_week))
            if eligible[i] and used < 2 and (use_core or use_fallback):
                signal[i] = True
                if use_core:
                    reason[i] = 1
                elif precision_late_week:
                    reason[i] = 2
                else:
                    reason[i] = 3
                chosen_days.append(dates[i])
                used += 1
    return signal, trailing_rate, days_since, reason
