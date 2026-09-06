"""Sparse causal backstop state machine for AP27."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


POLICIES = {
    's200_cat95_r70': ('s200', .95, .70, None),
    's200_cat100_r70': ('s200', 1.00, .70, None),
    's200_cat95_r60': ('s200', .95, .60, None),
    's100_cat95_r70': ('s100', .95, .70, None),
    's200_cat_silence14_r70': ('s200', None, .70, 14),
}


def triple_paced_month_policy(primary_score, pace_score, backstop_score,
                              reserve_score, dates, currencies, eligible,
                              backstop_rate, backstop_threshold,
                              silence_days=None, window=250, warmup=40):
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    pace_rank = causal_rank_percentile(pace_score, currencies, window, warmup)
    backstop_rank = causal_rank_percentile(backstop_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    signal = np.zeros(len(primary_score), dtype=bool)
    trailing_rate = np.full(len(primary_score), np.nan)
    days_since = np.full(len(primary_score), np.nan)
    reason = np.zeros(len(primary_score), dtype=np.int8)
    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        first_day = dates[indices[0]]
        week, used, month, month_used, last = None, 0, None, 0, None
        chosen_days = []
        for i in indices:
            iso = dates[i].isocalendar()[:2]
            current_month = (dates[i].year, dates[i].month)
            if iso != week:
                week, used = iso, 0
            if current_month != month:
                month, month_used = current_month, 0
            start = max(first_day, dates[i] - dt.timedelta(days=365))
            weeks = max((dates[i] - start).days / 7., 1.)
            rate = sum(start <= selected < dates[i] for selected in chosen_days) / weeks
            age = 999 if last is None else (dates[i] - last).days
            trailing_rate[i] = rate
            days_since[i] = age
            primary = np.isfinite(primary_rank[i]) and primary_rank[i] > .70
            pace = ((dates[i] - first_day).days >= 84 and rate < 1.
                    and np.isfinite(pace_rank[i]) and pace_rank[i] > .55
                    and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            if silence_days is None:
                deficit = rate < backstop_rate
            else:
                deficit = age >= silence_days
            backstop = ((dates[i] - first_day).days >= 84 and not pace and deficit
                        and np.isfinite(backstop_rank[i])
                        and backstop_rank[i] > backstop_threshold
                        and np.isfinite(reserve_rank[i])
                        and reserve_rank[i] > .70)
            month_rescue = (dates[i].day >= 24 and month_used == 0
                            and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            if eligible[i] and used < 2 and (primary or pace or backstop or month_rescue):
                signal[i] = True
                reason[i] = 1 if primary else (2 if pace else (3 if backstop else 4))
                chosen_days.append(dates[i])
                used += 1
                month_used += 1
                last = dates[i]
    return (signal, primary_rank, pace_rank, backstop_rank, reserve_rank,
            trailing_rate, days_since, reason)
