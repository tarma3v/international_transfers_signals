"""Dual-score causal pacing with AP17 thresholds and a monthly rescue."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


PAIRINGS = (
    'roll_ap18',
    'roll_cat',
    'roll_ap12',
    'local_ap18',
    'local_cat',
    'roll75local25_ap1850cat50',
)


def dual_paced_month_policy(primary_score, pace_score, reserve_score, dates,
                            currencies, eligible, window=250, warmup=40):
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    pace_rank = causal_rank_percentile(pace_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    signal = np.zeros(len(primary_score), dtype=bool)
    trailing_rate = np.full(len(primary_score), np.nan)
    reason = np.zeros(len(primary_score), dtype=np.int8)
    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        first_day = dates[indices[0]]
        week, used, month, month_used = None, 0, None, 0
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
            trailing_rate[i] = rate
            primary = np.isfinite(primary_rank[i]) and primary_rank[i] > .70
            pace = ((dates[i] - first_day).days >= 84 and rate < 1.
                    and np.isfinite(pace_rank[i]) and pace_rank[i] > .55
                    and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            month_rescue = (dates[i].day >= 24 and month_used == 0
                            and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            if eligible[i] and used < 2 and (primary or pace or month_rescue):
                signal[i] = True
                reason[i] = 1 if primary else (2 if pace else 3)
                chosen_days.append(dates[i])
                used += 1
                month_used += 1
    return signal, primary_rank, pace_rank, reserve_rank, trailing_rate, reason


def pairing_scores(rolling, local, ap12, ap18, cat):
    return {
        'roll_ap18': (rolling, ap18),
        'roll_cat': (rolling, cat),
        'roll_ap12': (rolling, ap12),
        'local_ap18': (local, ap18),
        'local_cat': (local, cat),
        'roll75local25_ap1850cat50': (
            .75 * rolling + .25 * local,
            .50 * ap18 + .50 * cat,
        ),
    }
