"""AP16 causal per-currency deficit-pacing controllers."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


POLICY_KINDS = (
    'pace365_p60',
    'pace365_p55_r70',
    'paceall_p60',
    'pace365_p60_month24',
    'adaptive105_month24',
)


def _past_rate(day, first_day, chosen_days, window_days=None):
    start = first_day if window_days is None else max(
        first_day, day - dt.timedelta(days=window_days))
    weeks = max((day - start).days / 7., 1.)
    return sum(start <= selected < day for selected in chosen_days) / weeks


def paced_policy(primary_score, reserve_score, dates, currencies, eligible, kind,
                 window=250, warmup=40):
    """Return signal plus prior-only ranks, rate, threshold and route reason."""
    if kind not in POLICY_KINDS:
        raise ValueError(kind)
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    signal = np.zeros(len(primary_score), dtype=bool)
    threshold = np.full(len(primary_score), .70)
    trailing_rate = np.full(len(primary_score), np.nan)
    reason = np.zeros(len(primary_score), dtype=np.int8)  # 1 primary,2 pace,3 month

    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        first_day = dates[indices[0]]
        week, used = None, 0
        month, month_used = None, 0
        chosen_days = []
        for i in indices:
            iso = dates[i].isocalendar()[:2]
            current_month = (dates[i].year, dates[i].month)
            if iso != week:
                week, used = iso, 0
            if current_month != month:
                month, month_used = current_month, 0
            age = (dates[i] - first_day).days
            high_primary = np.isfinite(primary_rank[i]) and primary_rank[i] > .70
            pace = False
            if kind == 'adaptive105_month24':
                rate = _past_rate(dates[i], first_day, chosen_days, 182)
                trailing_rate[i] = rate
                if age < 56:
                    cut = .675
                elif rate < .95:
                    cut = .65
                elif rate < 1.05:
                    cut = .675
                else:
                    cut = .70
                threshold[i] = cut
                pace = (not high_primary and np.isfinite(primary_rank[i])
                        and primary_rank[i] > cut)
            else:
                use_window = None if kind == 'paceall_p60' else 365
                rate = _past_rate(dates[i], first_day, chosen_days, use_window)
                trailing_rate[i] = rate
                if age >= 84 and rate < 1.0:
                    cut = .55 if kind == 'pace365_p55_r70' else .60
                    threshold[i] = cut
                    pace = np.isfinite(primary_rank[i]) and primary_rank[i] > cut
                    if kind == 'pace365_p55_r70':
                        pace &= np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70
                    pace &= not high_primary
            use_month = kind.endswith('month24')
            month_rescue = (use_month and dates[i].day >= 24 and month_used == 0
                            and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            if eligible[i] and used < 2 and (high_primary or pace or month_rescue):
                signal[i] = True
                reason[i] = 1 if high_primary else (2 if pace else 3)
                chosen_days.append(dates[i])
                used += 1
                month_used += 1
    return signal, primary_rank, reserve_rank, trailing_rate, threshold, reason
