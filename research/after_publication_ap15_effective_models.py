"""AP15 narrow causal frequency and empty-month controllers."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


POLICY_KINDS = (
    'top3125',
    'silence14_month24',
    'top3125_month24',
    'adaptive100',
    'adaptive100_month24',
)


def _narrow_threshold(day, first_day, chosen_days):
    if (day - first_day).days < 84:
        return .6875
    start = max(first_day, day - dt.timedelta(days=365))
    weeks = max((day - start).days / 7., 1.)
    trailing_rate = sum(start <= selected < day for selected in chosen_days) / weeks
    return .675 if trailing_rate < 1.0 else .70


def targeted_policy(primary_score, reserve_score, dates, currencies, eligible, kind,
                    window=250, warmup=40):
    """Apply one prespecified causal AP15 controller and expose its full state."""
    if kind not in POLICY_KINDS:
        raise ValueError(kind)
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    signal = np.zeros(len(primary_score), dtype=bool)
    threshold = np.full(len(primary_score), np.nan)
    reason = np.zeros(len(primary_score), dtype=np.int8)  # 1 primary, 2 silence, 3 month

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
            if kind.startswith('adaptive100'):
                cut = _narrow_threshold(dates[i], first_day, chosen_days)
            elif kind.startswith('top3125'):
                cut = .6875
            else:
                cut = .70
            threshold[i] = cut
            primary = np.isfinite(primary_rank[i]) and primary_rank[i] > cut
            silence = False
            if kind == 'silence14_month24':
                last = chosen_days[-1] if chosen_days else first_day
                silence = ((dates[i] - last).days >= 14
                           and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .80)
            use_month = kind.endswith('month24')
            month_rescue = (use_month and dates[i].day >= 24 and month_used == 0
                            and np.isfinite(reserve_rank[i]) and reserve_rank[i] > .70)
            if eligible[i] and used < 2 and (primary or silence or month_rescue):
                signal[i] = True
                reason[i] = 1 if primary else (2 if silence else 3)
                chosen_days.append(dates[i])
                used += 1
                month_used += 1
    return signal, primary_rank, reserve_rank, threshold, reason
