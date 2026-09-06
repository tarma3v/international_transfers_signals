"""AP14 light causal cadence controllers over frozen AP12/AP13 scores."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS
from research.after_publication_ap12_effective_models import causal_rank_percentile


SCORE_NAMES = ('extra_roll2', 'local_extra', 'blend_roll2_local', 'extra_ap12')
POLICY_KINDS = ('top325', 'top35', 'silence14_r80', 'silence21_r70', 'adaptive105')


def fixed_scores(ap13, ap12):
    """Return the three frozen experts and one outcome-free 50/50 ensemble."""
    rolling = np.asarray(ap13['prediction__extra_roll2'], dtype=float)
    local = np.asarray(ap13['prediction__local_extra'], dtype=float)
    extra = np.asarray(ap12['prediction__extra_h5'], dtype=float)
    return {
        'extra_roll2': rolling,
        'local_extra': local,
        'blend_roll2_local': .5 * rolling + .5 * local,
        'extra_ap12': extra,
    }


def _adaptive_threshold(day, first_day, chosen_days):
    observed_days = (day - first_day).days
    if observed_days < 56:
        return .675
    start = max(first_day, day - dt.timedelta(days=182))
    weeks = max((day - start).days / 7., 1.)
    trailing = sum(start <= selected < day for selected in chosen_days) / weeks
    if trailing < .95:
        return .65
    if trailing < 1.05:
        return .675
    return .70


def light_cadence_policy(primary_score, reserve_score, dates, currencies, eligible, kind,
                         window=250, warmup=40):
    """Causal threshold/fallback with prior ranks, known-down veto and cap2."""
    if kind not in POLICY_KINDS:
        raise ValueError(kind)
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    primary_rank = causal_rank_percentile(primary_score, currencies, window, warmup)
    reserve_rank = causal_rank_percentile(reserve_score, currencies, window, warmup)
    signal = np.zeros(len(primary_score), dtype=bool)
    threshold = np.full(len(primary_score), np.nan)
    reason = np.zeros(len(primary_score), dtype=np.int8)  # 1 primary, 2 reserve

    for currency in CORRIDORS:
        indices = np.flatnonzero(currencies == currency)
        if not len(indices):
            continue
        first_day = dates[indices[0]]
        week, used = None, 0
        chosen_days = []
        for i in indices:
            iso = dates[i].isocalendar()[:2]
            if iso != week:
                week, used = iso, 0
            if kind == 'top325':
                cut = .675
            elif kind == 'top35':
                cut = .65
            elif kind == 'adaptive105':
                cut = _adaptive_threshold(dates[i], first_day, chosen_days)
            else:
                cut = .70
            threshold[i] = cut
            primary = np.isfinite(primary_rank[i]) and primary_rank[i] > cut
            fallback = False
            if kind.startswith('silence'):
                last = chosen_days[-1] if chosen_days else first_day
                age = (dates[i] - last).days
                reserve_cut = .80 if kind == 'silence14_r80' else .70
                silence = 14 if kind == 'silence14_r80' else 21
                fallback = (age >= silence and np.isfinite(reserve_rank[i])
                            and reserve_rank[i] > reserve_cut)
            if eligible[i] and used < 2 and (primary or fallback):
                signal[i] = True
                reason[i] = 1 if primary else 2
                chosen_days.append(dates[i])
                used += 1
    return signal, primary_rank, reserve_rank, threshold, reason
