"""Decision-level causal router for AP32."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS


def decision_deficit_router(core, fallback, dates, currencies, eligible):
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    core = np.asarray(core, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    signal = np.zeros(len(core), dtype=bool)
    trailing_rate = np.full(len(core), np.nan)
    reason = np.zeros(len(core), dtype=np.int8)
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
            start = max(first_day, dates[i] - dt.timedelta(days=365))
            weeks = max((dates[i] - start).days / 7., 1.)
            rate = sum(start <= chosen < dates[i]
                       for chosen in chosen_days) / weeks
            trailing_rate[i] = rate
            use_core = core[i]
            use_fallback = not use_core and rate < 1. and fallback[i]
            if eligible[i] and used < 2 and (use_core or use_fallback):
                signal[i] = True
                reason[i] = 1 if use_core else 2
                chosen_days.append(dates[i])
                used += 1
    return signal, trailing_rate, reason
