"""Calendar-aware decision router for AP33."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS


def calendar_fallback_router(core, fallback, dates, currencies, eligible):
    dates = np.asarray(dates)
    currencies = np.asarray(currencies)
    core = np.asarray(core, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
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
            use_fallback = (not use_core and warmed and rate < 1.
                            and fallback[i] and (late_week or silence))
            if eligible[i] and used < 2 and (use_core or use_fallback):
                signal[i] = True
                if use_core:
                    reason[i] = 1
                elif late_week:
                    reason[i] = 2
                else:
                    reason[i] = 3
                chosen_days.append(dates[i])
                used += 1
    return signal, trailing_rate, days_since, reason
