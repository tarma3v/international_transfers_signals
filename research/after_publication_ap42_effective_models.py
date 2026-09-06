"""Causal AP26-core/AP23-substitute router for AP42."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import CORRIDORS


def specialist_substitution_router(core, fallback, core_quality, dates,
                                   currencies, eligible,
                                   core_floor=1., fallback_floor=1.05):
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    core = np.asarray(core, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
    core_quality = np.asarray(core_quality, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    signal = np.zeros(len(core), dtype=bool)
    trailing_rate = np.full(len(core), np.nan)
    days_since = np.full(len(core), np.nan)
    reason = np.zeros(len(core), dtype=np.int8)
    core_veto = np.zeros(len(core), dtype=bool)
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
            allow_core = core_quality[i] or rate < core_floor
            use_core = core[i] and allow_core
            core_veto[i] = eligible[i] and core[i] and not allow_core
            use_fallback = (not core[i] and warmed and rate < fallback_floor
                            and fallback[i])
            if eligible[i] and used < 2 and (use_core or use_fallback):
                signal[i] = True
                reason[i] = 1 if core_quality[i] and use_core else (
                    2 if use_core else 3)
                chosen_days.append(dates[i])
                used += 1
    return signal, trailing_rate, days_since, reason, core_veto
