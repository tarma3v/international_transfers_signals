"""Sequential same-week expert substitution for AP48."""
from __future__ import annotations

import numpy as np

from ml.data import CORRIDORS


def same_week_substitution_router(primary, core_veto, fallback, dates,
                                  currencies, eligible):
    primary = np.asarray(primary, dtype=bool)
    core_veto = np.asarray(core_veto, dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)
    dates = np.asarray(dates, dtype=object)
    currencies = np.asarray(currencies)
    eligible = np.asarray(eligible, dtype=bool)
    signal = np.zeros(len(primary), dtype=bool)
    reason = np.zeros(len(primary), dtype=np.int8)
    pending_before = np.zeros(len(primary), dtype=bool)
    used_before = np.zeros(len(primary), dtype=np.int8)
    for currency in CORRIDORS:
        week, used, pending = None, 0, False
        for i in np.flatnonzero(currencies == currency):
            iso = dates[i].isocalendar()[:2]
            if iso != week:
                week, used, pending = iso, 0, False
            pending_before[i] = pending
            used_before[i] = used
            if not eligible[i]:
                continue
            if primary[i] and used < 2:
                signal[i] = True
                reason[i] = 1
                used += 1
            if core_veto[i]:
                pending = True
            if not signal[i] and pending and fallback[i] and used < 2:
                signal[i] = True
                reason[i] = 2
                used += 1
                pending = False
    return signal, reason, pending_before, used_before
