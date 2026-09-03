"""Conditional h=1 audit using the CBR rate announced during the current day.

The CBR effective-date series labels a rate by the day on which it applies.
Operationally, the next effective rate is normally posted on the preceding
working day.  Therefore ``values[i + 1]`` is permitted here ONLY for a signal
sent after that publication.  It is intentionally isolated from the ordinary
feature matrix: if the product sends earlier, this whole scenario is invalid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.evaluate import rate_per_week
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.extended_features import load_or_build
from research.model_study import FINAL_TEST_YEARS, REGIME_VALID_YEARS

OUT = Path("results/research")
# A three-calendar-day cooldown gives 1--2 alerts per corridor/week on both the
# predeclared 2022--2023 validation and the 2024--2026 audit period.
COOLDOWN_DAYS = 3


def evaluate(y1, y, benefit, dates, currencies, years, h):
    scope = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    scope[:] = np.asarray([day.year in years for day in dates]) & ~np.isnan(y)
    for currency in CORRIDORS:
        last_fire = None
        for row in np.where(scope & (currencies == currency))[0]:
            known_next_is_no_worse = y1[row] == 1
            enough_gap = (
                last_fire is None or (dates[row] - last_fire).days >= COOLDOWN_DAYS
            )
            if known_next_is_no_worse and enough_gap:
                fired[row] = True
                last_fire = dates[row]
    valid = scope & ~np.isnan(y)
    active = fired & valid
    base = float(y[valid].mean())
    hit = float(y[active].mean())
    corridor_lifts = []
    corridor_freqs = []
    for currency in CORRIDORS:
        cm = valid & (currencies == currency)
        fm = active & (currencies == currency)
        corridor_lifts.append(float(y[fm].mean() / y[cm].mean()))
        corridor_freqs.append(rate_per_week(int(fm.sum()), 1, dates, cm))
    year_lifts = []
    for year in years:
        ym = valid & np.asarray([day.year == year for day in dates])
        fm = active & np.asarray([day.year == year for day in dates])
        year_lifts.append(float(y[fm].mean() / y[ym].mean()))
    return {
        "h": h, "years": str(tuple(years)), "n": int(active.sum()),
        "frequency": rate_per_week(int(active.sum()), len(CORRIDORS), dates, valid),
        "hit_rate": hit, "base_rate": base, "lift": hit / base,
        "forward_benefit_bps": float(np.nanmean(benefit[active])),
        "year_lift_min": min(year_lifts),
        "corridor_lift_min": min(corridor_lifts),
        "corridor_freq_min": min(corridor_freqs),
        "corridor_freq_max": max(corridor_freqs),
        "cooldown_days": COOLDOWN_DAYS,
        "timing_condition": "send only after next effective CBR rate is published",
    }


def main():
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y1 = targets["fav_h1"]
    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        benefit = np.full(len(index), np.nan)
        for row, (currency, i, _day) in enumerate(index):
            value = benefit_forward_only(series[currency].values, i, h)
            if value is not None:
                benefit[row] = value
        rows.extend([
            evaluate(y1, y, benefit, dates, currencies, REGIME_VALID_YEARS, h),
            evaluate(y1, y, benefit, dates, currencies, FINAL_TEST_YEARS, h),
        ])
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "publication_timing_h1.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
