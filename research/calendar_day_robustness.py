"""Calendar-day robustness check with causal forward-fill.

Official rates remain unchanged between publications.  This check expands each
series to a daily grid, forward-fills only already published values, builds the
five targets in calendar days, and permits alerts only on an update day.  It is
kept separate from the primary publication-step protocol because the two
horizon definitions answer different product questions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS

OUT = Path("results/research")
HORIZONS = (1, 3, 5, 10, 20)
RATE = 0.25


def _calendar_frame(series, currency: str) -> pd.DataFrame:
    s = series[currency]
    known_next = {
        day: bool(nxt >= cur)
        for day, cur, nxt in zip(s.dates[:-1], s.values[:-1], s.values[1:])
    }
    source_index = pd.DatetimeIndex(s.dates)
    days = pd.date_range(source_index.min(), source_index.max(), freq="D")
    values = pd.Series(s.values, index=source_index).reindex(days).ffill().to_numpy()
    is_update = days.isin(source_index)
    rows = []
    for i in range(90, len(days)):
        window = values[i - 89:i + 1]
        lo, hi = float(window.min()), float(window.max())
        pct = 50.0 if hi <= lo else (float(values[i]) - lo) / (hi - lo) * 100.0
        ret20 = -(float(values[i]) / float(values[i - 20]) - 1.0) * 10000.0
        ret60 = -(float(values[i]) / float(values[i - 60]) - 1.0) * 10000.0
        day = days[i].date()
        rows.append((day, float(values[i]), bool(is_update[i]),
                     known_next.get(day, False), pct + .035 * ret20 + .015 * ret60, i))
    return pd.DataFrame(rows, columns=[
        "date", "value", "is_update", "known_next_no_worse", "score", "i"
    ])


def main():
    from research.extended_features import load_or_build

    _X, _names, _index, series = load_or_build()
    frames = {currency: _calendar_frame(series, currency) for currency in CORRIDORS}
    rows = []
    timing_rows = []
    for h in HORIZONS:
        all_test = []
        for currency, frame in frames.items():
            values = pd.Series(series[currency].values,
                               index=pd.DatetimeIndex(series[currency].dates))
            grid = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="D")).ffill()
            arr = grid.to_numpy()
            local = frame.copy()
            y = np.full(len(local), np.nan)
            benefit = np.full(len(local), np.nan)
            for r, i in enumerate(local.i.to_numpy()):
                if i + h < len(arr):
                    future = arr[i + 1:i + h + 1]
                    y[r] = float(arr[i] <= future.min())
                    benefit[r] = -(arr[i] - float(future.mean())) / float(future.mean()) * 10000.0
            local["y"] = y
            local["benefit"] = benefit
            local["currency"] = currency
            local["fired"] = False
            for year in (2024, 2025, 2026):
                cal = (
                    (local.date >= pd.Timestamp(year - 1, 1, 1).date())
                    & (local.date < pd.Timestamp(year, 1, 1).date())
                    & local.is_update & local.y.notna()
                )
                test = (
                    local.date.map(lambda d: d.year == year)
                    & local.is_update & local.y.notna()
                )
                cutoff = float(local.loc[cal, "score"].quantile(1.0 - RATE))
                local.loc[test, "fired"] = local.loc[test, "score"] >= cutoff
            all_test.append(local[
                local.date.map(lambda d: d.year in (2024, 2025, 2026))
                & local.is_update & local.y.notna()
            ])
        test = pd.concat(all_test, ignore_index=True)
        active = test.fired.to_numpy(bool)
        base = float(test.y.mean())
        hit = float(test.loc[active, "y"].mean())
        span_weeks = (test.date.max() - test.date.min()).days / 7.0
        corridor_lifts = []
        corridor_freqs = []
        for currency in CORRIDORS:
            z = test[test.currency == currency]
            corridor_lifts.append(float(z.loc[z.fired, "y"].mean() / z.y.mean()))
            corridor_freqs.append(float(z.fired.sum() / span_weeks))
        rows.append({
            "h_calendar_days": h, "rate_target": RATE, "n": int(active.sum()),
            "frequency": float(active.sum() / len(CORRIDORS) / span_weeks),
            "hit_rate": hit, "base_rate": base, "lift": hit / base,
            "forward_benefit_bps": float(test.loc[active, "benefit"].mean()),
            "corridor_lift_min": min(corridor_lifts),
            "corridor_freq_min": min(corridor_freqs),
            "corridor_freq_max": max(corridor_freqs),
            "decision_days": "publication_only",
        })
        timing_active = np.zeros(len(test), dtype=bool)
        for currency in CORRIDORS:
            last_fire = None
            for row in test.index[test.currency == currency]:
                enough_gap = (
                    last_fire is None or (test.at[row, "date"] - last_fire).days >= 3
                )
                if bool(test.at[row, "known_next_no_worse"]) and enough_gap:
                    timing_active[row] = True
                    last_fire = test.at[row, "date"]
        timing_hit = float(test.loc[timing_active, "y"].mean())
        timing_corridor_lifts = []
        timing_corridor_freqs = []
        for currency in CORRIDORS:
            z = test[test.currency == currency]
            fired_currency = timing_active[z.index]
            timing_corridor_lifts.append(
                float(z.loc[fired_currency, "y"].mean() / z.y.mean())
            )
            timing_corridor_freqs.append(float(fired_currency.sum() / span_weeks))
        timing_rows.append({
            "h_calendar_days": h, "n": int(timing_active.sum()),
            "frequency": float(timing_active.sum() / len(CORRIDORS) / span_weeks),
            "hit_rate": timing_hit, "base_rate": base, "lift": timing_hit / base,
            "forward_benefit_bps": float(test.loc[timing_active, "benefit"].mean()),
            "corridor_lift_min": min(timing_corridor_lifts),
            "corridor_freq_min": min(timing_corridor_freqs),
            "corridor_freq_max": max(timing_corridor_freqs),
            "decision_days": "publication_only_after_next_rate_release",
            "cooldown_days": 3,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "calendar_day_robustness.csv", index=False)
    print(result.to_string(index=False))
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(OUT / "calendar_day_publication_timing.csv", index=False)
    print("\nTIMESTAMP-AWARE")
    print(timing.to_string(index=False))


if __name__ == "__main__":
    main()
