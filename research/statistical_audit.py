"""Four-week moving-block bootstrap for the principal policies."""
from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import HORIZONS, benefit_forward_only, build_targets
from research.extended_features import load_or_build

OUT = Path("results/research")
FINAL_YEARS = (2024, 2025, 2026)
SEED = 20260904


def _four_week_bootstrap(y, benefit, dates, valid, fired, B=4000):
    frame = pd.DataFrame({
        "date": dates[valid], "y": y[valid], "fired": fired[valid],
        "benefit": benefit[valid],
    })
    frame["week"] = frame.date.map(
        lambda d: d - dt.timedelta(days=d.weekday())
    )
    grouped = []
    for _week, z in frame.groupby("week", sort=True):
        active = z.fired.to_numpy(bool)
        grouped.append((
            float(z.y.sum()), len(z), float(z.loc[active, "y"].sum()),
            int(active.sum()), float(z.loc[active, "benefit"].sum()),
            int(z.loc[active, "benefit"].notna().sum()),
        ))
    stats = np.asarray(grouped, dtype=float)
    n_weeks, block = len(stats), 4
    n_blocks = int(np.ceil(n_weeks / block))
    rng = np.random.default_rng(SEED)
    lift_draws = np.empty(B)
    benefit_draws = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n_weeks - block + 1, size=n_blocks)
        pick = np.concatenate([np.arange(s, s + block) for s in starts])[:n_weeks]
        z = stats[pick].sum(axis=0)
        base = z[0] / z[1]
        hit = z[2] / z[3]
        lift_draws[b] = hit / base
        benefit_draws[b] = z[4] / z[5]
    return {
        "lift_ci_low": float(np.quantile(lift_draws, .025)),
        "lift_ci_high": float(np.quantile(lift_draws, .975)),
        "benefit_ci_low": float(np.quantile(benefit_draws, .025)),
        "benefit_ci_high": float(np.quantile(benefit_draws, .975)),
    }


def _benefit(series, index, h):
    result = np.full(len(index), np.nan)
    for row, (currency, i, _day) in enumerate(index):
        value = benefit_forward_only(series[currency].values, i, h)
        if value is not None:
            result[row] = value
    return result


def _timestamp_fired(y1, y, dates, currencies):
    valid = np.asarray([d.year in FINAL_YEARS for d in dates]) & ~np.isnan(y)
    fired = np.zeros(len(y), dtype=bool)
    for currency in CORRIDORS:
        last = None
        for row in np.where(valid & (currencies == currency))[0]:
            if y1[row] == 1 and (last is None or (dates[row] - last).days >= 3):
                fired[row] = True
                last = dates[row]
    return valid, fired


def _anchor_fired(outputs, y, dates, currencies):
    valid = np.zeros(len(y), dtype=bool)
    fired = np.zeros(len(y), dtype=bool)
    for year in FINAL_YEARS:
        z = outputs[year]
        ca, te = z["calib_idx"], z["test_idx"]
        valid[te] = ~np.isnan(y[te])
        for currency in CORRIDORS:
            cm = currencies[ca] == currency
            tm = currencies[te] == currency
            cutoff = float(np.quantile(z["calib_score"][cm], .75))
            fired[te[tm]] = z["test_score"][tm] >= cutoff
    return valid, fired


def _summary(name, h, y, benefit, dates, valid, fired):
    active = valid & fired
    base = float(y[valid].mean())
    hit = float(y[active].mean())
    result = {
        "policy": name, "h": h, "n": int(active.sum()), "base_rate": base,
        "hit_rate": hit, "lift": hit / base,
        "forward_benefit_bps": float(np.nanmean(benefit[active])),
    }
    result.update(_four_week_bootstrap(y, benefit, dates, valid, fired))
    return result


def main():
    _X, _names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    targets = build_targets(series, index)
    y1 = targets["fav_h1"]
    rows = []
    for h in HORIZONS:
        y = targets[f"fav_h{h}"]
        benefit = _benefit(series, index, h)
        valid, fired = _timestamp_fired(y1, y, dates, currencies)
        rows.append(_summary("publication_timing", h, y, benefit, dates, valid, fired))

    with (OUT / "candidate_outputs_h5_v2.pkl").open("rb") as fh:
        outputs = pickle.load(fh)["anchor_trend"]
    y = targets["fav_h5"]
    benefit = _benefit(series, index, 5)
    valid, fired = _anchor_fired(outputs, y, dates, currencies)
    rows.append(_summary("anchor_trend_posthoc", 5, y, benefit, dates, valid, fired))

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "statistical_audit.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
