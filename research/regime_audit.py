"""Descriptive regime audit around the predeclared 24-Feb-2022 break."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from ml.targets import build_targets
from research.extended_features import load_or_build

OUT = Path("results/research")
PERIODS = {
    "pre_2022": (dt.date(2017, 1, 1), dt.date(2022, 2, 23)),
    "shock_adaptation": (dt.date(2022, 2, 24), dt.date(2023, 12, 31)),
    "mature_postshock": (dt.date(2024, 1, 1), dt.date(2026, 12, 31)),
}


def main():
    X, names, index, series = load_or_build()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    ret1 = X[:, names.index("ret_1")]
    pct90 = X[:, names.index("pct_range_90")]
    rows = []
    for period, (start, end) in PERIODS.items():
        pm = np.asarray([start <= day <= end for day in dates])
        for currency in CORRIDORS:
            mask = pm & (currencies == currency) & ~np.isnan(y)
            rows.append({
                "period": period, "currency": currency, "n": int(mask.sum()),
                "fav_h5_base_rate": float(y[mask].mean()),
                "mean_abs_daily_move_bps": float(np.abs(ret1[mask]).mean()),
                "daily_move_std_bps": float(ret1[mask].std()),
                "pct90_mean": float(pct90[mask].mean()),
                "share_at_90d_high": float((pct90[mask] >= 95).mean()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "regime_audit.csv", index=False)
    summary = result.groupby("period").agg({
        "fav_h5_base_rate": "mean", "mean_abs_daily_move_bps": "mean",
        "daily_move_std_bps": "mean", "pct90_mean": "mean",
        "share_at_90d_high": "mean",
    })
    summary.to_csv(OUT / "regime_audit_summary.csv")
    print(summary.to_string())
    print("\nPer corridor")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
