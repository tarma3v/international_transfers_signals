"""Build one like-for-like case matrix for AP37 and mandatory baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.baselines import BASELINES
from ml.data import CORRIDORS, REFERENCE, load
from ml.features import build_matrix
from ml.targets import HORIZONS, target_window_closing
from research.after_publication_ap37_effective import CANDIDATE, DATA


SOURCE = Path("results/research/after_publication/ap37_effective")
OUT = Path("results/research/final_case_metric_matrix.csv")


def _baseline_signals(series, panel):
    x, names, index = build_matrix(series, CORRIDORS, REFERENCE)
    lookup = {(currency, position): row
              for row, (currency, position, _day) in enumerate(index)}
    rows = np.asarray([
        lookup[(currency, int(position))]
        for currency, position in zip(panel.currency, panel.current_index)
    ])
    return {
        name: np.asarray(function(x, names)[rows], dtype=bool)
        for name, function in BASELINES.items()
    }


def _window_closing(series, panel, horizon):
    values = np.full(len(panel), np.nan)
    for row, (currency, position) in enumerate(zip(
            panel.currency, panel.current_index)):
        value = target_window_closing(
            series[currency].values, int(position), horizon)
        if value is not None:
            values[row] = value
    return values


def _metric_row(panel, signal, target, scope, symmetric, forward,
                indicator, target_name, currency, horizon):
    dates = panel.date.to_numpy()
    corridor = panel.currency.eq(currency).to_numpy()
    valid = scope & corridor & np.isfinite(target)
    fired = valid & signal
    base = float(np.mean(target[valid])) if valid.any() else np.nan
    hit = float(np.mean(target[fired])) if fired.any() else np.nan
    lift = hit / base if fired.any() and base > 0 else np.nan
    years = np.asarray([day.year for day in dates])
    covered_weeks = 0.0
    for year in sorted(set(years[valid])):
        year_scope = valid & (years == year)
        if year_scope.any():
            covered_weeks += (
                (max(dates[year_scope]) - min(dates[year_scope])).days + 1
            ) / 7.0
    weeks = int(fired.sum()) / covered_weeks if covered_weeks else np.nan
    selected_dates = dates[fired]
    if len(selected_dates):
        week_counts = pd.Series(1, index=pd.to_datetime(selected_dates)).groupby(
            lambda stamp: stamp.isocalendar()[:2]).sum()
        weekly_max = int(week_counts.max())
        max_gap = (max(np.diff(sorted(set(selected_dates)))) .days
                   if len(set(selected_dates)) > 1 else np.nan)
    else:
        weekly_max = 0
        max_gap = np.nan
    return {
        "indicator": indicator,
        "target": target_name,
        "corridor": currency,
        "h": horizon,
        "n_scope": int(valid.sum()),
        "n_signals": int(fired.sum()),
        "hit_rate": hit,
        "base_rate": base,
        "lift": lift,
        "signals_per_week": weeks,
        "weekly_max": weekly_max,
        "calendar_gap_max_days": max_gap,
        "symmetric_bps": float(np.nanmean(symmetric[fired]))
        if fired.any() else np.nan,
        "future_only_bps": float(np.nanmean(forward[fired]))
        if fired.any() else np.nan,
        "decision_clock": "18:30 Europe/Moscow (calendar-assumed receipt)",
        "reference": "today-effective CBR",
        "evaluation_period": "2024-01-01..2026-08-31 opened retrospective",
    }


def build_matrix_table():
    panel = pd.read_csv(SOURCE / "announcement_panel.csv")
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    with np.load(SOURCE / "outputs.npz") as source:
        arrays = {key: source[key] for key in source.files}
    policies = {
        "AP37 mature-precision router": arrays["signal__" + CANDIDATE].astype(bool),
        **_baseline_signals(series, panel),
    }
    scope = arrays["later"].astype(bool)
    rows = []
    for horizon in HORIZONS:
        targets = {
            "now_favourable": arrays[f"y{horizon}"],
            "window_closing": _window_closing(series, panel, horizon),
        }
        symmetric = arrays[f"sym{horizon}"]
        forward = arrays[f"forward{horizon}"]
        for indicator, signal in policies.items():
            for target_name, target in targets.items():
                for currency in CORRIDORS:
                    rows.append(_metric_row(
                        panel, signal, target, scope, symmetric, forward,
                        indicator, target_name, currency, horizon))
    return pd.DataFrame(rows)


def main():
    table = build_matrix_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    ap37 = table[
        table.indicator.eq("AP37 mature-precision router")
        & table.target.eq("now_favourable")
    ]
    print(ap37[["corridor", "h", "n_signals", "hit_rate", "lift",
                 "signals_per_week", "symmetric_bps",
                 "future_only_bps"]].to_string(index=False))
    print(f"\nwrote {OUT}: {len(table)} rows")


if __name__ == "__main__":
    main()
