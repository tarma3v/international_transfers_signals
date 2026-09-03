"""Generate compact scientific figures for the final PDF report."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data import CORRIDORS
from research.extended_features import load_or_build

OUT = Path("results/research")
COLORS = ["#2563eb", "#ef4444", "#16a34a", "#9333ea", "#f59e0b"]


def summary_figure():
    stats = pd.read_csv(OUT / "statistical_audit.csv")
    pub = stats[stats.policy == "publication_timing"].sort_values("h")
    cal = pd.read_csv(OUT / "calendar_day_robustness.csv")
    cal_timing = pd.read_csv(OUT / "calendar_day_publication_timing.csv")
    regime = pd.read_csv(OUT / "regime_audit_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    err = np.vstack([pub.lift - pub.lift_ci_low, pub.lift_ci_high - pub.lift])
    ax.errorbar(pub.h, pub.lift, yerr=err, marker="o", capsize=4, color=COLORS[0])
    ax.axhline(1.3, color="#dc2626", linestyle="--", linewidth=1, label="case target 1.30")
    ax.set_xticks(pub.h)
    ax.set_ylim(1.2, 2.3)
    ax.set_title("Timestamp-aware future-only lift (2024-2026)")
    ax.set_xlabel("h, CBR publications")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(cal.h_calendar_days, cal.lift, marker="o", label="past-only anchor", color=COLORS[1])
    ax.plot(cal_timing.h_calendar_days, cal_timing.lift, marker="o",
            label="after next-rate release", color=COLORS[2])
    ax.axhline(1.3, color="#111827", linestyle="--", linewidth=1)
    ax.set_xticks(cal.h_calendar_days)
    ax.set_title("Calendar-day robustness with causal forward-fill")
    ax.set_xlabel("h, calendar days")
    ax.legend(frameon=False)

    order = ["pre_2022", "shock_adaptation", "mature_postshock"]
    labels = ["2017-Feb 2022", "Feb 2022-2023", "2024-2026"]
    z = regime.set_index("period").loc[order]
    ax = axes[1, 0]
    x = np.arange(3)
    ax.bar(x - .18, z.mean_abs_daily_move_bps, width=.36, color=COLORS[3],
           label="mean |daily move|, bps")
    ax2 = ax.twinx()
    ax2.bar(x + .18, z.fav_h5_base_rate, width=.36, color=COLORS[4],
            label="h5 base rate")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("bps")
    ax2.set_ylabel("base rate")
    ax.set_title("Predeclared 24-Feb-2022 regime break")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels2 = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels2, frameon=False, fontsize=8, loc="upper right")

    ax = axes[1, 1]
    models = ["ETS-5", "SARIMA-5", "GRU", "pct_range_90"]
    auc = [.478, .523, .540, .572]
    bars = ax.bar(models, auc, color=["#94a3b8", "#64748b", "#475569", COLORS[0]])
    ax.axhline(.5, color="#111827", linestyle="--", linewidth=1)
    ax.set_ylim(.44, .60)
    ax.set_ylabel("walk-forward ROC AUC")
    ax.set_title("h=5 ranking: complex TS models vs simple anchor")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    fig.savefig(OUT / "report_summary.png", dpi=190)
    plt.close(fig)


def regime_rates_figure():
    _X, _names, _index, series = load_or_build()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for currency, color in zip(CORRIDORS, COLORS):
        s = series[currency]
        frame = pd.Series(s.values, index=pd.DatetimeIndex(s.dates))
        frame = frame[frame.index >= "2019-01-01"].resample("MS").mean().dropna()
        base = frame.loc[frame.index >= "2022-01-01"].iloc[0]
        normalized = frame / base * 100.0
        axes[0].plot(normalized.index, normalized, label=currency, color=color, linewidth=1.5)
        returns = np.log(frame).diff() * 10000
        axes[1].plot(returns.index, returns.rolling(3).std(), color=color,
                     label=currency, linewidth=1.3)
    cut = pd.Timestamp(dt.date(2022, 2, 24))
    for ax in axes:
        ax.axvline(cut, color="#111827", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
    axes[0].set_title("CBR corridor rates, monthly mean (Jan-2022 = 100)")
    axes[0].set_ylabel("index")
    axes[0].legend(ncol=5, frameon=False)
    axes[1].set_title("Rolling 3-month volatility of monthly log changes")
    axes[1].set_ylabel("bps")
    axes[1].set_xlabel("effective date")
    fig.savefig(OUT / "report_regimes.png", dpi=190)
    plt.close(fig)


def seasonality_figure():
    season = pd.read_csv(OUT / "train_seasonality.csv")
    z = season[season.field == "month"].pivot(index="currency", columns="level", values="lift")
    z = z.loc[list(CORRIDORS)]
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    image = ax.imshow(z.to_numpy(), cmap="RdYlBu_r", vmin=.65, vmax=1.45, aspect="auto")
    ax.set_xticks(np.arange(12), [str(i) for i in range(1, 13)])
    ax.set_yticks(np.arange(len(z)), z.index)
    ax.set_xlabel("month")
    ax.set_title("Month lift for fav_h5, development sample only (through 2016)")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=.85, label="lift vs currency base rate")
    fig.savefig(OUT / "report_seasonality.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    summary_figure()
    regime_rates_figure()
    seasonality_figure()
    print("wrote report_summary.png, report_regimes.png, report_seasonality.png")
