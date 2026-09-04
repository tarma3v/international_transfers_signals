"""Build compact figures for the third deep-research report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "research" / "round3"

NAVY = "#0f172a"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#d97706"
RED = "#dc2626"
GRAY = "#64748b"
LIGHT = "#e2e8f0"


LABELS = {
    "anchor_multiscale_locked": "Locked anchor",
    "anchor_trend_posthoc": "Trend anchor (posthoc)",
    "global_extra": "Global ExtraTrees",
    "router_equal_original": "Equal expert mix",
    "router_equal_balanced": "Balanced equal mix",
    "router_regime_soft": "Soft regime router",
    "recency_short_mix": "Short-window mix",
    "round3_consensus_geometric": "Geometric consensus",
    "round3_online_local_headline": "Online local Hedge",
    "postshock_reset_xgb_stable": "Post-2022 reset XGB",
    "recent_reset_anchor_blend": "Reset + anchor",
}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "axes.edgecolor": LIGHT,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def final_comparison(master: pd.DataFrame) -> None:
    frame = master[master.period == "retrospective_2024_2026"].copy()
    frame["label"] = frame.policy.map(LABELS)
    frame = frame.sort_values("lift")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(y - .18, frame.lift, height=.34, color=BLUE, label="Headline lift")
    ax.barh(y + .18, frame.macro_year_lift, height=.34, color=GREEN,
            label="Средний lift по годам")
    ax.axvline(1.0, color=NAVY, lw=1)
    ax.axvline(1.4, color=ORANGE, lw=1, ls="--", label="Ориентир 1.40")
    ax.set_yticks(y, frame.label)
    ax.set_xlim(.85, 1.48)
    ax.set_xlabel("Lift, h=5")
    ax.set_title("2024-2026: headline против устойчивости по годам")
    ax.grid(axis="x", color=LIGHT, lw=.7)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "report_final_comparison.png")


def benefit_comparison(master: pd.DataFrame) -> None:
    wanted = [
        "anchor_multiscale_locked", "router_equal_original",
        "router_regime_soft", "round3_consensus_geometric",
        "round3_online_local_headline", "postshock_reset_xgb_stable",
    ]
    frame = master[(master.period == "retrospective_2024_2026") &
                   master.policy.isin(wanted)].copy()
    frame["label"] = frame.policy.map(LABELS)
    frame = frame.set_index("policy").loc[wanted].reset_index()
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(y - .18, frame.forward_benefit_bps, height=.34, color=GREEN,
            label="Только будущее")
    ax.barh(y + .18, frame.symmetric_benefit_bps, height=.34, color=ORANGE,
            label="Официальное окно +-h")
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_yticks(y, frame.label)
    ax.set_xlabel("Средняя выгода на сигнал, б.п.")
    ax.set_title("Одна и та же политика по двум определениям цены дня")
    ax.grid(axis="x", color=LIGHT, lw=.7)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "report_benefit_comparison.png")


def annual_stability(breakdown: pd.DataFrame) -> None:
    wanted = [
        "anchor_multiscale_locked", "router_equal_original",
        "round3_consensus_geometric", "round3_online_local_headline",
        "postshock_reset_xgb_stable",
    ]
    frame = breakdown[(breakdown.breakdown == "year") &
                      breakdown.policy.isin(wanted)].copy()
    years = ["2024", "2025", "2026"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for policy in wanted:
        part = frame[frame.policy == policy].set_index("group").loc[years]
        axes[0].plot(years, part.lift, marker="o", lw=2, label=LABELS[policy])
        axes[1].plot(years, part.frequency, marker="o", lw=2, label=LABELS[policy])
    axes[0].axhline(1, color=NAVY, lw=1)
    axes[0].axhline(1.4, color=ORANGE, lw=1, ls="--")
    axes[0].set_title("Lift по годам")
    axes[0].set_ylabel("Lift")
    axes[1].axhspan(1, 2, color="#dcfce7", alpha=.8)
    axes[1].set_title("Частота по годам")
    axes[1].set_ylabel("Сигналов / валюта / неделя")
    for ax in axes:
        ax.grid(color=LIGHT, lw=.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.23)
    save(fig, "report_annual_stability.png")


def transfer_chart() -> None:
    rows = [
        ("Consensus", 1.375, 1.189, 1.264),
        ("Delayed labels", 1.316, 1.159, 1.184),
        ("Barrier path", 1.140, 1.076, 1.224),
        ("Cross-sectional", 1.371, 1.038, 1.173),
        ("Pooled hazard", 1.292, 1.089, 1.206),
        ("Currency champions", 1.342, .888, 1.044),
        ("Online SGD", 1.12, 1.047, 1.082),
    ]
    frame = pd.DataFrame(rows, columns=["family", "general", "shock", "final"])
    x = np.arange(len(frame))
    w = .25
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w, frame.general, width=w, color=GRAY, label="2017-2020")
    ax.bar(x, frame.shock, width=w, color=ORANGE, label="2022-2023")
    ax.bar(x + w, frame.final, width=w, color=BLUE, label="2024-2026*")
    ax.axhline(1, color=NAVY, lw=1)
    ax.set_xticks(x, frame.family, rotation=22, ha="right")
    ax.set_ylim(.7, 1.45)
    ax.set_ylabel("Лучший lift семейства, h=5")
    ax.set_title("Новые идеи: перенос через смену режима")
    ax.grid(axis="y", color=LIGHT, lw=.7)
    ax.legend(frameon=False, ncol=3)
    save(fig, "report_transfer.png")


def main() -> None:
    setup()
    master = pd.read_csv(OUT / "master_policy_metrics.csv")
    breakdown = pd.read_csv(OUT / "master_final_breakdown.csv", dtype={"group": str})
    final_comparison(master)
    benefit_comparison(master)
    annual_stability(breakdown)
    transfer_chart()
    print("wrote", *(path.name for path in sorted(OUT.glob("report_*.png"))))


if __name__ == "__main__":
    main()
