"""Publication-quality figures for the round-two PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/research/round2")
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#d97706"
PURPLE = "#7c3aed"
SLATE = "#475569"
RED = "#dc2626"


LABELS = {
    "anchor_multiscale_locked": "Locked anchor",
    "new_global_extra": "Global ExtraTrees",
    "router_equal": "Equal experts",
    "router_regime_soft": "Soft regime router",
    "router_global_soft": "Trailing global weights",
    "recency_window_short": "Short-window mix",
    "recency_window3": "3-year ExtraTrees",
    "local_to_global_xgb": "Local -> global residual",
}


def model_comparison():
    z = pd.read_csv(OUT / "round2_block_bootstrap.csv")
    order = [
        "anchor_multiscale_locked", "new_global_extra", "router_equal",
        "router_regime_soft", "router_global_soft", "recency_window_short",
        "local_to_global_xgb",
    ]
    periods = ["shock_2022_2023", "retrospective_2024_2026"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, constrained_layout=True)
    colors = [SLATE, BLUE, GREEN, PURPLE, ORANGE, "#0891b2", "#94a3b8"]
    for ax, period, title in zip(axes, periods, ["2022-2023 validation", "2024-2026 retrospective"]):
        p = z[z.period == period].set_index("policy").loc[order]
        y = np.arange(len(order))
        err = np.vstack([p.lift - p.lift_ci_low, p.lift_ci_high - p.lift])
        ax.errorbar(p.lift, y, xerr=err, fmt="none", color="#94a3b8", capsize=3, zorder=1)
        ax.scatter(p.lift, y, c=colors, s=55, zorder=2)
        ax.axvline(1.0, color="#111827", linewidth=1)
        ax.axvline(1.3, color=RED, linestyle="--", linewidth=1.2, label="case target 1.30")
        ax.set_yticks(y, [LABELS[name] for name in order])
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
        ax.set_xlabel("future-only lift (95% four-week block CI)")
        ax.set_title(title)
        for yi, (_, row) in enumerate(p.iterrows()):
            ax.text(row.lift + .015, yi - .18, f"{row.lift:.3f} | {row.frequency:.2f}/wk",
                    fontsize=8, color="#111827")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("New model families: effect size, uncertainty and alert frequency", fontsize=14)
    fig.savefig(OUT / "report_model_comparison.png", dpi=200)
    plt.close(fig)


def year_stability():
    z = pd.read_csv(OUT / "round2_finalist_breakdown.csv")
    z = z[(z.breakdown == "year") & z.policy.isin([
        "anchor_multiscale_locked", "router_equal", "router_regime_soft",
        "router_global_soft", "recency_window_short",
    ])]
    colors = {
        "anchor_multiscale_locked": SLATE, "router_equal": GREEN,
        "router_regime_soft": PURPLE, "router_global_soft": ORANGE,
        "recency_window_short": "#0891b2",
    }
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    for policy, q in z.groupby("policy"):
        q = q.assign(year=q.group.astype(int)).sort_values("year")
        ax.plot(q.year, q.lift, marker="o", linewidth=2, color=colors[policy],
                label=LABELS[policy])
    ax.axhline(1.0, color="#111827", linewidth=1)
    ax.axhline(1.3, color=RED, linestyle="--", linewidth=1.1)
    ax.axvspan(2023.5, 2026.5, color="#f8fafc", zorder=-2)
    ax.text(2024.02, ax.get_ylim()[1] if ax.get_ylim()[1] else 2, "retrospective block",
            color="#64748b", fontsize=9, va="top")
    ax.set_xticks(sorted(z.group.astype(int).unique()))
    ax.set_ylabel("lift")
    ax.set_title("The maximum average is not the most stable policy")
    ax.grid(alpha=.18); ax.legend(ncol=3, frameon=False, loc="lower left")
    fig.savefig(OUT / "report_year_stability.png", dpi=200)
    plt.close(fig)


def data_structure():
    pca = pd.read_csv(OUT / "eda_common_factor.csv")
    season = pd.read_csv(OUT / "eda_seasonal_stability.csv")
    block_order = ["development", "general_validation", "transition",
                   "shock_adaptation", "retrospective_final"]
    labels = ["2011-16", "2017-20", "2021", "2022-23", "2024-26*"]
    pairs = [
        ("development", "general_validation", "Dev -> general"),
        ("general_validation", "shock_adaptation", "General -> shock"),
        ("shock_adaptation", "retrospective_final", "Shock -> final*"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    q = pca.set_index("block").loc[block_order]
    bars = axes[0].bar(labels, q.pc1_variance_share, color=[SLATE, BLUE, "#64748b", ORANGE, GREEN])
    axes[0].bar_label(bars, labels=[f"{v:.0%}" for v in q.pc1_variance_share], padding=3)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("share of standardized daily variance")
    axes[0].set_title("One common FX factor dominates")
    matrix = []
    for left, right, _label in pairs:
        values = []
        for currency in ("TJS", "UZS", "KGS", "AMD", "KZT"):
            row = season[(season.currency == currency) & (season.left_block == left)
                         & (season.right_block == right)]
            values.append(float(row.spearman_month_pattern.iloc[0]))
        matrix.append(values)
    image = axes[1].imshow(np.asarray(matrix), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(5), ["TJS", "UZS", "KGS", "AMD", "KZT"])
    axes[1].set_yticks(range(3), [p[2] for p in pairs])
    axes[1].set_title("Monthly patterns do not transfer reliably")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axes[1].text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1], shrink=.8, label="Spearman correlation")
    fig.savefig(OUT / "report_data_structure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    model_comparison(); year_stability(); data_structure()
    print("wrote three round-two report figures")
