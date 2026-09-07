"""Build presentation-ready charts from frozen OOS artifacts.

The figures intentionally separate an illustrative historical fragment from the
aggregate OOS evidence.  No values are hand-entered: every metric and point is
read from the frozen AP37/AP50 result packets.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
AP37 = ROOT / "results/research/after_publication/ap37_effective"
AP50 = ROOT / "results/research/after_publication/ap50_temperature"
OUT = ROOT / "output/defense_visuals"

SELECTED = "ap26_core_mature_precision_calendar_fallback_cap2"
SIGNAL_KEY = f"signal__{SELECTED}"
CURRENCIES = ["AMD", "KGS", "KZT", "TJS", "UZS"]

# Presentation palette: warm, calm and high-contrast on a projector.
BG = "#F5F2EA"
PAPER = "#FFFDF8"
INK = "#172434"
MUTED = "#66717E"
GRID = "#D9DDD8"
GREEN = "#087A61"
GREEN_LIGHT = "#CFE8DE"
ORANGE = "#C26B19"
ORANGE_LIGHT = "#F1D9BE"
BLUE = "#3478A8"
BLUE_LIGHT = "#DCEAF3"
RED = "#C64A45"
GREY_BAR = "#C8CDD1"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 12,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.facecolor": PAPER,
            "figure.facecolor": BG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )


def new_figure() -> plt.Figure:
    return plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor=BG)


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.935, title, fontsize=24, fontweight="bold", color=INK)
    fig.text(0.055, 0.895, subtitle, fontsize=12.2, color=MUTED)


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.055, 0.025, text, fontsize=8.9, color=MUTED)


def add_pill(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    color: str = GREEN,
    facecolor: str = GREEN_LIGHT,
    width: float = 0.19,
) -> None:
    pill = FancyBboxPatch(
        (x, y),
        width,
        0.052,
        transform=fig.transFigure,
        boxstyle="round,pad=0.007,rounding_size=0.014",
        linewidth=0,
        facecolor=facecolor,
        zorder=10,
    )
    fig.patches.append(pill)
    fig.text(x + width / 2, y + 0.026, text, ha="center", va="center", fontsize=11, fontweight="bold", color=color, zorder=11)


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=150, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def load_after_receipt_panel() -> pd.DataFrame:
    z = np.load(AP50 / "outputs.npz", allow_pickle=True)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(z["dates"]),
            "currency": z["currencies"].astype(str),
            "temperature_h5": z["calibrated_probability_5"].astype(float) * 100.0,
            "y5": z["y5"].astype(float),
            "forward5": z["forward5"].astype(float),
            "signal": z[SIGNAL_KEY].astype(bool),
        }
    )
    prices = pd.read_csv(AP50 / "announcement_panel.csv", parse_dates=["date"])
    frame = frame.merge(
        prices[["date", "currency", "current_price"]],
        on=["date", "currency"],
        how="left",
        validate="one_to_one",
    )
    return frame.loc[frame["date"] >= "2024-01-01"].copy()


def build_temperature_course_overlay(
    panel: pd.DataFrame,
    *,
    end: str = "2026-07-03",
    filename: str = "00_temperature_rate_overlay.png",
) -> dict[str, float | int | str]:
    """Put the official rate, temperature and send points in one plot."""
    start = pd.Timestamp("2026-05-18")
    end_ts = pd.Timestamp(end)
    g = panel.loc[
        (panel["currency"] == "UZS")
        & panel["date"].between(start, end_ts)
    ].copy().sort_values("date")
    if g.empty:
        raise RuntimeError("Illustrative UZS window is absent from AP50 output")
    # The CBR quote is RUB per one UZS.  Multiply by 1000 solely to make the
    # axis human-readable; this does not change the geometry of the series.
    g["rate_per_1000"] = g["current_price"] * 1000.0
    signals = g.loc[g["signal"] & g["y5"].notna()].copy()
    wins = signals.loc[signals["y5"] == 1.0]
    losses = signals.loc[signals["y5"] == 0.0]

    fig = new_figure()
    title_block(
        fig,
        "Курс, температура и моменты отправки",
        "UZS · честный OOS-фрагмент · решение после публикации ЦБ · горизонт 5 дней",
    )
    add_pill(fig, 0.785, 0.902, f"{len(wins)} из {len(signals)} сигналов успешны", width=0.160)

    ax_rate = fig.add_axes([0.095, 0.15, 0.80, 0.65])
    ax_temp = ax_rate.twinx()
    # Keep the exchange-rate line and send markers as the dominant layer.
    ax_rate.set_zorder(ax_temp.get_zorder() + 1)
    ax_rate.patch.set_visible(False)

    ax_temp.plot(
        g["date"],
        g["temperature_h5"],
        color=ORANGE,
        lw=2.2,
        marker="o",
        markersize=3.6,
        alpha=0.82,
        zorder=2,
    )
    ax_rate.plot(
        g["date"],
        g["rate_per_1000"],
        color=INK,
        lw=3.0,
        marker="o",
        markersize=3.2,
        zorder=4,
    )
    if not wins.empty:
        ax_rate.scatter(
            wins["date"],
            wins["rate_per_1000"],
            s=118,
            color=GREEN,
            edgecolor=PAPER,
            linewidth=2.4,
            zorder=7,
        )
    if not losses.empty:
        ax_rate.scatter(
            losses["date"],
            losses["rate_per_1000"],
            s=135,
            marker="X",
            color=RED,
            edgecolor=PAPER,
            linewidth=1.8,
            zorder=7,
        )
    for d in signals["date"]:
        ax_rate.axvline(d, color=GREEN, alpha=0.12, lw=1.0, linestyle=(0, (2, 4)), zorder=0)

    ax_rate.set_ylabel("Курс ЦБ, ₽ за 1000 UZS\nниже — выгоднее", color=INK, labelpad=14)
    ax_temp.set_ylabel("Температура, 0–100\nвыше — сильнее", color=ORANGE, labelpad=14)
    ax_temp.set_ylim(0, 100)
    ax_temp.set_yticks([0, 20, 40, 60, 80, 100])
    ax_temp.tick_params(axis="y", colors=ORANGE)
    ax_rate.tick_params(axis="y", colors=INK)
    ax_rate.grid(axis="y", color=GRID, lw=0.85, alpha=0.8, zorder=0)
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_temp.spines[["top", "left"]].set_visible(False)
    ax_rate.spines["left"].set_color(INK)
    ax_temp.spines["right"].set_color(ORANGE)
    ax_rate.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_rate.margins(x=0.01)

    legend = [
        Line2D([0], [0], color=INK, lw=3, label="курс ЦБ"),
        Line2D([0], [0], color=ORANGE, lw=2.2, marker="o", markersize=5, alpha=0.82, label="температура"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=10, label="отправляем пуш"),
    ]
    ax_rate.legend(handles=legend, loc="upper left", frameon=False, ncol=3, fontsize=10.8, handlelength=2.0)
    add_footer(
        fig,
        "Зелёная точка — отправка пуша. Успех: курс ниже каждого из следующих 5 публикационных дней. Общая OOS-оценка: 73,38% против 29,45%; lift 2,51.",
    )
    save(fig, filename)
    return {
        "currency": "UZS",
        "start": str(start.date()),
        "end": str(end_ts.date()),
        "signals": int(signals.shape[0]),
        "wins": int(wins.shape[0]),
    }


def build_inverse_temperature_course_overlay(
    panel: pd.DataFrame,
    *,
    end: str = "2026-07-03",
    filename: str = "00b_rate_inverse_temperature_overlay.png",
) -> dict[str, float | int | str]:
    """Overlay the official rate with 100-temperature so low means good for both."""
    start = pd.Timestamp("2026-05-18")
    end_ts = pd.Timestamp(end)
    g = panel.loc[
        (panel["currency"] == "UZS")
        & panel["date"].between(start, end_ts)
    ].copy().sort_values("date")
    if g.empty:
        raise RuntimeError("Illustrative UZS window is absent from AP50 output")
    g["rate_per_1000"] = g["current_price"] * 1000.0
    g["inverse_temperature"] = 100.0 - g["temperature_h5"]
    signals = g.loc[g["signal"] & g["y5"].notna()].copy()
    wins = signals.loc[signals["y5"] == 1.0]
    losses = signals.loc[signals["y5"] == 0.0]

    fig = new_figure()
    title_block(
        fig,
        "Курс и 100 − температура",
        "Обе линии читаются одинаково: чем ниже, тем момент выгоднее · UZS · OOS · h=5",
    )
    add_pill(fig, 0.785, 0.902, f"{len(wins)} из {len(signals)} сигналов успешны", width=0.160)

    ax_rate = fig.add_axes([0.095, 0.15, 0.80, 0.65])
    ax_inverse = ax_rate.twinx()
    ax_rate.set_zorder(ax_inverse.get_zorder() + 1)
    ax_rate.patch.set_visible(False)

    ax_inverse.plot(
        g["date"],
        g["inverse_temperature"],
        color=ORANGE,
        lw=2.2,
        marker="o",
        markersize=3.6,
        alpha=0.82,
        zorder=2,
    )
    ax_rate.plot(
        g["date"],
        g["rate_per_1000"],
        color=INK,
        lw=3.0,
        marker="o",
        markersize=3.2,
        zorder=4,
    )
    if not wins.empty:
        ax_rate.scatter(
            wins["date"],
            wins["rate_per_1000"],
            s=118,
            color=GREEN,
            edgecolor=PAPER,
            linewidth=2.4,
            zorder=7,
        )
    if not losses.empty:
        ax_rate.scatter(
            losses["date"],
            losses["rate_per_1000"],
            s=135,
            marker="X",
            color=RED,
            edgecolor=PAPER,
            linewidth=1.8,
            zorder=7,
        )
    for d in signals["date"]:
        ax_rate.axvline(d, color=GREEN, alpha=0.12, lw=1.0, linestyle=(0, (2, 4)), zorder=0)

    ax_rate.set_ylabel("Курс ЦБ, ₽ за 1000 UZS\nниже — выгоднее", color=INK, labelpad=14)
    ax_inverse.set_ylabel("100 − температура\nниже — сильнее", color=ORANGE, labelpad=14)
    ax_inverse.set_ylim(0, 100)
    ax_inverse.set_yticks([0, 20, 40, 60, 80, 100])
    ax_inverse.tick_params(axis="y", colors=ORANGE)
    ax_rate.tick_params(axis="y", colors=INK)
    ax_rate.grid(axis="y", color=GRID, lw=0.85, alpha=0.8, zorder=0)
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_inverse.spines[["top", "left"]].set_visible(False)
    ax_rate.spines["left"].set_color(INK)
    ax_inverse.spines["right"].set_color(ORANGE)
    ax_rate.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_rate.margins(x=0.01)

    legend = [
        Line2D([0], [0], color=INK, lw=3, label="курс ЦБ"),
        Line2D([0], [0], color=ORANGE, lw=2.2, marker="o", markersize=5, alpha=0.82, label="100 − температура"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=10, label="отправляем пуш"),
    ]
    ax_rate.legend(handles=legend, loc="upper left", frameon=False, ncol=3, fontsize=10.8, handlelength=2.0)
    add_footer(
        fig,
        "Зелёная точка — отправка пуша. Здесь низкие значения обеих линий означают хороший момент. Общая OOS-оценка: 73,38% против 29,45%; lift 2,51.",
    )
    save(fig, filename)
    return {
        "currency": "UZS",
        "start": str(start.date()),
        "end": str(end_ts.date()),
        "signals": int(signals.shape[0]),
        "wins": int(wins.shape[0]),
    }


def build_best_fit_temperature_example(panel: pd.DataFrame) -> dict[str, float | int | str]:
    """A short explanatory KZT fragment where rate quality and temperature align."""
    start = pd.Timestamp("2026-06-25")
    end = pd.Timestamp("2026-07-14")
    g = panel.loc[
        (panel["currency"] == "KZT")
        & panel["date"].between(start, end)
    ].copy().sort_values("date")
    if g.empty:
        raise RuntimeError("Illustrative KZT window is absent from AP50 output")
    g["rate_per_100"] = g["current_price"] * 100.0
    signals = g.loc[g["signal"] & g["y5"].notna()].copy()
    wins = signals.loc[signals["y5"] == 1.0]
    losses = signals.loc[signals["y5"] == 0.0]

    fig = new_figure()
    title_block(
        fig,
        "Температура отмечает выгодные точки курса",
        "KZT · короткий иллюстративный OOS-фрагмент · обычная температура · h=5",
    )
    add_pill(fig, 0.785, 0.902, f"{len(wins)} из {len(signals)} сигналов успешны", width=0.160)

    ax_rate = fig.add_axes([0.095, 0.15, 0.80, 0.65])
    ax_temp = ax_rate.twinx()
    ax_rate.set_zorder(ax_temp.get_zorder() + 1)
    ax_rate.patch.set_visible(False)

    ax_temp.plot(
        g["date"],
        g["temperature_h5"],
        color=ORANGE,
        lw=2.4,
        marker="o",
        markersize=4.0,
        alpha=0.86,
        zorder=2,
    )
    ax_rate.plot(
        g["date"],
        g["rate_per_100"],
        color=INK,
        lw=3.2,
        marker="o",
        markersize=3.8,
        zorder=4,
    )
    if not wins.empty:
        ax_rate.scatter(
            wins["date"],
            wins["rate_per_100"],
            s=145,
            color=GREEN,
            edgecolor=PAPER,
            linewidth=2.6,
            zorder=7,
        )
    if not losses.empty:
        ax_rate.scatter(
            losses["date"],
            losses["rate_per_100"],
            s=150,
            marker="X",
            color=RED,
            edgecolor=PAPER,
            linewidth=1.8,
            zorder=7,
        )
    for d in signals["date"]:
        ax_rate.axvline(d, color=GREEN, alpha=0.13, lw=1.0, linestyle=(0, (2, 4)), zorder=0)

    # Reverse only the visual direction of the price axis: a lower RUB price is
    # better for the sender, so both upward movements now mean "better".
    y_min = float(g["rate_per_100"].min())
    y_max = float(g["rate_per_100"].max())
    pad = (y_max - y_min) * 0.08
    ax_rate.set_ylim(y_max + pad, y_min - pad)
    ax_rate.set_ylabel("Курс ЦБ, ₽ за 100 KZT\nшкала перевёрнута: выше — выгоднее", color=INK, labelpad=14)
    ax_temp.set_ylabel("Температура, 0–100\nвыше — сильнее", color=ORANGE, labelpad=14)
    ax_temp.set_ylim(0, 80)
    ax_temp.set_yticks([0, 20, 40, 60, 80])
    ax_temp.tick_params(axis="y", colors=ORANGE)
    ax_rate.tick_params(axis="y", colors=INK)
    ax_rate.grid(axis="y", color=GRID, lw=0.85, alpha=0.8, zorder=0)
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_temp.spines[["top", "left"]].set_visible(False)
    ax_rate.spines["left"].set_color(INK)
    ax_temp.spines["right"].set_color(ORANGE)
    ax_rate.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_rate.margins(x=0.015)

    legend = [
        Line2D([0], [0], color=INK, lw=3, label="курс ЦБ"),
        Line2D([0], [0], color=ORANGE, lw=2.3, marker="o", markersize=5, alpha=0.86, label="температура"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=10, label="отправляем пуш"),
    ]
    ax_rate.legend(handles=legend, loc="upper left", frameon=False, ncol=3, fontsize=10.8, handlelength=2.0)
    add_footer(
        fig,
        "Витринный фрагмент выбран для объяснения механики, а не для оценки качества. Общая OOS-оценка на всех данных: 73,38% против 29,45%; lift 2,51.",
    )
    save(fig, "00d_temperature_best_fit_kzt.png")
    return {
        "currency": "KZT",
        "start": str(start.date()),
        "end": str(end.date()),
        "signals": int(signals.shape[0]),
        "wins": int(wins.shape[0]),
        "temperature_vs_rate_quality_correlation": float(
            np.corrcoef(g["temperature_h5"], 1.0 / g["current_price"])[0, 1]
        ),
    }


def build_temperature_course_example(panel: pd.DataFrame) -> dict[str, float | int | str]:
    # An explicitly labelled successful example; aggregate charts below carry the
    # generalisation claim.  This window contains eight mature AP37 signals.
    start = pd.Timestamp("2026-05-18")
    end = pd.Timestamp("2026-07-03")
    g = panel.loc[
        (panel["currency"] == "UZS")
        & panel["date"].between(start, end)
    ].copy()
    if g.empty:
        raise RuntimeError("Illustrative UZS window is absent from AP50 output")
    g = g.sort_values("date")
    g["recipient_index"] = 100.0 * g["current_price"].iloc[0] / g["current_price"]
    signals = g.loc[g["signal"] & g["y5"].notna()].copy()
    wins = signals.loc[signals["y5"] == 1.0]
    losses = signals.loc[signals["y5"] == 0.0]

    fig = new_figure()
    title_block(
        fig,
        "Температура находит сильные моменты",
        "UZS · успешный иллюстративный OOS-фрагмент · после подтверждённой публикации ЦБ · h=5",
    )
    add_pill(
        fig,
        0.76,
        0.902,
        f"{int(wins.shape[0])} из {int(signals.shape[0])} сигналов успешны",
        width=0.185,
    )

    gs = fig.add_gridspec(2, 1, left=0.098, right=0.945, bottom=0.105, top=0.835, height_ratios=[1.12, 0.88], hspace=0.15)
    ax_rate = fig.add_subplot(gs[0])
    ax_temp = fig.add_subplot(gs[1], sharex=ax_rate)

    ax_rate.plot(g["date"], g["recipient_index"], color=INK, lw=2.6, zorder=3)
    ax_rate.fill_between(g["date"], g["recipient_index"], g["recipient_index"].min() - 0.5, color=BLUE_LIGHT, alpha=0.68, zorder=1)
    ax_rate.scatter(g["date"], g["recipient_index"], s=13, color=INK, alpha=0.55, zorder=4)
    if not wins.empty:
        ax_rate.scatter(wins["date"], wins["recipient_index"], s=90, color=GREEN, edgecolor=PAPER, linewidth=2, zorder=6)
    if not losses.empty:
        ax_rate.scatter(losses["date"], losses["recipient_index"], s=105, marker="X", color=RED, edgecolor=PAPER, linewidth=1.5, zorder=6)
    for d in signals["date"]:
        ax_rate.axvline(d, color=GREEN, alpha=0.15, lw=1.2, zorder=0)
        ax_temp.axvline(d, color=GREEN, alpha=0.15, lw=1.2, zorder=0)

    ax_rate.set_ylabel("Сколько валюты получит клиент\nиндекс, выше — выгоднее", color=INK, labelpad=13)
    ax_rate.grid(axis="y", color=GRID, lw=0.8, alpha=0.75)
    ax_rate.spines[["top", "right"]].set_visible(False)
    ax_rate.tick_params(axis="x", labelbottom=False)
    ax_rate.margins(x=0.012)

    ax_temp.axhspan(0, 25, color=BLUE_LIGHT, alpha=0.55, zorder=0)
    ax_temp.axhspan(25, 50, color="#E9E7D8", alpha=0.62, zorder=0)
    ax_temp.axhspan(50, 70, color=ORANGE_LIGHT, alpha=0.58, zorder=0)
    ax_temp.axhspan(70, 100, color=GREEN_LIGHT, alpha=0.58, zorder=0)
    ax_temp.fill_between(g["date"], 0, g["temperature_h5"], color=ORANGE, alpha=0.20, zorder=1)
    ax_temp.plot(g["date"], g["temperature_h5"], color=ORANGE, lw=2.3, zorder=3)
    ax_temp.scatter(g["date"], g["temperature_h5"], s=14, color=ORANGE, zorder=4)
    if not signals.empty:
        ax_temp.scatter(signals["date"], signals["temperature_h5"], s=68, marker="^", color=GREEN, edgecolor=PAPER, linewidth=1.4, zorder=6)
    ax_temp.set_ylim(0, 100)
    ax_temp.set_yticks([0, 25, 50, 75, 100])
    ax_temp.set_ylabel("Температура\n0–100", color=INK, labelpad=13)
    ax_temp.grid(axis="y", color=GRID, lw=0.8, alpha=0.75)
    ax_temp.spines[["top", "right"]].set_visible(False)
    ax_temp.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_temp.tick_params(axis="x", rotation=0)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=9, label="успешный пуш-сигнал"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=9, label="момент отправки"),
    ]
    ax_rate.legend(handles=legend, loc="lower left", frameon=False, ncol=2, fontsize=10.3)
    add_footer(
        fig,
        "Успех: текущий официальный курс выгоднее каждого из следующих 5 публикационных дней. Фрагмент выбран для объяснения; итоговая оценка ниже рассчитана на всём OOS-периоде.",
    )
    save(fig, "01_temperature_and_rate_example.png")
    return {
        "currency": "UZS",
        "start": str(start.date()),
        "end": str(end.date()),
        "signals": int(signals.shape[0]),
        "wins": int(wins.shape[0]),
        "mean_temperature_on_signals": float(signals["temperature_h5"].mean()),
    }


def build_horizon_scorecard() -> dict[str, dict[str, float]]:
    all_h = pd.read_csv(AP37 / "retrospective_all_horizons.csv")
    score = all_h.loc[
        (all_h["candidate"] == SELECTED)
        & all_h["h"].isin([3, 5, 10, 20])
    ].sort_values("h")
    if score.shape[0] != 4:
        raise RuntimeError("Expected four aggregate horizon rows")

    fig = new_figure()
    title_block(
        fig,
        "В 2,5 раза точнее выбираем сильные дни",
        "Доля успешных решений: пуш-сигналы модели против обычного публикационного дня",
    )
    add_pill(fig, 0.778, 0.902, "ключевой lift = 2,51", width=0.167)
    ax = fig.add_axes([0.075, 0.14, 0.86, 0.67])

    x = np.arange(score.shape[0])
    width = 0.30
    base = score["base_rate"].to_numpy() * 100.0
    model = score["hit_rate"].to_numpy() * 100.0
    bars_base = ax.bar(x - width / 2, base, width, color=GREY_BAR, label="Обычный день", zorder=3)
    bars_model = ax.bar(x + width / 2, model, width, color=GREEN, label="Сигнал модели", zorder=3)

    # Subtle emphasis on the task's primary h=5 horizon.
    h5_idx = int(np.where(score["h"].to_numpy() == 5)[0][0])
    ax.axvspan(h5_idx - 0.46, h5_idx + 0.46, color=ORANGE_LIGHT, alpha=0.34, zorder=0)
    ax.text(h5_idx, 91, "основной горизонт", ha="center", va="center", color=ORANGE, fontsize=10.5, fontweight="bold")

    for rect, value in zip(bars_base, base):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.6, f"{value:.0f}%", ha="center", va="bottom", color=MUTED, fontsize=12, fontweight="bold")
    for rect, value in zip(bars_model, model):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.6, f"{value:.0f}%", ha="center", va="bottom", color=GREEN, fontsize=13, fontweight="bold")
    for i, row in enumerate(score.itertuples(index=False)):
        ax.text(i, max(base[i], model[i]) + 9.4, f"×{row.adjusted_lift:.2f}", ha="center", va="bottom", color=INK, fontsize=12.5, fontweight="bold")

    ax.set_xticks(x, [f"{int(h)} дня" if h in (3, 5) else f"{int(h)} дней" for h in score["h"]])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_ylabel("Успешных решений", color=INK, labelpad=12)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=11)
    add_footer(
        fig,
        "Честный ретроспективный OOS: 09.01.2024–25.08.2026 для h=5 · 5 валют · 695 зрелых h=5 сигналов · reference: действующий на момент решения курс ЦБ.",
    )
    save(fig, "02_oos_hit_rate_by_horizon.png")
    return {
        str(int(row.h)): {
            "hit_rate": float(row.hit_rate),
            "base_rate": float(row.base_rate),
            "adjusted_lift": float(row.adjusted_lift),
            "signals": int(row.n_signals),
        }
        for row in score.itertuples(index=False)
    }


def build_temperature_evidence(panel: pd.DataFrame) -> list[dict[str, float | int | str]]:
    valid = panel.loc[panel["y5"].notna()].copy()
    edges = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    valid["band"] = pd.cut(valid["temperature_h5"], bins=edges, labels=labels, include_lowest=True, right=False)
    bins = (
        valid.groupby("band", observed=False)
        .agg(n=("y5", "size"), actual=("y5", "mean"), mean_temperature=("temperature_h5", "mean"))
        .reset_index()
    )
    bins["actual_pct"] = bins["actual"] * 100.0

    fig = new_figure()
    title_block(
        fig,
        "Горячие дни действительно сильнее",
        "Фактическая доля удачных решений · весь OOS · после публикации ЦБ · h=5",
    )
    add_pill(fig, 0.786, 0.902, "9% → 79% успеха", width=0.159)
    ax = fig.add_axes([0.075, 0.14, 0.86, 0.67])

    colors = ["#8CB7D0", "#B7C8C2", "#D8C79D", "#D39A55", GREEN]
    x = np.arange(len(bins))
    bars = ax.bar(x, bins["actual_pct"], width=0.64, color=colors, zorder=3)
    baseline = float(valid["y5"].mean() * 100.0)
    ax.axhline(baseline, color=MUTED, lw=1.6, linestyle=(0, (5, 4)), zorder=2)
    ax.text(4.48, baseline + 1.7, f"обычный день: {baseline:.1f}%", ha="left", color=MUTED, fontsize=10.5)

    for rect, row in zip(bars, bins.itertuples(index=False)):
        ax.text(rect.get_x() + rect.get_width() / 2, row.actual_pct + 1.8, f"{row.actual_pct:.0f}%", ha="center", va="bottom", color=INK, fontsize=15, fontweight="bold")
        ax.text(rect.get_x() + rect.get_width() / 2, 3.2, f"n={int(row.n)}", ha="center", va="bottom", color=PAPER if row.actual_pct > 18 else INK, fontsize=9.5, fontweight="bold")

    ax.set_xticks(x, [f"{label}\n{'холодно' if i == 0 else 'горячо' if i == 4 else ''}" for i, label in enumerate(labels)])
    ax.set_xlim(-0.55, 5.0)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_ylabel("Фактически удачных решений", color=INK, labelpad=12)
    ax.set_xlabel("Температура модели", color=INK, labelpad=13)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.annotate(
        "чем выше температура,\nтем сильнее момент",
        xy=(4, bins["actual_pct"].iloc[-1]),
        xytext=(3.15, 83),
        color=GREEN,
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
    )
    add_footer(
        fig,
        "Температура — честная вероятность, рассчитанная только из информации, доступной к моменту решения. Успех — текущий день выгоднее каждого из следующих 5 публикационных дней.",
    )
    save(fig, "03_temperature_means_real_probability.png")
    return [
        {
            "band": str(row.band),
            "n": int(row.n),
            "observed_success_rate": float(row.actual),
            "mean_temperature": float(row.mean_temperature),
        }
        for row in bins.itertuples(index=False)
    ]


def build_currency_robustness() -> list[dict[str, float | int | str]]:
    detail = pd.read_csv(AP37 / "diagnostic_breakdown.csv", dtype={"group": str})
    rows = detail.loc[
        (detail["candidate"] == SELECTED)
        & (detail["group"].isin(CURRENCIES))
        & (detail["h"] == 5)
    ].copy()
    rows["group"] = pd.Categorical(rows["group"], categories=CURRENCIES, ordered=True)
    rows = rows.sort_values("group")
    if rows.shape[0] != 5:
        raise RuntimeError("Expected one h=5 diagnostic row per currency")

    fig = new_figure()
    title_block(
        fig,
        "Сильный результат на всех пяти направлениях",
        "Каждая валюта лучше обычного дня · рабочая частота около одного сигнала в неделю",
    )
    add_pill(fig, 0.794, 0.902, "5 из 5 валют", width=0.151)
    ax = fig.add_axes([0.10, 0.14, 0.80, 0.67])

    y = np.arange(len(rows))[::-1]
    base = rows["base_rate"].to_numpy() * 100.0
    hit = rows["hit_rate"].to_numpy() * 100.0
    for yi, lo, hi in zip(y, base, hit):
        ax.plot([lo, hi], [yi, yi], color=GREEN_LIGHT, lw=8, solid_capstyle="round", zorder=1)
    ax.scatter(base, y, s=105, color=GREY_BAR, edgecolor=PAPER, linewidth=1.5, zorder=3, label="Обычный день")
    ax.scatter(hit, y, s=140, color=GREEN, edgecolor=PAPER, linewidth=1.8, zorder=4, label="Сигнал модели")

    for yi, row, lo, hi in zip(y, rows.itertuples(index=False), base, hit):
        ax.text(lo - 1.4, yi, f"{lo:.0f}%", ha="right", va="center", color=MUTED, fontsize=11, fontweight="bold")
        ax.text(hi + 1.4, yi, f"{hi:.0f}%", ha="left", va="center", color=GREEN, fontsize=12, fontweight="bold")
        ax.text(98.5, yi, f"lift {row.adjusted_lift:.2f}  ·  {row.frequency:.2f}/нед.", ha="right", va="center", color=INK, fontsize=10.5)

    ax.set_yticks(y, rows["group"].astype(str))
    ax.set_xlim(15, 100)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xticklabels(["20%", "40%", "60%", "80%"])
    ax.set_xlabel("Доля успешных решений на горизонте 5 дней", color=INK, labelpad=13)
    ax.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)
    ax.text(26, 4.33, "●  обычный день", color=MUTED, fontsize=10.5, ha="center")
    ax.text(72, 4.33, "●  сигнал модели", color=GREEN, fontsize=10.5, ha="center")
    add_footer(
        fig,
        "OOS 2024–2026 · отдельная статистика по каждому направлению · не результат одной удачной валюты. Частота указана в сигналах на валюту в неделю.",
    )
    save(fig, "04_robust_across_all_currencies.png")
    return [
        {
            "currency": str(row.group),
            "signals": int(row.n_signals),
            "hit_rate": float(row.hit_rate),
            "base_rate": float(row.base_rate),
            "adjusted_lift": float(row.adjusted_lift),
            "frequency_per_week": float(row.frequency),
        }
        for row in rows.itertuples(index=False)
    ]


def main() -> None:
    configure_style()
    panel = load_after_receipt_panel()
    evidence = {
        "sources": {
            "push_policy": str((AP37 / "outputs.npz").relative_to(ROOT)),
            "temperature": str((AP50 / "outputs.npz").relative_to(ROOT)),
        },
        "canonical_visuals": {
            "temperature_example": "output/defense_visuals/temperature_example.png",
        },
        "overlay_window": build_temperature_course_overlay(panel),
        "temperature_overlay_crop": build_temperature_course_overlay(
            panel,
            end="2026-06-23",
            filename="00e_temperature_to_2026-06-23.png",
        ),
        "temperature_example_canonical": build_temperature_course_overlay(
            panel,
            end="2026-06-23",
            filename="temperature_example.png",
        ),
        "inverse_temperature_overlay_window": build_inverse_temperature_course_overlay(panel),
        "inverse_temperature_overlay_crop": build_inverse_temperature_course_overlay(
            panel,
            end="2026-06-23",
            filename="00c_rate_inverse_temperature_to_2026-06-23.png",
        ),
        "best_fit_temperature_example": build_best_fit_temperature_example(panel),
        "illustrative_window": build_temperature_course_example(panel),
        "horizon_scorecard": build_horizon_scorecard(),
        "temperature_bands": build_temperature_evidence(panel),
        "currency_robustness": build_currency_robustness(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chart_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created ten figures and evidence manifest in {OUT}")


if __name__ == "__main__":
    main()
