"""Past-only feature engineering and target creation for lift tests."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from transfer_lift.data import CORRIDORS, REFERENCE, Series

RETURN_LAGS: tuple[int, ...] = (1, 3, 5, 10, 20, 60)
RANGE_WINDOWS: tuple[int, ...] = (30, 90, 180)
VOL_WINDOWS: tuple[int, ...] = (10, 30, 90)
WARMUP = max(max(RETURN_LAGS), max(RANGE_WINDOWS), max(VOL_WINDOWS)) + 1


def client_bps(new: float, old: float) -> float:
    """Client-positive rate move in basis points: lower RUB/FX is better."""
    return -(new - old) / old * 10_000.0 if old > 0 else 0.0


def _position_in_range(past: np.ndarray, window: int) -> float:
    values = past[-window:]
    lo, hi = float(values.min()), float(values.max())
    return 50.0 if hi <= lo else (float(values[-1]) - lo) / (hi - lo) * 100.0


def _days_beaten(past: np.ndarray, window: int) -> float:
    """Share of previous days in the window that had worse, higher RUB/FX rate."""
    values = past[-window:]
    if len(values) < 2:
        return 50.0
    return float((values[:-1] > values[-1]).mean()) * 100.0


def _streaks(past: np.ndarray) -> tuple[float, float]:
    down = up = 0
    for idx in range(len(past) - 1, 0, -1):
        if past[idx] < past[idx - 1]:
            if up:
                break
            down += 1
        elif past[idx] > past[idx - 1]:
            if down:
                break
            up += 1
        else:
            break
        if down + up >= 10:
            break
    return float(down), float(up)


def _row_features(
    past: np.ndarray,
    day: dt.date,
    corridor: str,
    ref_past: dict[str, np.ndarray],
) -> dict[str, float]:
    current = float(past[-1])
    row: dict[str, float] = {}

    for lag in RETURN_LAGS:
        row[f"ret_{lag}"] = client_bps(current, float(past[-1 - lag])) if len(past) > lag else 0.0

    down, up = _streaks(past)
    row["streak_down"] = down
    row["streak_up"] = up

    for window in RANGE_WINDOWS:
        if len(past) >= window:
            values = past[-window:]
            row[f"pct_range_{window}"] = _position_in_range(past, window)
            row[f"days_beaten_{window}"] = _days_beaten(past, window)
            row[f"z_{window}"] = (current - float(values.mean())) / float(values.std()) if values.std() > 0 else 0.0
            row[f"dist_min_bps_{window}"] = client_bps(current, float(values.min()))
            row[f"dist_max_bps_{window}"] = client_bps(current, float(values.max()))
        else:
            row[f"pct_range_{window}"] = 50.0
            row[f"days_beaten_{window}"] = 50.0
            row[f"z_{window}"] = 0.0
            row[f"dist_min_bps_{window}"] = 0.0
            row[f"dist_max_bps_{window}"] = 0.0

    returns = np.diff(past) / past[:-1] * 10_000.0 if len(past) > 1 else np.array([0.0])
    for window in VOL_WINDOWS:
        row[f"vol_{window}"] = float(returns[-window:].std()) if len(returns) >= window else 0.0
    row["vol_ratio_10_90"] = row["vol_10"] / row["vol_90"] if row["vol_90"] > 0 else 1.0

    for code, ref in ref_past.items():
        lower = code.lower()
        row[f"{lower}_ret_5"] = client_bps(float(ref[-1]), float(ref[-6])) if len(ref) > 5 else 0.0
        row[f"{lower}_ret_20"] = client_bps(float(ref[-1]), float(ref[-21])) if len(ref) > 20 else 0.0
    row["excess_ret_5_vs_usd"] = row["ret_5"] - row.get("usd_ret_5", 0.0)
    row["excess_ret_20_vs_usd"] = row["ret_20"] - row.get("usd_ret_20", 0.0)

    row["dow"] = float(day.weekday())
    row["dom"] = float(day.day)
    row["month"] = float(day.month)
    row["quarter"] = float((day.month - 1) // 3 + 1)
    row["is_month_start"] = 1.0 if day.day <= 5 else 0.0
    row["is_month_end"] = 1.0 if day.day >= 26 else 0.0
    row["corridor_id"] = float(CORRIDORS.index(corridor))
    return row


def target_now_favourable(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Message truth target: today should not be beaten by a lower rate within horizon."""
    if idx + horizon >= len(values):
        return None
    return 1.0 if values[idx] <= values[idx + 1 : idx + horizon + 1].min() else 0.0


def target_window_closing(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Window-closing target: rate later becomes higher than today."""
    if idx + horizon >= len(values):
        return None
    return 1.0 if values[idx + horizon] > values[idx] else 0.0


def forward_benefit_bps(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Benefit available to the client versus the next horizon observations."""
    if idx + horizon >= len(values):
        return None
    future_mean = float(values[idx + 1 : idx + horizon + 1].mean())
    return client_bps(float(values[idx]), future_mean) if future_mean > 0 else None


def target_published_next_favourable(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """As-of CBR target: tomorrow's published rate remains unbeaten for horizon rows."""
    if idx + horizon + 1 >= len(values):
        return None
    published_next = float(values[idx + 1])
    return 1.0 if published_next <= values[idx + 2 : idx + horizon + 2].min() else 0.0


def published_next_benefit_bps(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Forward benefit of the rate already published for the next effective date."""
    if idx + horizon + 1 >= len(values):
        return None
    future_mean = float(values[idx + 2 : idx + horizon + 2].mean())
    return client_bps(float(values[idx + 1]), future_mean) if future_mean > 0 else None


def published_next_features(values: np.ndarray, idx: int) -> dict[str, float]:
    """Features based on tomorrow's CBR rate, known today according to the case."""
    published_next = float(values[idx + 1])
    current = float(values[idx])
    window = values[max(0, idx + 2 - 90) : idx + 2]
    lo, hi = float(window.min()), float(window.max())
    pct_range = 50.0 if hi <= lo else (published_next - lo) / (hi - lo) * 100.0
    days_beaten = float((window[:-1] > published_next).mean()) * 100.0 if len(window) > 1 else 50.0
    return {
        "published_next_rate": published_next,
        "published_next_ret_1": client_bps(published_next, current),
        "published_next_pct_range_90": pct_range,
        "published_next_days_beaten_90": days_beaten,
    }


def build_dataset(
    series: dict[str, Series],
    corridors: tuple[str, ...] = CORRIDORS,
    refs: tuple[str, ...] = REFERENCE,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build one row per corridor/date using only data available by that date."""
    ref_dates = {code: np.array(series[code].dates, dtype=object) for code in refs}
    rows: list[dict[str, object]] = []
    for corridor in corridors:
        current_series = series[corridor]
        for idx in range(WARMUP, len(current_series.values) - horizon - 1):
            day = current_series.dates[idx]
            ref_past: dict[str, np.ndarray] = {}
            for ref in refs:
                ref_idx = int(np.searchsorted(ref_dates[ref], day, side="right")) - 1
                ref_past[ref] = series[ref].values[: ref_idx + 1]
            features = _row_features(current_series.values[: idx + 1], day, corridor, ref_past)
            features.update(published_next_features(current_series.values, idx))
            features.update(
                {
                    "date": pd.Timestamp(day),
                    "corridor": corridor,
                    "row_idx": idx,
                    "rate": float(current_series.values[idx]),
                    "target_fav": target_now_favourable(current_series.values, idx, horizon),
                    "target_close": target_window_closing(current_series.values, idx, horizon),
                    "target_pub_fav": target_published_next_favourable(
                        current_series.values, idx, horizon
                    ),
                    "benefit_bps": forward_benefit_bps(current_series.values, idx, horizon),
                    "published_next_benefit_bps": published_next_benefit_bps(
                        current_series.values, idx, horizon
                    ),
                }
            )
            rows.append(features)
    frame = pd.DataFrame(rows).sort_values(["date", "corridor"]).reset_index(drop=True)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns safe for model training."""
    blocked = {
        "date",
        "corridor",
        "row_idx",
        "rate",
        "target_fav",
        "target_close",
        "target_pub_fav",
        "benefit_bps",
        "published_next_benefit_bps",
    }
    return [column for column in frame.columns if column not in blocked]
