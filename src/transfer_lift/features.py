"""Past-only feature engineering and target creation for lift tests."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from transfer_lift.calendar_ref import (
    CORRIDOR_HOLIDAYS,
    HOLIDAYS,
    days_since_payday,
    days_since_prev,
    days_to_next,
    days_to_payday,
)
from transfer_lift.data import CORRIDORS, REFERENCE, Series

RETURN_LAGS: tuple[int, ...] = (1, 3, 5, 10, 20, 60)
RANGE_WINDOWS: tuple[int, ...] = (30, 90, 180)
VOL_WINDOWS: tuple[int, ...] = (10, 30, 90)
HOLIDAY_NAMES: tuple[str, ...] = tuple(HOLIDAYS)
WARMUP = max(*RETURN_LAGS, *RANGE_WINDOWS, *VOL_WINDOWS) + 1


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
        prefix_values = past[-window:]
        if len(past) < window:
            row.update(
                {
                    f"pct_range_{window}": 50.0,
                    f"days_beaten_{window}": 50.0,
                    f"z_{window}": 0.0,
                    f"dist_min_bps_{window}": 0.0,
                    f"dist_max_bps_{window}": 0.0,
                }
            )
            continue

        mean = float(prefix_values.mean())
        std = float(prefix_values.std())
        row.update(
            {
                f"pct_range_{window}": _position_in_range(past, window),
                f"days_beaten_{window}": _days_beaten(past, window),
                f"z_{window}": (current - mean) / std if std > 0 else 0.0,
                f"dist_min_bps_{window}": client_bps(current, float(prefix_values.min())),
                f"dist_max_bps_{window}": client_bps(current, float(prefix_values.max())),
            }
        )

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

    # Holiday dates are known in advance, so these are past-safe calendar features.
    to_payday = days_to_payday(day)
    since_payday = days_since_payday(day)
    row["days_to_payday"] = float(to_payday)
    row["days_since_payday"] = float(since_payday)
    row["in_payday_window"] = float(min(to_payday, since_payday) <= 3)
    relevant_holidays = CORRIDOR_HOLIDAYS.get(corridor, ())
    for name in HOLIDAY_NAMES:
        relevant = name in relevant_holidays
        row[f"to_{name}"] = float(days_to_next(day, HOLIDAYS[name])) if relevant else 999.0
        row[f"since_{name}"] = float(days_since_prev(day, HOLIDAYS[name])) if relevant else 999.0
    row["to_any_holiday"] = min(
        (float(days_to_next(day, HOLIDAYS[name])) for name in relevant_holidays),
        default=999.0,
    )
    row["pre_holiday_14d"] = float(row["to_any_holiday"] <= 14)
    return row


def target_now_favourable(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Message truth target: today should not be beaten by a lower rate within horizon."""
    if idx + horizon >= len(values):
        return None
    return 1.0 if values[idx] <= values[idx + 1 : idx + horizon + 1].min() else 0.0


def target_local_minimum(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Case-literal target: today is a local minimum inside a SYMMETRIC +-h window.

    The case asks models to decide "was today a local minimum of the rate within a
    +-h day window", not only a forward-looking favourable day. Both sides of the
    window are needed here: values[idx-h : idx+h+1]. This target legitimately looks
    both backward and forward because it is a LABEL, not a feature; see the module
    docstring and feature_columns() for the leakage boundary that still applies to
    features built from row_features strictly on values[:idx+1].
    """
    if idx - horizon < 0 or idx + horizon >= len(values):
        return None
    window = values[idx - horizon : idx + horizon + 1]
    return 1.0 if values[idx] <= window.min() else 0.0


def symmetric_benefit_bps(values: np.ndarray, idx: int, horizon: int) -> float | None:
    """Case-literal benefit metric: today's rate vs the mean rate in a +-h window.

    Matches "Выгода момента в базисных пунктах" from the case ("насколько курс в день
    сигнала лучше среднего курса в окне +-h дней"). Unlike forward_benefit_bps, this
    compares against BOTH past and future days, so it also credits a signal fired
    right after the move already happened. Use forward_benefit_bps to see how much of
    that benefit a client could still capture going forward.
    """
    if idx - horizon < 0 or idx + horizon >= len(values):
        return None
    window_mean = float(values[idx - horizon : idx + horizon + 1].mean())
    return client_bps(float(values[idx]), window_mean) if window_mean > 0 else None


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
            row: dict[str, object] = {
                **features,
                "date": pd.Timestamp(day),
                "corridor": corridor,
                "row_idx": idx,
                "rate": float(current_series.values[idx]),
                "target_fav": target_now_favourable(current_series.values, idx, horizon),
                "target_close": target_window_closing(current_series.values, idx, horizon),
                "target_local_min": target_local_minimum(current_series.values, idx, horizon),
                "benefit_bps": forward_benefit_bps(current_series.values, idx, horizon),
                "symmetric_benefit_bps": symmetric_benefit_bps(
                    current_series.values, idx, horizon
                ),
            }
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["date", "corridor"]).reset_index(drop=True)
    return frame


def feature_columns(frame: pd.DataFrame, target_col: str = "target_fav") -> list[str]:
    """Return model inputs, excluding identifiers and all future-derived labels."""
    blocked = {
        "date",
        "corridor",
        "row_idx",
        "rate",
        "target_fav",
        "target_close",
        "target_local_min",
        "benefit_bps",
        "symmetric_benefit_bps",
    }
    return [column for column in frame.columns if column not in blocked]
