"""Derived causal daily microstructure from the last completed CNYRUB_TOM session."""
from __future__ import annotations

import datetime as dt

import numpy as np

from research.round6_moex_features import MAX_STALENESS_DAYS


WINDOWS = (5, 20, 60)


def _bps(a, b):
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def _session_arrays(history):
    rows = history["CNYRUB_TOM"]
    dates = np.asarray([row["date"] for row in rows], dtype=object)
    raw = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ("open", "high", "low", "close", "waprice", "trades")
    }
    n = len(rows)
    values = {name: np.zeros(n, dtype=float) for name in (
        "pressure", "pressure_efficiency", "close_location", "wap_location",
        "body", "wick_asymmetry", "overnight_gap", "intraday_range",
        "log_trades", "agree_up", "agree_down", "disagree",
    )}
    for i in range(n):
        open_, high, low, close, wap = (
            raw[key][i] for key in ("open", "high", "low", "close", "waprice")
        )
        width = max(high - low, 1e-12)
        oc = _bps(close, open_)
        cw = _bps(close, wap) if np.isfinite(wap) else 0.0
        intraday = _bps(high, low)
        pressure = oc + cw
        values["pressure"][i] = pressure
        values["pressure_efficiency"][i] = np.clip(pressure / max(intraday, 1.0), -5.0, 5.0)
        values["close_location"][i] = np.clip(2.0 * (close - low) / width - 1.0, -1.0, 1.0)
        values["wap_location"][i] = (
            np.clip(2.0 * (wap - low) / width - 1.0, -1.0, 1.0)
            if np.isfinite(wap) else 0.0
        )
        values["body"][i] = np.clip(abs(close - open_) / width, 0.0, 1.0)
        upper = (high - max(open_, close)) / width
        lower = (min(open_, close) - low) / width
        values["wick_asymmetry"][i] = np.clip(upper - lower, -1.0, 1.0)
        values["overnight_gap"][i] = _bps(open_, raw["close"][i - 1]) if i else 0.0
        values["intraday_range"][i] = intraday
        values["log_trades"][i] = float(np.log1p(raw["trades"][i]))
        values["agree_up"][i] = float(oc > 0 and cw > 0)
        values["agree_down"][i] = float(oc < 0 and cw < 0)
        values["disagree"][i] = float(oc * cw < 0)
    return dates, values


def build_microstructure_features(index, history):
    dates, sessions = _session_arrays(history)
    base_names = (
        "pressure", "pressure_efficiency", "close_location", "wap_location",
        "body", "wick_asymmetry", "agree_up", "agree_down", "disagree",
    )
    z_names = ("pressure", "intraday_range", "close_location", "overnight_gap", "log_trades")
    result, names = [], None
    for _currency, _position, day in index:
        end = int(np.searchsorted(dates, day, side="left"))
        source = dates[end - 1] if end else None
        age = (day - source).days if source is not None else 999
        available = bool(end and age <= MAX_STALENESS_DAYS)
        values, row_names = [], []
        for name in base_names:
            values.append(float(sessions[name][end - 1]) if available else 0.0)
            row_names.append(f"cny_micro_{name}")
        for name in z_names:
            history_values = sessions[name][:end]
            for window in WINDOWS:
                recent = history_values[-window:]
                if available and len(recent) >= max(3, window // 4):
                    mean, std = float(np.mean(recent)), float(np.std(recent))
                    z = float((recent[-1] - mean) / std) if std > 1e-12 else 0.0
                else:
                    z = 0.0
                values.append(z); row_names.append(f"cny_micro_{name}_z_{window}")
        values.extend([float(min(age, 30)), float(not available)])
        row_names.extend(["cny_micro_age_days", "cny_micro_missing"])
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("CNY microstructure schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite CNY microstructure feature")
    return matrix, names or []


def causality_check(index, history, cutoff=dt.date(2025, 6, 30)):
    full, names = build_microstructure_features(index, history)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["date"] >= cutoff:
                for key in ("open", "high", "low", "close", "waprice", "trades"):
                    if np.isfinite(clone[key]):
                        clone[key] *= 100.0
            changed[ticker].append(clone)
    corrupted, changed_names = build_microstructure_features(index, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != changed_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future session changed a past microstructure feature")
    return True
