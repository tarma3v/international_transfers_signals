"""Causal spectral/path compression of 20 completed CNYRUB_TOM sessions."""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.fft import dct

from research.round6_moex_features import MAX_STALENESS_DAYS


WINDOW = 20
DCT_COMPONENTS = 8


def _summary(returns):
    values, names = [], []
    for window in (5, 10, 20):
        recent = returns[-window:]
        values.extend([float(np.mean(recent)), float(np.std(recent))])
        names.extend([f"cny_wave_mean_{window}", f"cny_wave_vol_{window}"])
    downside = np.minimum(returns, 0.0)
    upside = np.maximum(returns, 0.0)
    values.extend([
        float(np.sqrt(np.mean(downside * downside))),
        float(np.sqrt(np.mean(upside * upside))),
    ])
    names.extend(["cny_wave_downside_vol_20", "cny_wave_upside_vol_20"])
    centered = returns - np.mean(returns)
    std = float(np.std(returns))
    skew = float(np.mean(centered ** 3) / std ** 3) if std > 1e-12 else 0.0
    autocorrelation = (
        float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        if np.std(returns[:-1]) > 1e-12 and np.std(returns[1:]) > 1e-12
        else 0.0
    )
    values.extend([skew, autocorrelation])
    names.extend(["cny_wave_skew_20", "cny_wave_autocorr_1_20"])
    for window in (5, 10, 20):
        values.append(float(np.mean(returns[-window:] > 0.0)))
        names.append(f"cny_wave_positive_fraction_{window}")
    sign = np.sign(returns)
    values.append(float(np.mean(sign[1:] != sign[:-1])))
    names.append("cny_wave_sign_flip_fraction_20")
    path = np.r_[0.0, np.cumsum(returns)]
    drawdown = float(np.max(np.maximum.accumulate(path) - path))
    runup = float(np.max(path - np.minimum.accumulate(path)))
    values.extend([
        drawdown,
        runup,
        float(np.argmin(path) / WINDOW),
        float(np.argmax(path) / WINDOW),
        float((returns[-1] - np.mean(returns)) / std) if std > 1e-12 else 0.0,
        float(np.mean(returns[-5:]) - np.mean(returns[-10:-5])),
    ])
    names.extend([
        "cny_wave_max_drawdown_20", "cny_wave_max_runup_20",
        "cny_wave_min_position_20", "cny_wave_max_position_20",
        "cny_wave_last_z_20", "cny_wave_acceleration_5_5",
    ])
    return values, names


def build_waveform_features(index, history):
    rows = history["CNYRUB_TOM"]
    dates = np.asarray([row["date"] for row in rows], dtype=object)
    closes = np.asarray([row["close"] for row in rows], dtype=float)
    result, names = [], None
    for _currency, _position, day in index:
        end = int(np.searchsorted(dates, day, side="left"))
        age = (day - dates[end - 1]).days if end else 999
        available = bool(
            end >= WINDOW + 1 and age <= MAX_STALENESS_DAYS
        )
        values, row_names = [], []
        if available:
            returns = np.diff(np.log(closes[:end]))[-WINDOW:] * 10000.0
        else:
            returns = np.zeros(WINDOW, dtype=float)
        values.extend(map(float, returns))
        row_names.extend([
            f"cny_wave_ret_lag_{lag}" for lag in range(WINDOW, 0, -1)
        ])
        coefficients = dct(returns, type=2, norm="ortho")[:DCT_COMPONENTS]
        values.extend(map(float, coefficients))
        row_names.extend([
            f"cny_wave_dct_{i}" for i in range(DCT_COMPONENTS)
        ])
        summary, summary_names = _summary(returns)
        values.extend(summary)
        row_names.extend(summary_names)
        values.extend([float(min(age, 30)), float(not available)])
        row_names.extend(["cny_wave_age_days", "cny_wave_missing"])
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("CNY waveform feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite CNY waveform feature")
    return matrix, names or []


def causality_check(index, history, cutoff=dt.date(2025, 6, 30)):
    full, names = build_waveform_features(index, history)
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
    corrupted, changed_names = build_waveform_features(index, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != changed_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future CNY session changed a past waveform feature")
    return True
