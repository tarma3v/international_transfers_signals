"""Strictly lagged MOEX-versus-official-CBR CNY basis features."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import Series
from research.round6_moex_features import MAX_STALENESS_DAYS


LAGS = (1, 2, 5, 10, 20)
WINDOWS = (5, 20, 60)


def _bps(a, b):
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else np.nan


def _aligned_market(history, cbr_cny: Series):
    rows = history["CNYRUB_TOM"]
    dates = np.asarray([row["date"] for row in rows], dtype=object)
    values = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ("open", "high", "low", "close", "waprice", "trades")
    }
    cbr = np.full(len(rows), np.nan)
    cbr_date = np.empty(len(rows), dtype=object)
    cbr_date[:] = None
    for i, day in enumerate(dates):
        end = int(np.searchsorted(cbr_cny.dates, day, side="right"))
        if end:
            cbr[i] = float(cbr_cny.values[end - 1])
            cbr_date[i] = cbr_cny.dates[end - 1]
    basis = np.asarray([_bps(a, b) for a, b in zip(values["close"], cbr)])
    return dates, values, cbr, cbr_date, basis


def build_cny_basis_features(index, history, cbr_cny: Series):
    dates, market, cbr, cbr_dates, basis = _aligned_market(history, cbr_cny)
    result = []
    names = None
    for _currency, _position, day in index:
        end = int(np.searchsorted(dates, day, side="left"))
        source = dates[end - 1] if end else None
        market_age = (day - source).days if source is not None else 999
        available = bool(end and market_age <= MAX_STALENESS_DAYS and np.isfinite(basis[end - 1]))
        values, row_names = [], []
        for field in ("close", "open", "waprice", "high", "low"):
            value = _bps(float(market[field][end - 1]), float(cbr[end - 1])) if available else 0.0
            values.append(value if np.isfinite(value) else 0.0)
            row_names.append(f"cny_basis_{field}_bps")
        for lag in LAGS:
            if available and end > lag and np.isfinite(basis[end - 1 - lag]):
                change = float(basis[end - 1] - basis[end - 1 - lag])
                market_return = _bps(float(market["close"][end - 1]),
                                     float(market["close"][end - 1 - lag]))
                cbr_return = _bps(float(cbr[end - 1]), float(cbr[end - 1 - lag]))
                gap = market_return - cbr_return
            else:
                change, gap = 0.0, 0.0
            values.extend([change, gap])
            row_names.extend([f"cny_basis_change_{lag}", f"cny_basis_return_gap_{lag}"])
        history_basis = basis[:end]
        for window in WINDOWS:
            recent = history_basis[-window:]
            finite = recent[np.isfinite(recent)]
            if available and len(finite) >= max(3, window // 4):
                mean = float(np.mean(finite))
                std = float(np.std(finite))
                z = float((basis[end - 1] - mean) / std) if std > 1e-9 else 0.0
            else:
                mean, z = 0.0, 0.0
            values.extend([mean, z])
            row_names.extend([f"cny_basis_mean_{window}", f"cny_basis_z_{window}"])
        if end and cbr_dates[end - 1] is not None:
            cbr_age = (source - cbr_dates[end - 1]).days
        else:
            cbr_age = 999
        values.extend([float(min(market_age, 30)), float(min(cbr_age, 30)), float(not available)])
        row_names.extend(["cny_basis_market_age", "cny_basis_cbr_age", "cny_basis_missing"])
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("CNY basis schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite CNY basis feature")
    return matrix, names or []


def causality_check(index, history, cbr_cny, cutoff=dt.date(2025, 6, 30)):
    full, names = build_cny_basis_features(index, history, cbr_cny)
    changed_history = {}
    for ticker, rows in history.items():
        changed_history[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["date"] >= cutoff:
                for key in ("open", "high", "low", "close", "waprice", "trades"):
                    clone[key] *= 100.0
            changed_history[ticker].append(clone)
    cbr_values = cbr_cny.values.copy()
    future = cbr_cny.dates >= cutoff
    cbr_values[future] *= np.linspace(10.0, 100.0, int(future.sum()))
    changed_cbr = Series(cbr_cny.code, cbr_cny.dates.copy(), cbr_values)
    changed, changed_names = build_cny_basis_features(index, changed_history, changed_cbr)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != changed_names or not np.array_equal(full[past], changed[past]):
        raise AssertionError("future market/CBR value changed a past CNY basis feature")
    return True
