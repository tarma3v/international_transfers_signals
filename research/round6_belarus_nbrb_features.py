"""Strictly lagged RUB cross-rate features from the official NBRB archive."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.data import Series


DATA = Path("data/external_belarus_nbrb_rub_usd_cny_2016_2026.json")
LAGS = (1, 2, 5, 10, 20)


def load_nbrb(path: Path = DATA) -> tuple[dict[str, Series], str]:
    payload = path.read_bytes()
    raw = json.loads(payload)
    result = {}
    for code in ("RUB", "USD", "CNY"):
        rows = raw["rates"][code]
        result[code] = Series(
            code,
            np.asarray([dt.date.fromisoformat(row[0]) for row in rows], dtype=object),
            np.asarray([float(row[1]) for row in rows], dtype=float),
        )
    common_dates = result["RUB"].dates
    for code in ("USD", "CNY"):
        if not np.array_equal(common_dates, result[code].dates):
            raise ValueError(f"NBRB {code} dates are not aligned with RUB dates")
    return result, hashlib.sha256(payload).hexdigest()


def _last(series: Series, day: dt.date) -> tuple[int, float, int]:
    stop = int(np.searchsorted(series.dates, day, side="left"))
    if not stop:
        return stop, np.nan, 999
    return stop, float(series.values[stop - 1]), (day - series.dates[stop - 1]).days


def _ret(values: np.ndarray, stop: int, lag: int) -> float:
    if stop <= lag:
        return 0.0
    return float(np.log(values[stop - 1] / values[stop - 1 - lag]) * 10000.0)


def _basis(a: float, b: float) -> float:
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def build_nbrb_features(index, cbr_reference, nbrb):
    rows = []
    names = None
    usd_rub_history = nbrb["USD"].values / nbrb["RUB"].values
    cny_rub_history = nbrb["CNY"].values / nbrb["RUB"].values
    for _currency, _position, day in index:
        rstop, rub, rage = _last(nbrb["RUB"], day)
        ustop, usd, uage = _last(nbrb["USD"], day)
        cstop, cny, cage = _last(nbrb["CNY"], day)
        usd_stop = int(np.searchsorted(cbr_reference["USD"].dates, day, side="right"))
        cny_stop = int(np.searchsorted(cbr_reference["CNY"].dates, day, side="right"))
        cbr_usd = float(cbr_reference["USD"].values[usd_stop - 1]) if usd_stop else np.nan
        cbr_cny = float(cbr_reference["CNY"].values[cny_stop - 1]) if cny_stop else np.nan
        available = all(np.isfinite(value) and value > 0 for value in (
            rub, usd, cny, cbr_usd, cbr_cny,
        ))
        nbrb_usd_rub = usd / rub if available else 0.0
        nbrb_cny_rub = cny / rub if available else 0.0
        usd_basis = _basis(nbrb_usd_rub, cbr_usd)
        cny_basis = _basis(nbrb_cny_rub, cbr_cny)
        cross_basis = (
            _basis(nbrb_cny_rub / nbrb_usd_rub, cbr_cny / cbr_usd)
            if available else 0.0
        )
        values = [
            nbrb_usd_rub, nbrb_cny_rub, usd_basis, cny_basis,
            .5 * (usd_basis + cny_basis), cross_basis, usd_basis - cny_basis,
        ]
        row_names = [
            "nbrb_usd_rub", "nbrb_cny_rub", "nbrb_usd_basis_bps",
            "nbrb_cny_basis_bps", "nbrb_consensus_basis_bps",
            "nbrb_cny_usd_cross_basis_bps", "nbrb_usd_minus_cny_basis_bps",
        ]
        for code, stop in (("rub", rstop), ("usd", ustop), ("cny", cstop)):
            for lag in LAGS:
                values.append(_ret(nbrb[code.upper()].values, stop, lag))
                row_names.append(f"nbrb_{code}_byn_quote_ret_{lag}")
        for label, history, stop in (
            ("usd_rub", usd_rub_history, min(ustop, rstop)),
            ("cny_rub", cny_rub_history, min(cstop, rstop)),
        ):
            for lag in LAGS:
                values.append(_ret(history, stop, lag))
                row_names.append(f"nbrb_{label}_ret_{lag}")
        values.extend((
            float(min(rage, 30)), float(min(uage, 30)),
            float(min(cage, 30)), float(not available),
        ))
        row_names.extend((
            "nbrb_rub_age_days", "nbrb_usd_age_days", "nbrb_cny_age_days",
            "nbrb_missing",
        ))
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("NBRB feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite NBRB feature")
    return matrix, names or []


def causality_check(index, cbr_reference, nbrb, cutoff=dt.date(2025, 6, 30)):
    full, names = build_nbrb_features(index, cbr_reference, nbrb)
    changed = {}
    for code, series in nbrb.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_nbrb_features(index, cbr_reference, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future NBRB value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future NBRB corruption did not affect future rows")
    return True
