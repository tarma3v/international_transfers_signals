"""Strictly lagged RUB market features from the National Bank of Georgia.

NBG publishes GEL quotes for RUB, USD and CNY.  Their cross-rates provide an
independent estimate of RUB/USD and RUB/CNY without using the next CBR rate.
We deliberately use only NBG rates whose *valid-from* date is strictly before
the CBR signal date.  This is more conservative than the source's publication
timestamp and avoids any intraday-availability assumption.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

from ml.data import Series


DATA = Path("data/external_georgia_nbg_rub_usd_cny_2016_2026.csv")
LAGS = (1, 2, 5, 10, 20)


def load_nbg(path: Path = DATA) -> tuple[dict[str, Series], str]:
    """Load official NBG quotes as GEL per one unit of RUB/USD/CNY."""
    payload = path.read_bytes()
    records: dict[str, dict[dt.date, float]] = {
        code: {} for code in ("RUB", "USD", "CNY")
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["Code"].strip()
            if code not in records:
                continue
            day = dt.datetime.strptime(row["ValidFromDate"], "%m/%d/%Y").date()
            value = float(row["Rate"]) / float(row["Quantity"])
            previous = records[code].get(day)
            if previous is not None and not np.isclose(previous, value):
                raise ValueError(f"NBG has conflicting {code} rates on {day}")
            records[code][day] = value
    result = {}
    for code, mapping in records.items():
        ordered = sorted(mapping.items())
        if not ordered:
            raise ValueError(f"NBG archive has no {code} observations")
        result[code] = Series(
            code,
            np.asarray([row[0] for row in ordered], dtype=object),
            np.asarray([row[1] for row in ordered], dtype=float),
        )
    common_dates = result["RUB"].dates
    for code in ("USD", "CNY"):
        if not np.array_equal(common_dates, result[code].dates):
            raise ValueError(f"NBG {code} dates are not aligned with RUB dates")
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


def build_nbg_features(index, cbr_reference, nbg):
    rows = []
    names = None
    for _currency, _position, day in index:
        rstop, rub, rage = _last(nbg["RUB"], day)
        ustop, usd, uage = _last(nbg["USD"], day)
        cstop, cny, cage = _last(nbg["CNY"], day)
        usd_stop = int(np.searchsorted(cbr_reference["USD"].dates, day, side="right"))
        cny_stop = int(np.searchsorted(cbr_reference["CNY"].dates, day, side="right"))
        cbr_usd = (
            float(cbr_reference["USD"].values[usd_stop - 1]) if usd_stop else np.nan
        )
        cbr_cny = (
            float(cbr_reference["CNY"].values[cny_stop - 1]) if cny_stop else np.nan
        )
        available = all(np.isfinite(value) and value > 0 for value in (
            rub, usd, cny, cbr_usd, cbr_cny,
        ))
        nbg_usd_rub = usd / rub if available else 0.0
        nbg_cny_rub = cny / rub if available else 0.0
        usd_basis = _basis(nbg_usd_rub, cbr_usd)
        cny_basis = _basis(nbg_cny_rub, cbr_cny)
        consensus_basis = .5 * (usd_basis + cny_basis)
        cross_basis = (
            _basis(nbg_cny_rub / nbg_usd_rub, cbr_cny / cbr_usd)
            if available else 0.0
        )
        values = [
            nbg_usd_rub, nbg_cny_rub, usd_basis, cny_basis,
            consensus_basis, cross_basis, usd_basis - cny_basis,
        ]
        row_names = [
            "nbg_usd_rub", "nbg_cny_rub", "nbg_usd_basis_bps",
            "nbg_cny_basis_bps", "nbg_consensus_basis_bps",
            "nbg_cny_usd_cross_basis_bps", "nbg_usd_minus_cny_basis_bps",
        ]
        for code, stop in (("rub", rstop), ("usd", ustop), ("cny", cstop)):
            for lag in LAGS:
                values.append(_ret(nbg[code.upper()].values, stop, lag))
                row_names.append(f"nbg_{code}_gel_quote_ret_{lag}")
        for label, synthetic, stop in (
            ("usd_rub", nbg["USD"].values / nbg["RUB"].values, min(ustop, rstop)),
            ("cny_rub", nbg["CNY"].values / nbg["RUB"].values, min(cstop, rstop)),
        ):
            for lag in LAGS:
                values.append(_ret(synthetic, stop, lag))
                row_names.append(f"nbg_{label}_ret_{lag}")
        values.extend((
            float(min(rage, 30)), float(min(uage, 30)),
            float(min(cage, 30)), float(not available),
        ))
        row_names.extend((
            "nbg_rub_age_days", "nbg_usd_age_days", "nbg_cny_age_days",
            "nbg_missing",
        ))
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("NBG feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite NBG feature")
    return matrix, names or []


def causality_check(index, cbr_reference, nbg, cutoff=dt.date(2025, 6, 30)):
    full, names = build_nbg_features(index, cbr_reference, nbg)
    changed = {}
    for code, series in nbg.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_nbg_features(index, cbr_reference, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future NBG value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future NBG corruption did not affect future rows")
    return True
