"""Strictly lagged features from the National Bank of Kyrgyzstan XLS archive."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import xlrd

from ml.data import Series


DATA = Path("data/external_kyrgyzstan_nbkr_daily_2010_2026.xls")
LAGS = (1, 2, 5, 10, 20)


def load_kyrgyz_nbkr(path: Path = DATA) -> tuple[dict[str, Series], str]:
    payload = path.read_bytes()
    book = xlrd.open_workbook(
        file_contents=payload, ignore_workbook_corruption=True,
    )
    records = {code: {} for code in ("RUB", "USD", "CNY")}
    for sheet in book.sheets():
        headers = [str(value).strip().upper() for value in sheet.row_values(0)]
        columns = {code: headers.index(code) for code in records if code in headers}
        for row in range(1, sheet.nrows):
            raw_day = sheet.cell_value(row, 0)
            if not isinstance(raw_day, (float, int)):
                continue
            day = xlrd.xldate_as_datetime(float(raw_day), book.datemode).date()
            for code, column in columns.items():
                raw_value = sheet.cell_value(row, column)
                if isinstance(raw_value, (float, int)) and float(raw_value) > 0:
                    records[code][day] = float(raw_value)
    result = {}
    for code, mapping in records.items():
        ordered = sorted(mapping.items())
        if not ordered:
            raise ValueError(f"Kyrgyz NBKR archive has no {code} observations")
        result[code] = Series(
            code,
            np.asarray([row[0] for row in ordered], dtype=object),
            np.asarray([row[1] for row in ordered], dtype=float),
        )
    return result, hashlib.sha256(payload).hexdigest()


def _last(series: Series, day: dt.date, strict: bool) -> tuple[int, float, int]:
    stop = int(np.searchsorted(series.dates, day, side="left" if strict else "right"))
    if not stop:
        return stop, np.nan, 999
    return stop, float(series.values[stop - 1]), (day - series.dates[stop - 1]).days


def _basis(a: float, b: float) -> float:
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def _ret(values: np.ndarray, stop: int, lag: int) -> float:
    if stop <= lag:
        return 0.0
    return float(np.log(values[stop - 1] / values[stop - 1 - lag]) * 10000.0)


def build_kyrgyz_nbkr_features(index, target_series, cbr_reference, local):
    rows = []
    names = None
    for _currency, _position, day in index:
        rstop, rub, rage = _last(local["RUB"], day, strict=True)
        ustop, usd, uage = _last(local["USD"], day, strict=True)
        cstop, cny, cage = _last(local["CNY"], day, strict=True)
        _kstop, cbr_kgs, kage = _last(target_series["KGS"], day, strict=False)
        _ustop, cbr_usd, _ = _last(cbr_reference["USD"], day, strict=False)
        _cstop, cbr_cny, _ = _last(cbr_reference["CNY"], day, strict=False)
        available = all(np.isfinite(value) and value > 0 for value in (
            rub, usd, cny, cbr_kgs, cbr_usd, cbr_cny,
        ))
        direct = 1.0 / rub if available else 0.0
        via_usd = cbr_usd / usd if available else 0.0
        via_cny = cbr_cny / cny if available else 0.0
        implied = (direct, via_usd, via_cny)
        values = []
        row_names = []
        for label, value in zip(("direct", "usd", "cny"), implied):
            values.extend((value, _basis(value, cbr_kgs)))
            row_names.extend((
                f"kyrgyz_nbkr_{label}_implied_kgs_rub",
                f"kyrgyz_nbkr_{label}_basis_bps",
            ))
        consensus = float(np.exp(np.mean(np.log(implied)))) if available else 0.0
        values.extend((
            consensus, _basis(consensus, cbr_kgs),
            _basis(direct, via_usd), _basis(direct, via_cny),
            _basis(via_usd, via_cny),
        ))
        row_names.extend((
            "kyrgyz_nbkr_consensus_implied_kgs_rub",
            "kyrgyz_nbkr_consensus_basis_bps",
            "kyrgyz_nbkr_direct_minus_usd_bps",
            "kyrgyz_nbkr_direct_minus_cny_bps",
            "kyrgyz_nbkr_usd_minus_cny_bps",
        ))
        for code, stop in (("rub", rstop), ("usd", ustop), ("cny", cstop)):
            for lag in LAGS:
                values.append(_ret(local[code.upper()].values, stop, lag))
                row_names.append(f"kyrgyz_nbkr_{code}_quote_ret_{lag}")
        values.extend((
            float(min(rage, 30)), float(min(uage, 30)),
            float(min(cage, 30)), float(min(kage, 30)), float(not available),
        ))
        row_names.extend((
            "kyrgyz_nbkr_rub_age_days", "kyrgyz_nbkr_usd_age_days",
            "kyrgyz_nbkr_cny_age_days", "kyrgyz_nbkr_cbr_kgs_age_days",
            "kyrgyz_nbkr_missing",
        ))
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("Kyrgyz NBKR feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite Kyrgyz NBKR feature")
    return matrix, names or []


def causality_check(
    index, target_series, cbr_reference, local,
    cutoff=dt.date(2025, 6, 30),
):
    full, names = build_kyrgyz_nbkr_features(
        index, target_series, cbr_reference, local,
    )
    changed = {}
    for code, series in local.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_kyrgyz_nbkr_features(
        index, target_series, cbr_reference, changed,
    )
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future Kyrgyz NBKR value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future Kyrgyz NBKR corruption did not affect future rows")
    return True
