"""Strictly lagged features from the Central Bank of Uzbekistan XLS archive."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

import numpy as np
import xlrd

from ml.data import Series


FILES = tuple(
    Path(f"data/external_uzbekistan_cbu_{year}.xls")
    for year in range(2016, 2027)
)
NUMERIC_CODES = {"RUB": "643", "USD": "840", "CNY": "156"}
LAGS = (1, 2, 5, 10, 20)


def _header_code(value: object) -> str | None:
    match = re.search(r"\((\d{3})\)\s*$", str(value))
    return match.group(1) if match else None


def load_uzbek_cbu(paths=FILES) -> tuple[dict[str, Series], str]:
    """Load official quotes as UZS per one unit of RUB/USD/CNY."""
    records: dict[str, dict[dt.date, float]] = {code: {} for code in NUMERIC_CODES}
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(payload)
        book = xlrd.open_workbook(
            file_contents=payload, ignore_workbook_corruption=True,
        )
        sheet = book.sheet_by_index(0)
        headers = sheet.row_values(1)
        value_columns = {
            code: next(
                column for column, value in enumerate(headers)
                if _header_code(value) == numeric
                and "единиц" not in str(value).lower()
            )
            for code, numeric in NUMERIC_CODES.items()
        }
        for row in range(2, sheet.nrows):
            raw_day = str(sheet.cell_value(row, 0)).strip()
            if not raw_day:
                continue
            day = dt.date.fromisoformat(raw_day[:10])
            for code, column in value_columns.items():
                nominal = float(sheet.cell_value(row, column - 1))
                value = float(sheet.cell_value(row, column))
                if nominal > 0 and value > 0:
                    records[code][day] = value / nominal
    result = {}
    for code, mapping in records.items():
        ordered = sorted(mapping.items())
        if not ordered:
            raise ValueError(f"Uzbek CBU archive has no {code} observations")
        result[code] = Series(
            code,
            np.asarray([row[0] for row in ordered], dtype=object),
            np.asarray([row[1] for row in ordered], dtype=float),
        )
    return result, digest.hexdigest()


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


def build_uzbek_cbu_features(index, target_series, cbr_reference, local):
    rows = []
    names = None
    for _currency, _position, day in index:
        rstop, rub, rage = _last(local["RUB"], day, strict=True)
        ustop, usd, uage = _last(local["USD"], day, strict=True)
        cstop, cny, cage = _last(local["CNY"], day, strict=True)
        _zstop, cbr_uzs, zage = _last(target_series["UZS"], day, strict=False)
        _ustop, cbr_usd, _ = _last(cbr_reference["USD"], day, strict=False)
        _cstop, cbr_cny, _ = _last(cbr_reference["CNY"], day, strict=False)
        available = all(np.isfinite(value) and value > 0 for value in (
            rub, usd, cny, cbr_uzs, cbr_usd, cbr_cny,
        ))
        direct = 1.0 / rub if available else 0.0
        via_usd = cbr_usd / usd if available else 0.0
        via_cny = cbr_cny / cny if available else 0.0
        implied = (direct, via_usd, via_cny)
        values = []
        row_names = []
        for label, value in zip(("direct", "usd", "cny"), implied):
            values.extend((value, _basis(value, cbr_uzs)))
            row_names.extend((
                f"uzbek_cbu_{label}_implied_uzs_rub",
                f"uzbek_cbu_{label}_basis_bps",
            ))
        consensus = float(np.exp(np.mean(np.log(implied)))) if available else 0.0
        values.extend((
            consensus, _basis(consensus, cbr_uzs),
            _basis(direct, via_usd), _basis(direct, via_cny),
            _basis(via_usd, via_cny),
        ))
        row_names.extend((
            "uzbek_cbu_consensus_implied_uzs_rub",
            "uzbek_cbu_consensus_basis_bps",
            "uzbek_cbu_direct_minus_usd_bps",
            "uzbek_cbu_direct_minus_cny_bps",
            "uzbek_cbu_usd_minus_cny_bps",
        ))
        for code, stop in (("rub", rstop), ("usd", ustop), ("cny", cstop)):
            for lag in LAGS:
                values.append(_ret(local[code.upper()].values, stop, lag))
                row_names.append(f"uzbek_cbu_{code}_quote_ret_{lag}")
        values.extend((
            float(min(rage, 30)), float(min(uage, 30)),
            float(min(cage, 30)), float(min(zage, 30)), float(not available),
        ))
        row_names.extend((
            "uzbek_cbu_rub_age_days", "uzbek_cbu_usd_age_days",
            "uzbek_cbu_cny_age_days", "uzbek_cbu_cbr_uzs_age_days",
            "uzbek_cbu_missing",
        ))
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("Uzbek CBU feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite Uzbek CBU feature")
    return matrix, names or []


def causality_check(
    index, target_series, cbr_reference, local,
    cutoff=dt.date(2025, 6, 30),
):
    full, names = build_uzbek_cbu_features(
        index, target_series, cbr_reference, local,
    )
    changed = {}
    for code, series in local.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_uzbek_cbu_features(
        index, target_series, cbr_reference, changed,
    )
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future Uzbek CBU value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future Uzbek CBU corruption did not affect future rows")
    return True
