"""Strictly lagged features from the Central Bank of Armenia SOAP archive."""
from __future__ import annotations

import datetime as dt
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ml.data import Series


DATA = Path("data/external_cba_rub_usd_cny_2016_2026.xml")
LAGS = (1, 2, 5, 10, 20)


def _local(element):
    return element.tag.rsplit("}", 1)[-1]


def load_cba(path: Path = DATA) -> tuple[dict[str, Series], str]:
    payload = path.read_bytes()
    root = ET.fromstring(payload)
    rows: dict[str, dict[dt.date, float]] = {code: {} for code in ("RUB", "USD", "CNY")}
    for record in root.iter():
        if _local(record) != "ExchangeRatesByRange":
            continue
        values = {_local(child): (child.text or "").strip() for child in record}
        code = values.get("ISO")
        if code not in rows:
            continue
        day = dt.date.fromisoformat(values["RateDate"][:10])
        rows[code][day] = float(values["Rate"]) / float(values["Amount"])
    result = {}
    for code, mapping in rows.items():
        ordered = sorted(mapping.items())
        if not ordered:
            raise ValueError(f"CBA archive has no {code} rows")
        result[code] = Series(
            code,
            np.asarray([item[0] for item in ordered], dtype=object),
            np.asarray([item[1] for item in ordered], dtype=float),
        )
    return result, hashlib.sha256(payload).hexdigest()


def _last(series: Series, day: dt.date, strict: bool) -> tuple[int, float, int]:
    stop = int(np.searchsorted(series.dates, day, side="left" if strict else "right"))
    if not stop:
        return stop, np.nan, 999
    return stop, float(series.values[stop - 1]), (day - series.dates[stop - 1]).days


def _basis(a, b):
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def _ret(values, stop, lag):
    if stop <= lag:
        return 0.0
    return float(np.log(values[stop - 1] / values[stop - 1 - lag]) * 10000.0)


def build_cba_features(index, target_series, cbr_reference, cba):
    rows = []
    names = None
    for _currency, _position, day in index:
        rstop, rub, rage = _last(cba["RUB"], day, strict=True)
        ustop, usd, uage = _last(cba["USD"], day, strict=True)
        cstop, cny, cage = _last(cba["CNY"], day, strict=True)
        _astop, cbr_amd, aage = _last(target_series["AMD"], day, strict=False)
        _ustop, cbr_usd, _ = _last(cbr_reference["USD"], day, strict=False)
        _cstop, cbr_cny, _ = _last(cbr_reference["CNY"], day, strict=False)
        available = all(np.isfinite(value) and value > 0 for value in (
            rub, usd, cny, cbr_amd, cbr_usd, cbr_cny,
        ))
        direct = 1.0 / rub if available else 0.0
        via_usd = cbr_usd / usd if available else 0.0
        via_cny = cbr_cny / cny if available else 0.0
        implied = (direct, via_usd, via_cny)
        values = []
        row_names = []
        for label, value in zip(("direct", "usd", "cny"), implied):
            values.extend([value, _basis(value, cbr_amd)])
            row_names.extend([
                f"cba_{label}_implied_amd_rub", f"cba_{label}_basis_bps",
            ])
        consensus = float(np.exp(np.mean(np.log(implied)))) if available else 0.0
        values.extend([
            consensus, _basis(consensus, cbr_amd),
            _basis(direct, via_usd), _basis(direct, via_cny),
            _basis(via_usd, via_cny),
        ])
        row_names.extend([
            "cba_consensus_implied_amd_rub", "cba_consensus_basis_bps",
            "cba_direct_minus_usd_bps", "cba_direct_minus_cny_bps",
            "cba_usd_minus_cny_bps",
        ])
        for code, stop in (("rub", rstop), ("usd", ustop), ("cny", cstop)):
            for lag in LAGS:
                values.append(_ret(cba[code.upper()].values, stop, lag))
                row_names.append(f"cba_{code}_quote_ret_{lag}")
        values.extend([
            float(min(rage, 30)), float(min(uage, 30)), float(min(cage, 30)),
            float(min(aage, 30)), float(not available),
        ])
        row_names.extend([
            "cba_rub_age_days", "cba_usd_age_days", "cba_cny_age_days",
            "cba_cbr_amd_age_days", "cba_missing",
        ])
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("CBA feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite CBA feature")
    return matrix, names or []


def causality_check(index, target_series, cbr_reference, cba, cutoff=dt.date(2025, 6, 30)):
    full, names = build_cba_features(index, target_series, cbr_reference, cba)
    changed = {}
    for code, series in cba.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_cba_features(index, target_series, cbr_reference, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future CBA value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future CBA corruption did not affect future rows")
    return True
