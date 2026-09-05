"""Causal features from the National Bank of Tajikistan's official FX archive.

The archive is daily, including weekends.  To avoid making an unsupported
assumption about its intraday publication time, every feature uses an NBT
effective date strictly earlier than the CBR signal date.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ml.data import Series


FILES = {
    "RUB": Path("data/external_nbt_rub_2016_2026.xml"),
    "USD": Path("data/external_nbt_usd_2016_2026.xml"),
    "CNY": Path("data/external_nbt_cny_2016_2026.xml"),
}
LAGS = (1, 2, 5, 10, 20)
WINDOWS = (5, 20, 60)


def load_nbt(paths: dict[str, Path] = FILES) -> tuple[dict[str, Series], str]:
    """Load NBT quotes as TJS per one unit of the quoted currency."""
    result: dict[str, Series] = {}
    digest = hashlib.sha256()
    for expected, path in sorted(paths.items()):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(payload)
        root = ET.fromstring(payload.decode("ascii"))
        rows: dict[dt.date, float] = {}
        for record in root.findall("Record"):
            code = (record.findtext("CharCode") or "").strip()
            if code != expected:
                raise ValueError(f"{path}: expected {expected}, got {code}")
            day = dt.datetime.strptime(record.attrib["Date"], "%d.%m.%Y").date()
            nominal = float(record.findtext("Nominal"))
            value = float(record.findtext("Value")) / nominal
            rows[day] = value
        ordered = sorted(rows.items())
        result[expected] = Series(
            expected,
            np.asarray([row[0] for row in ordered], dtype=object),
            np.asarray([row[1] for row in ordered], dtype=float),
        )
    return result, digest.hexdigest()


def _last(series: Series, day: dt.date, strict: bool) -> tuple[int, float, int]:
    side = "left" if strict else "right"
    stop = int(np.searchsorted(series.dates, day, side=side))
    if not stop:
        return stop, np.nan, 999
    return stop, float(series.values[stop - 1]), (day - series.dates[stop - 1]).days


def _ret(values: np.ndarray, stop: int, lag: int) -> float:
    if stop <= lag:
        return 0.0
    return float(np.log(values[stop - 1] / values[stop - 1 - lag]) * 10000.0)


def _basis(a: float, b: float) -> float:
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def build_nbt_features(index, target_series, cbr_reference, nbt):
    """Build strictly lagged local-CB shadow-rate and dynamics features."""
    rows = []
    names = None
    for _currency, _position, day in index:
        rub_stop, rub_quote, rub_age = _last(nbt["RUB"], day, strict=True)
        usd_stop, usd_quote, usd_age = _last(nbt["USD"], day, strict=True)
        cny_stop, cny_quote, cny_age = _last(nbt["CNY"], day, strict=True)
        _tjs_stop, cbr_tjs, cbr_tjs_age = _last(target_series["TJS"], day, strict=False)
        cbr_usd_stop, cbr_usd, _cbr_usd_age = _last(
            cbr_reference["USD"], day, strict=False,
        )
        cbr_cny_stop, cbr_cny, _cbr_cny_age = _last(
            cbr_reference["CNY"], day, strict=False,
        )
        available = all(np.isfinite(value) and value > 0 for value in (
            rub_quote, usd_quote, cny_quote, cbr_tjs, cbr_usd, cbr_cny,
        ))
        direct = 1.0 / rub_quote if available else 0.0
        via_usd = cbr_usd / usd_quote if available else 0.0
        via_cny = cbr_cny / cny_quote if available else 0.0
        implied = (direct, via_usd, via_cny)
        values = []
        row_names = []
        for label, value in zip(("direct", "usd", "cny"), implied):
            values.extend([value, _basis(value, cbr_tjs)])
            row_names.extend([
                f"nbt_{label}_implied_tjs_rub",
                f"nbt_{label}_basis_bps",
            ])
        consensus = float(np.exp(np.mean(np.log(implied)))) if available else 0.0
        values.extend([
            consensus,
            _basis(consensus, cbr_tjs),
            _basis(direct, via_usd),
            _basis(direct, via_cny),
            _basis(via_usd, via_cny),
        ])
        row_names.extend([
            "nbt_consensus_implied_tjs_rub", "nbt_consensus_basis_bps",
            "nbt_direct_minus_usd_bps", "nbt_direct_minus_cny_bps",
            "nbt_usd_minus_cny_bps",
        ])
        for code, stop in (("rub", rub_stop), ("usd", usd_stop), ("cny", cny_stop)):
            source = nbt[code.upper()]
            for lag in LAGS:
                values.append(_ret(source.values, stop, lag))
                row_names.append(f"nbt_{code}_quote_ret_{lag}")
        for label, value in zip(("direct", "usd", "cny"), implied):
            stop = {"direct": rub_stop, "usd": usd_stop, "cny": cny_stop}[label]
            series_values = {
                "direct": 1.0 / nbt["RUB"].values,
                "usd": 1.0 / nbt["USD"].values,
                "cny": 1.0 / nbt["CNY"].values,
            }[label]
            current_local = {
                "direct": direct,
                "usd": 1.0 / usd_quote if available else 0.0,
                "cny": 1.0 / cny_quote if available else 0.0,
            }[label]
            for window in WINDOWS:
                history = series_values[max(0, stop - window):stop]
                if available and len(history) >= max(3, window // 4):
                    logs = np.log(history)
                    std = float(np.std(logs))
                    z = float((np.log(current_local) - np.mean(logs)) / std) if std > 1e-12 else 0.0
                else:
                    z = 0.0
                values.append(z)
                row_names.append(f"nbt_{label}_z_{window}")
        values.extend([
            float(min(rub_age, 30)), float(min(usd_age, 30)),
            float(min(cny_age, 30)), float(min(cbr_tjs_age, 30)),
            float(not available),
        ])
        row_names.extend([
            "nbt_rub_age_days", "nbt_usd_age_days", "nbt_cny_age_days",
            "nbt_cbr_tjs_age_days", "nbt_missing",
        ])
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("NBT feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite NBT feature")
    return matrix, names or []


def causality_check(index, target_series, cbr_reference, nbt, cutoff=dt.date(2025, 6, 30)):
    full, names = build_nbt_features(index, target_series, cbr_reference, nbt)
    changed = {}
    for code, series in nbt.items():
        values = series.values.copy()
        future = series.dates >= cutoff
        values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        changed[code] = Series(code, series.dates.copy(), values)
    altered, altered_names = build_nbt_features(
        index, target_series, cbr_reference, changed,
    )
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future NBT value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future NBT corruption did not affect future rows")
    return True
