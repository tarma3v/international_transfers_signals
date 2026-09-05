"""Strictly lagged features from MOEX perpetual FX futures."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_perpetual_fx_history_2022_2026.json")
MANIFEST = Path("results/research/round6/moex_perpetual/data_manifest.json")
TICKERS = ("CNYRUBF", "USDRUBF")
REFERENCE = {"CNYRUBF": "CNY", "USDRUBF": "USD"}
RETURN_WINDOWS = (1, 2, 5, 10, 20)
VOL_WINDOWS = (5, 20)
POSITION_WINDOWS = (1, 5)
MAX_STALENESS_DAYS = 7


def load_perpetual_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX perpetual futures payload digest changed")
    result = {}
    for instrument in payload["instruments"]:
        columns = list(instrument["columns"])
        position = {name: columns.index(name) for name in columns}
        rows = []
        for raw in instrument["rows"]:
            close = raw[position["CLOSE"]]
            settle = raw[position["SETTLEPRICE"]]
            trades = raw[position["NUMTRADES"]]
            if (
                close is None or float(close) <= 0.0
                or settle is None or float(settle) <= 0.0
                or trades is None or float(trades) <= 0.0
            ):
                continue
            def value(name, default=0.0):
                item = raw[position[name]]
                return float(item) if item is not None else float(default)
            rows.append({
                "date": dt.date.fromisoformat(raw[position["TRADEDATE"]]),
                "open": value("OPEN", close),
                "low": value("LOW", close),
                "high": value("HIGH", close),
                "close": float(close),
                "settle": float(settle),
                "waprice": value("WAPRICE", close),
                "volume": value("VOLUME"),
                "open_position": value("OPENPOSITION"),
                "swap_rate": value("SWAPRATE"),
                "trades": float(trades),
            })
        result[instrument["ticker"]] = rows
    return result, digest


def _ratio(numerator, denominator):
    return float(np.log(numerator / denominator) * 10000.0) if denominator > 0 else 0.0


def _reference_last(series, day):
    stop = int(np.searchsorted(series.dates, day, side="right"))
    return float(series.values[stop - 1]) if stop else np.nan


def build_perpetual_features(index, history, references):
    arrays = {}
    for ticker in TICKERS:
        rows = history[ticker]
        arrays[ticker] = {
            key: np.asarray([
                row[key] for row in rows
            ], dtype=object if key == "date" else float)
            for key in (
                "date", "open", "low", "high", "close", "settle",
                "waprice", "volume", "open_position", "swap_rate", "trades",
            )
        }
    result, names = [], None
    for _currency, _position, day in index:
        values, row_names, last = [], [], {}
        for ticker in TICKERS:
            item = arrays[ticker]
            end = int(np.searchsorted(item["date"], day, side="left"))
            age = (day - item["date"][end - 1]).days if end else 999
            available = bool(end and age <= MAX_STALENESS_DAYS)
            prefix = f"moex_{ticker.lower()}"
            close = item["close"][:end]
            settle = item["settle"][:end]
            for window in RETURN_WINDOWS:
                score = (
                    _ratio(float(close[-1]), float(close[-1 - window]))
                    if available and len(close) > window else 0.0
                )
                values.append(score)
                row_names.append(f"{prefix}_ret_{window}")
            returns = np.diff(np.log(close)) * 10000.0 if len(close) > 1 else np.asarray([])
            for window in VOL_WINDOWS:
                score = (
                    float(np.std(returns[-window:]))
                    if available and len(returns) else 0.0
                )
                values.append(score)
                row_names.append(f"{prefix}_vol_{window}")
            if available:
                open_, low, high, close_, settle_, waprice = (
                    float(item[key][end - 1])
                    for key in ("open", "low", "high", "close", "settle", "waprice")
                )
                reference = _reference_last(references[REFERENCE[ticker]], day)
                daily = (
                    _ratio(close_, open_),
                    _ratio(high, low),
                    _ratio(close_, waprice),
                    _ratio(close_, settle_),
                    float(np.log1p(item["volume"][end - 1])),
                    float(np.log1p(item["trades"][end - 1])),
                    float(np.log1p(item["open_position"][end - 1])),
                    float(item["swap_rate"][end - 1]),
                    _ratio(close_, reference) if np.isfinite(reference) else 0.0,
                    _ratio(settle_, reference) if np.isfinite(reference) else 0.0,
                )
                last[ticker] = {
                    "close": close_, "settle": settle_,
                    "swap_rate": float(item["swap_rate"][end - 1]),
                    "ret_1": _ratio(close_, float(close[-2])) if len(close) > 1 else 0.0,
                    "ret_5": _ratio(close_, float(close[-6])) if len(close) > 5 else 0.0,
                }
            else:
                daily = (0.0,) * 10
                last[ticker] = None
            values.extend(daily)
            row_names.extend([
                f"{prefix}_open_close", f"{prefix}_intraday_range",
                f"{prefix}_close_wap", f"{prefix}_close_settle",
                f"{prefix}_log_volume", f"{prefix}_log_trades",
                f"{prefix}_log_open_position", f"{prefix}_swap_rate",
                f"{prefix}_close_cbr_basis", f"{prefix}_settle_cbr_basis",
            ])
            for window in POSITION_WINDOWS:
                score = (
                    float(
                        np.log1p(item["open_position"][end - 1])
                        - np.log1p(item["open_position"][end - 1 - window])
                    ) if available and end > window else 0.0
                )
                values.append(score)
                row_names.append(f"{prefix}_open_position_change_{window}")
            values.extend((float(min(age, 30)), float(not available)))
            row_names.extend((f"{prefix}_age_days", f"{prefix}_missing"))

        cny, usd = last["CNYRUBF"], last["USDRUBF"]
        cbr_cny = _reference_last(references["CNY"], day)
        cbr_usd = _reference_last(references["USD"], day)
        if cny is not None and usd is not None and np.isfinite(cbr_cny) and np.isfinite(cbr_usd):
            cross = (
                _ratio(usd["close"] / cny["close"], cbr_usd / cbr_cny),
                _ratio(usd["settle"] / cny["settle"], cbr_usd / cbr_cny),
                cny["ret_1"] - usd["ret_1"],
                cny["ret_5"] - usd["ret_5"],
                cny["swap_rate"] - usd["swap_rate"],
            )
        else:
            cross = (0.0,) * 5
        values.extend(cross)
        row_names.extend((
            "moex_perpetual_close_cross_basis",
            "moex_perpetual_settle_cross_basis",
            "moex_perpetual_return_divergence_1",
            "moex_perpetual_return_divergence_5",
            "moex_perpetual_swap_spread",
        ))
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("perpetual feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite perpetual futures feature")
    return matrix, names or []


def causality_check(index, history, references, cutoff=dt.date(2025, 6, 30)):
    full, names = build_perpetual_features(index, history, references)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["date"] >= cutoff:
                for key in (
                    "open", "low", "high", "close", "settle", "waprice",
                    "volume", "open_position", "swap_rate", "trades",
                ):
                    clone[key] *= 100.0
            changed[ticker].append(clone)
    altered, altered_names = build_perpetual_features(index, changed, references)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future futures value changed a past feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future futures corruption did not affect future features")
    return True
