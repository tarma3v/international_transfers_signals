"""Noon-Moscow features from official MOEX hourly spot-FX candles."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_spot_fx_hourly_2022_2026.json")
MANIFEST = Path("results/research/round6/moex_spot_hourly/data_manifest.json")
TICKERS = ("CNYRUB_TOM", "USD000UTSTOM")
REFERENCE = {"CNYRUB_TOM": "CNY", "USD000UTSTOM": "USD"}
PERPETUAL = {"CNYRUB_TOM": "CNYRUBF", "USD000UTSTOM": "USDRUBF"}
DECISION_TIME = dt.time(12, 0)
SPOT_ONLY_FEATURES = 29


def load_spot_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX hourly spot payload digest changed")
    history = {}
    for instrument in payload["instruments"]:
        columns = list(instrument["columns"])
        position = {name: columns.index(name) for name in columns}
        rows = []
        for raw in instrument["rows"]:
            rows.append({
                "begin": dt.datetime.fromisoformat(raw[position["begin"]]),
                "end": dt.datetime.fromisoformat(raw[position["end"]]),
                "open": float(raw[position["open"]]),
                "close": float(raw[position["close"]]),
                "high": float(raw[position["high"]]),
                "low": float(raw[position["low"]]),
            })
        history[instrument["ticker"]] = rows
    return history, digest


def _ratio(numerator, denominator):
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return float(np.log(numerator / denominator) * 10000.0)


def _reference_last(series, day):
    stop = int(np.searchsorted(series.dates, day, side="right"))
    return float(series.values[stop - 1]) if stop else np.nan


def _arrays(history):
    return {
        ticker: {
            key: np.asarray(
                [row[key] for row in rows],
                dtype=object if key in ("begin", "end") else float,
            )
            for key in ("begin", "end", "open", "close", "high", "low")
        }
        for ticker, rows in history.items()
    }


def _state(item, midnight, cutoff, reference):
    start = int(np.searchsorted(item["begin"], midnight, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff, side="left"))
    rows = np.arange(start, stop, dtype=int)
    rows = rows[item["begin"][rows] >= midnight]
    previous_stop = int(np.searchsorted(item["end"], midnight, side="left"))
    previous_close = float(item["close"][previous_stop - 1]) if previous_stop else np.nan
    if not len(rows):
        previous_end = item["end"][previous_stop - 1] if previous_stop else None
        age = (
            min((cutoff - previous_end).total_seconds() / 3600.0, 720.0)
            if previous_end is not None else 720.0
        )
        return (0.0,) * 11 + (float(age), 1.0), None
    opens, closes = item["open"][rows], item["close"][rows]
    highs, lows = item["high"][rows], item["low"][rows]
    open_, close = float(opens[0]), float(closes[-1])
    high, low = float(np.max(highs)), float(np.min(lows))
    returns = np.diff(np.log(closes)) * 10000.0
    slope = (
        float(np.polyfit(np.arange(len(closes)), np.log(closes), 1)[0] * 10000.0)
        if len(closes) > 1 else 0.0
    )
    position = (close - low) / (high - low) if high > low else .5
    values = (
        close,
        _ratio(close, previous_close) if np.isfinite(previous_close) else 0.0,
        _ratio(close, open_),
        _ratio(close, float(closes[-2])) if len(closes) > 1 else 0.0,
        _ratio(close, float(closes[-3])) if len(closes) > 2 else 0.0,
        _ratio(high, low),
        float(np.std(returns)) if len(returns) else 0.0,
        float(len(closes)), slope, float(position),
        _ratio(close, reference) if np.isfinite(reference) else 0.0,
        float((cutoff - item["end"][rows[-1]]).total_seconds() / 3600.0),
        0.0,
    )
    compact = {
        "close": close, "open_return": values[2], "last_return": values[3],
    }
    return values, compact


def build_spot_features(index, history, references, perpetual_history):
    spot_arrays = _arrays(history)
    perpetual_arrays = _arrays(perpetual_history)
    result, names = [], None
    base_names = (
        "last", "overnight_return", "open_to_cutoff_return",
        "last_1h_return", "last_2h_return", "cutoff_range",
        "realized_vol", "completed_candles", "log_price_slope",
        "range_position", "cbr_basis", "age_hours", "missing",
    )
    for _currency, _position, day in index:
        midnight = dt.datetime.combine(day, dt.time())
        cutoff = dt.datetime.combine(day, DECISION_TIME)
        values, row_names, spot_last, futures_last = [], [], {}, {}
        for ticker in TICKERS:
            reference = _reference_last(references[REFERENCE[ticker]], day)
            feature, spot_last[ticker] = _state(
                spot_arrays[ticker], midnight, cutoff, reference,
            )
            values.extend(feature)
            prefix = f"moex_hourly_spot_{ticker.lower()}"
            row_names.extend(f"{prefix}_{name}" for name in base_names)
            _perp_feature, futures_last[ticker] = _state(
                perpetual_arrays[PERPETUAL[ticker]], midnight, cutoff, reference,
            )

        cny, usd = spot_last["CNYRUB_TOM"], spot_last["USD000UTSTOM"]
        cbr_cny = _reference_last(references["CNY"], day)
        cbr_usd = _reference_last(references["USD"], day)
        if cny is not None and usd is not None and np.isfinite(cbr_cny) and np.isfinite(cbr_usd):
            cross = (
                _ratio(usd["close"] / cny["close"], cbr_usd / cbr_cny),
                cny["last_return"] - usd["last_return"],
                cny["open_return"] - usd["open_return"],
            )
        else:
            cross = (0.0, 0.0, 0.0)
        values.extend(cross)
        row_names.extend((
            "moex_hourly_spot_cross_basis",
            "moex_hourly_spot_last_1h_divergence",
            "moex_hourly_spot_open_to_cutoff_divergence",
        ))

        derived = []
        for ticker in TICKERS:
            spot, future = spot_last[ticker], futures_last[ticker]
            if spot is not None and future is not None:
                derived.extend((
                    _ratio(spot["close"], future["close"]),
                    spot["last_return"] - future["last_return"],
                    spot["open_return"] - future["open_return"],
                ))
            else:
                derived.extend((0.0, 0.0, 0.0))
        if all(spot_last[t] is not None and futures_last[t] is not None for t in TICKERS):
            spot_cross = spot_last["USD000UTSTOM"]["close"] / spot_last["CNYRUB_TOM"]["close"]
            future_cross = futures_last["USD000UTSTOM"]["close"] / futures_last["CNYRUB_TOM"]["close"]
            derived.append(_ratio(spot_cross, future_cross))
        else:
            derived.append(0.0)
        values.extend(derived)
        row_names.extend((
            "moex_hourly_spot_perpetual_cny_basis",
            "moex_hourly_spot_perpetual_cny_last_1h_divergence",
            "moex_hourly_spot_perpetual_cny_open_divergence",
            "moex_hourly_spot_perpetual_usd_basis",
            "moex_hourly_spot_perpetual_usd_last_1h_divergence",
            "moex_hourly_spot_perpetual_usd_open_divergence",
            "moex_hourly_spot_perpetual_cross_basis_divergence",
        ))
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("hourly spot feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if matrix.shape[1] != SPOT_ONLY_FEATURES + 7:
        raise AssertionError(f"unexpected spot feature width {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite hourly spot feature")
    return matrix, names or []


def causality_check(
    index, history, references, perpetual_history,
    cutoff=dt.date(2025, 6, 30),
):
    full, names = build_spot_features(index, history, references, perpetual_history)
    after_noon = {}
    for ticker, rows in history.items():
        after_noon[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["begin"].date() == cutoff and row["begin"].time() >= DECISION_TIME:
                for key in ("open", "close", "high", "low"):
                    clone[key] *= 100.0
            after_noon[ticker].append(clone)
    noon_altered, noon_names = build_spot_features(
        index, after_noon, references, perpetual_history,
    )
    through_cutoff = np.asarray([row[2] <= cutoff for row in index])
    if names != noon_names or not np.array_equal(
        full[through_cutoff], noon_altered[through_cutoff],
    ):
        raise AssertionError("noon-or-later spot candle entered noon features")

    changed = {}
    cutoff_time = dt.datetime.combine(cutoff, DECISION_TIME)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["end"] >= cutoff_time:
                for key in ("open", "close", "high", "low"):
                    clone[key] *= 100.0
            changed[ticker].append(clone)
    altered, altered_names = build_spot_features(
        index, changed, references, perpetual_history,
    )
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future spot candle changed a past noon feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future spot corruption did not affect future rows")
    return True
