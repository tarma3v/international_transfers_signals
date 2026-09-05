"""Methodology-window 15:30 features from MOEX 10-minute spot candles."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_spot_fx_10min_2022_2026.json")
MANIFEST = Path("results/research/round6/moex_spot_1530/data_manifest.json")
TICKERS = ("CNYRUB_TOM", "USD000UTSTOM")
REFERENCE = {"CNYRUB_TOM": "CNY", "USD000UTSTOM": "USD"}
SESSION_START = dt.time(10, 0)
DECISION_TIME = dt.time(15, 30)


def load_spot_1530_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX 10-minute spot payload digest changed")
    history = {}
    for instrument in payload["instruments"]:
        columns = list(instrument["columns"])
        position = {name: columns.index(name) for name in columns}
        history[instrument["ticker"]] = [{
            "begin": dt.datetime.fromisoformat(raw[position["begin"]]),
            "end": dt.datetime.fromisoformat(raw[position["end"]]),
            "open": float(raw[position["open"]]),
            "close": float(raw[position["close"]]),
            "high": float(raw[position["high"]]),
            "low": float(raw[position["low"]]),
        } for raw in instrument["rows"]]
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


def _state(item, day, reference):
    start_time = dt.datetime.combine(day, SESSION_START)
    cutoff = dt.datetime.combine(day, DECISION_TIME)
    start = int(np.searchsorted(item["begin"], start_time, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff, side="left"))
    rows = np.arange(start, stop, dtype=int)
    rows = rows[item["begin"][rows] >= start_time]
    previous_stop = int(np.searchsorted(item["end"], start_time, side="left"))
    previous_close = float(item["close"][previous_stop - 1]) if previous_stop else np.nan
    if not len(rows):
        previous_end = item["end"][previous_stop - 1] if previous_stop else None
        age = (
            min((cutoff - previous_end).total_seconds() / 3600.0, 720.0)
            if previous_end is not None else 720.0
        )
        return (0.0,) * 14 + (float(age), 1.0), None
    opens, closes = item["open"][rows], item["close"][rows]
    highs, lows = item["high"][rows], item["low"][rows]
    open_, close = float(opens[0]), float(closes[-1])
    mean_close = float(np.mean(closes))
    high, low = float(np.max(highs)), float(np.min(lows))
    returns = np.diff(np.log(closes)) * 10000.0
    slope = (
        float(np.polyfit(np.arange(len(closes)), np.log(closes), 1)[0] * 10000.0)
        if len(closes) > 1 else 0.0
    )
    position = (close - low) / (high - low) if high > low else .5
    values = (
        close, mean_close,
        _ratio(open_, previous_close) if np.isfinite(previous_close) else 0.0,
        _ratio(close, open_),
        _ratio(close, float(closes[-2])) if len(closes) > 1 else 0.0,
        _ratio(close, float(closes[-4])) if len(closes) > 3 else 0.0,
        _ratio(close, float(closes[-7])) if len(closes) > 6 else 0.0,
        _ratio(high, low),
        float(np.std(returns)) if len(returns) else 0.0,
        float(len(closes)), slope, float(position),
        _ratio(close, reference) if np.isfinite(reference) else 0.0,
        _ratio(mean_close, reference) if np.isfinite(reference) else 0.0,
        float((cutoff - item["end"][rows[-1]]).total_seconds() / 3600.0),
        0.0,
    )
    compact = {
        "close": close, "mean": mean_close,
        "last_return": values[4], "open_return": values[3],
    }
    return values, compact


def build_spot_1530_features(index, history, references):
    arrays = _arrays(history)
    result, names = [], None
    base_names = (
        "last", "session_mean", "overnight_return", "open_to_cutoff_return",
        "last_10m_return", "last_30m_return", "last_60m_return",
        "cutoff_range", "realized_vol", "completed_candles",
        "log_price_slope", "range_position", "last_cbr_basis",
        "mean_cbr_basis", "age_hours", "missing",
    )
    for _currency, _position, day in index:
        values, row_names, last = [], [], {}
        for ticker in TICKERS:
            reference = _reference_last(references[REFERENCE[ticker]], day)
            feature, last[ticker] = _state(arrays[ticker], day, reference)
            values.extend(feature)
            prefix = f"moex_1530_{ticker.lower()}"
            row_names.extend(f"{prefix}_{name}" for name in base_names)
        cny, usd = last["CNYRUB_TOM"], last["USD000UTSTOM"]
        cbr_cny = _reference_last(references["CNY"], day)
        cbr_usd = _reference_last(references["USD"], day)
        if cny is not None and usd is not None and np.isfinite(cbr_cny) and np.isfinite(cbr_usd):
            cross = (
                _ratio(usd["close"] / cny["close"], cbr_usd / cbr_cny),
                _ratio(usd["mean"] / cny["mean"], cbr_usd / cbr_cny),
                cny["last_return"] - usd["last_return"],
                cny["open_return"] - usd["open_return"],
            )
        else:
            cross = (0.0, 0.0, 0.0, 0.0)
        values.extend(cross)
        row_names.extend((
            "moex_1530_cross_last_basis", "moex_1530_cross_mean_basis",
            "moex_1530_last_10m_divergence", "moex_1530_open_divergence",
        ))
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("15:30 spot feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if matrix.shape[1] != 36 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid 15:30 feature matrix {matrix.shape}")
    return matrix, names or []


def causality_check(index, history, references, cutoff=dt.date(2025, 6, 30)):
    full, names = build_spot_1530_features(index, history, references)
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
    altered, altered_names = build_spot_1530_features(index, changed, references)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("15:30/future candle changed an admissible feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future 10-minute corruption did not affect future rows")
    return True
