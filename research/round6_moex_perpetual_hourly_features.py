"""Noon-Moscow features from official MOEX hourly perpetual-FX candles."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_perpetual_fx_hourly_2022_2026.json")
MANIFEST = Path("results/research/round6/moex_perpetual_hourly/data_manifest.json")
TICKERS = ("CNYRUBF", "USDRUBF")
REFERENCE = {"CNYRUBF": "CNY", "USDRUBF": "USD"}
DECISION_TIME = dt.time(12, 0)


def load_hourly_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX hourly futures payload digest changed")
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
                "volume": float(raw[position["volume"]] or 0.0),
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
    result = {}
    for ticker in TICKERS:
        result[ticker] = {
            key: np.asarray(
                [row[key] for row in history[ticker]],
                dtype=object if key in ("begin", "end") else float,
            )
            for key in ("begin", "end", "open", "close", "high", "low", "volume")
        }
    return result


def build_hourly_features(index, history, references):
    arrays = _arrays(history)
    result, names = [], None
    for _currency, _position, day in index:
        midnight = dt.datetime.combine(day, dt.time())
        cutoff = dt.datetime.combine(day, DECISION_TIME)
        values, row_names, last = [], [], {}
        for ticker in TICKERS:
            item = arrays[ticker]
            start = int(np.searchsorted(item["begin"], midnight, side="left"))
            stop = int(np.searchsorted(item["end"], cutoff, side="left"))
            session_rows = np.arange(start, stop, dtype=int)
            session_rows = session_rows[item["begin"][session_rows] >= midnight]
            previous_stop = int(np.searchsorted(item["end"], midnight, side="left"))
            previous_close = (
                float(item["close"][previous_stop - 1]) if previous_stop else np.nan
            )
            prefix = f"moex_hourly_{ticker.lower()}"
            if len(session_rows):
                opens = item["open"][session_rows]
                closes = item["close"][session_rows]
                highs = item["high"][session_rows]
                lows = item["low"][session_rows]
                volumes = item["volume"][session_rows]
                open_ = float(opens[0])
                close = float(closes[-1])
                high = float(np.max(highs))
                low = float(np.min(lows))
                returns = np.diff(np.log(closes)) * 10000.0
                slope = (
                    float(np.polyfit(np.arange(len(closes)), np.log(closes), 1)[0] * 10000.0)
                    if len(closes) > 1 else 0.0
                )
                position = (close - low) / (high - low) if high > low else .5
                reference = _reference_last(references[REFERENCE[ticker]], day)
                feature = (
                    close,
                    _ratio(close, previous_close) if np.isfinite(previous_close) else 0.0,
                    _ratio(close, open_),
                    _ratio(close, float(closes[-2])) if len(closes) > 1 else 0.0,
                    _ratio(close, float(closes[-3])) if len(closes) > 2 else 0.0,
                    _ratio(high, low),
                    float(np.std(returns)) if len(returns) else 0.0,
                    float(np.log1p(np.sum(volumes))),
                    float(len(closes)),
                    slope,
                    float(position),
                    _ratio(close, reference) if np.isfinite(reference) else 0.0,
                    float((cutoff - item["end"][session_rows[-1]]).total_seconds() / 3600.0),
                    0.0,
                )
                last[ticker] = {
                    "close": close,
                    "open_return": feature[2],
                    "last_return": feature[3],
                }
            else:
                previous_end = item["end"][previous_stop - 1] if previous_stop else None
                age = (
                    min((cutoff - previous_end).total_seconds() / 3600.0, 720.0)
                    if previous_end is not None else 720.0
                )
                feature = (0.0,) * 12 + (float(age), 1.0)
                last[ticker] = None
            values.extend(feature)
            row_names.extend((
                f"{prefix}_last",
                f"{prefix}_overnight_return",
                f"{prefix}_open_to_cutoff_return",
                f"{prefix}_last_1h_return",
                f"{prefix}_last_2h_return",
                f"{prefix}_cutoff_range",
                f"{prefix}_realized_vol",
                f"{prefix}_log_volume",
                f"{prefix}_completed_candles",
                f"{prefix}_log_price_slope",
                f"{prefix}_range_position",
                f"{prefix}_cbr_basis",
                f"{prefix}_age_hours",
                f"{prefix}_missing",
            ))

        cny, usd = last["CNYRUBF"], last["USDRUBF"]
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
            "moex_hourly_perpetual_cross_basis",
            "moex_hourly_perpetual_last_1h_divergence",
            "moex_hourly_perpetual_open_to_cutoff_divergence",
        ))
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("hourly perpetual feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite hourly perpetual feature")
    return matrix, names or []


def causality_check(index, history, references, cutoff=dt.date(2025, 6, 30)):
    full, names = build_hourly_features(index, history, references)

    # Candles at noon or later must be invisible even on their own date.
    after_noon = {}
    for ticker, rows in history.items():
        after_noon[ticker] = []
        for row in rows:
            clone = dict(row)
            if (
                row["begin"].date() == cutoff
                and row["begin"].time() >= DECISION_TIME
            ):
                for key in ("open", "close", "high", "low", "volume"):
                    clone[key] *= 100.0
            after_noon[ticker].append(clone)
    noon_altered, noon_names = build_hourly_features(index, after_noon, references)
    through_cutoff = np.asarray([row[2] <= cutoff for row in index])
    if names != noon_names or not np.array_equal(
        full[through_cutoff], noon_altered[through_cutoff],
    ):
        raise AssertionError("noon-or-later candle entered a noon feature")

    changed = {}
    cutoff_time = dt.datetime.combine(cutoff, DECISION_TIME)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["end"] >= cutoff_time:
                for key in ("open", "close", "high", "low", "volume"):
                    clone[key] *= 100.0
            changed[ticker].append(clone)
    altered, altered_names = build_hourly_features(index, changed, references)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != altered_names or not np.array_equal(full[past], altered[past]):
        raise AssertionError("future hourly candle changed a past noon feature")
    if not np.any(full[~past] != altered[~past]):
        raise AssertionError("future hourly corruption did not affect future rows")
    return True
