"""Frozen early-session feature builder for T9."""
from __future__ import annotations

import datetime as dt

import numpy as np

from research.round6_moex_spot_1530_features import (
    REFERENCE,
    TICKERS,
    _arrays,
    _ratio,
    _reference_last,
)


SESSION_START = dt.time(6, 50)
CLOCKS = (
    dt.time(7, 30), dt.time(8, 0), dt.time(8, 30),
    dt.time(9, 0), dt.time(9, 30), dt.time(10, 0),
)
BASE_NAMES = (
    "mean_cbr_basis", "last_cbr_basis", "overnight_return",
    "open_to_cutoff_return", "last_10m_return", "last_30m_return",
    "last_60m_return", "cutoff_range", "realized_vol", "log_price_slope",
    "range_position", "completed_candles", "age_hours", "missing",
)


def clock_name(clock: dt.time) -> str:
    return f"early_{clock.hour:02d}{clock.minute:02d}"


def _ticker_state(item, day, cutoff, reference):
    start_time = dt.datetime.combine(day, SESSION_START)
    cutoff_time = dt.datetime.combine(day, cutoff)
    start = int(np.searchsorted(item["begin"], start_time, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff_time, side="left"))
    rows = np.arange(start, stop, dtype=int)
    if len(rows):
        nominal_end = np.asarray([
            item["begin"][i] + dt.timedelta(minutes=10) for i in rows
        ], dtype=object)
        rows = rows[
            (item["begin"][rows] >= start_time)
            & (item["end"][rows] < cutoff_time)
            & (nominal_end <= cutoff_time)
        ]
    previous_stop = int(np.searchsorted(
        item["end"], start_time, side="left"))
    previous_close = (
        float(item["close"][previous_stop - 1]) if previous_stop else np.nan)
    previous_end = item["end"][previous_stop - 1] if previous_stop else None
    if not len(rows):
        age = min((cutoff_time - previous_end).total_seconds() / 3600., 720.) \
            if previous_end is not None else 720.
        return (0.,) * 12 + (float(age), 1.), None, None

    opens = item["open"][rows]
    closes = item["close"][rows]
    highs = item["high"][rows]
    lows = item["low"][rows]
    open_, close = float(opens[0]), float(closes[-1])
    mean_close = float(np.mean(closes))
    high, low = float(np.max(highs)), float(np.min(lows))
    returns = np.diff(np.log(closes)) * 10000.
    slope = (float(np.polyfit(
        np.arange(len(closes)), np.log(closes), 1)[0] * 10000.)
        if len(closes) > 1 else 0.)
    position = (close - low) / (high - low) if high > low else .5
    values = (
        _ratio(mean_close, reference) if np.isfinite(reference) else 0.,
        _ratio(close, reference) if np.isfinite(reference) else 0.,
        _ratio(open_, previous_close) if np.isfinite(previous_close) else 0.,
        _ratio(close, open_),
        _ratio(close, float(closes[-2])) if len(closes) > 1 else 0.,
        _ratio(close, float(closes[-4])) if len(closes) > 3 else 0.,
        _ratio(close, float(closes[-7])) if len(closes) > 6 else 0.,
        _ratio(high, low),
        float(np.std(returns)) if len(returns) else 0.,
        slope, float(position), float(len(rows)),
        float((cutoff_time - item["end"][rows[-1]]).total_seconds() / 3600.),
        0.,
    )
    compact = {
        "close": close, "mean": mean_close,
        "open_return": values[3], "last_return": values[4],
    }
    return values, compact, item["end"][rows[-1]]


def build_early_market_features(index, history, references, cutoff):
    arrays = _arrays(history)
    matrix, available, source_at = [], [], []
    names = []
    for ticker in TICKERS:
        names.extend(f"{ticker.lower()}_{name}" for name in BASE_NAMES)
    names.extend((
        "cross_last_cbr_basis", "cross_mean_cbr_basis",
        "cross_last_return_divergence", "cross_open_return_divergence",
    ))
    for _currency, _position, day in index:
        row, states, sources = [], {}, []
        for ticker in TICKERS:
            reference = _reference_last(references[REFERENCE[ticker]], day)
            values, states[ticker], source = _ticker_state(
                arrays[ticker], day, cutoff, reference)
            row.extend(values)
            if source is not None:
                sources.append(source)
        cny, usd = states["CNYRUB_TOM"], states["USD000UTSTOM"]
        cbr_cny = _reference_last(references["CNY"], day)
        cbr_usd = _reference_last(references["USD"], day)
        if (cny is not None and usd is not None and np.isfinite(cbr_cny)
                and np.isfinite(cbr_usd)):
            row.extend((
                _ratio(usd["close"] / cny["close"], cbr_usd / cbr_cny),
                _ratio(usd["mean"] / cny["mean"], cbr_usd / cbr_cny),
                cny["last_return"] - usd["last_return"],
                cny["open_return"] - usd["open_return"],
            ))
        else:
            row.extend((0., 0., 0., 0.))
        matrix.append(row)
        available.append(cny is not None)
        source_at.append(max(sources) if sources else None)
    result = np.asarray(matrix, dtype=np.float32)
    if result.shape != (len(index), 32) or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid early market matrix {result.shape}")
    return result, names, np.asarray(available), np.asarray(source_at, dtype=object)


def physical_causality_check(index, history, references, cutoff,
                             boundary=dt.date(2023, 7, 3)):
    original = build_early_market_features(
        index, history, references, cutoff)[0]
    boundary_time = dt.datetime.combine(boundary, cutoff)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for item in rows:
            clone = dict(item)
            if item["end"] >= boundary_time:
                for key in ("open", "close", "high", "low"):
                    clone[key] *= 7.
            changed[ticker].append(clone)
    altered = build_early_market_features(
        index, changed, references, cutoff)[0]
    past = np.asarray([row[2] <= boundary for row in index])
    np.testing.assert_array_equal(original[past], altered[past])
    if not np.any(original[~past] != altered[~past]):
        raise AssertionError(f"future corruption inert for {cutoff}")
    return True
