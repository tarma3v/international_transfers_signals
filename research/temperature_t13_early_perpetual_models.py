"""Frozen perpetual-FX prefix features and available-only HGB for T13."""
from __future__ import annotations

import datetime as dt

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from research.round6_moex_perpetual_hourly_features import (
    REFERENCE,
    TICKERS,
    _arrays,
    _ratio,
    _reference_last,
)
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE, SEED


CLOCKS = (dt.time(9, 0), dt.time(10, 0))
BASE_NAMES = (
    "mean_cbr_basis", "last_cbr_basis", "overnight_return",
    "open_to_cutoff_return", "last_1h_return", "last_2h_return",
    "cutoff_range", "realized_vol", "log_volume", "log_price_slope",
    "range_position", "completed_candles", "age_hours", "missing",
)


def clock_name(clock: dt.time) -> str:
    return f"perp_{clock.hour:02d}{clock.minute:02d}"


def _ticker_state(item, day, cutoff, reference):
    midnight = dt.datetime.combine(day, dt.time())
    cutoff_time = dt.datetime.combine(day, cutoff)
    start = int(np.searchsorted(item["begin"], midnight, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff_time, side="left"))
    rows = np.arange(start, stop, dtype=int)
    if len(rows):
        nominal_end = np.asarray([
            item["begin"][i] + dt.timedelta(hours=1) for i in rows
        ], dtype=object)
        rows = rows[
            (item["begin"][rows] >= midnight)
            & (item["end"][rows] < cutoff_time)
            & (nominal_end <= cutoff_time)
        ]
    previous_stop = int(np.searchsorted(item["end"], midnight, side="left"))
    previous_close = (
        float(item["close"][previous_stop - 1]) if previous_stop else np.nan)
    previous_end = item["end"][previous_stop - 1] if previous_stop else None
    if not len(rows):
        age = (min((cutoff_time - previous_end).total_seconds() / 3600., 720.)
               if previous_end is not None else 720.)
        return (0.,) * 12 + (float(age), 1.), None, None

    opens = item["open"][rows]
    closes = item["close"][rows]
    highs = item["high"][rows]
    lows = item["low"][rows]
    volumes = item["volume"][rows]
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
        _ratio(close, float(closes[-3])) if len(closes) > 2 else 0.,
        _ratio(high, low),
        float(np.std(returns)) if len(returns) else 0.,
        float(np.log1p(np.sum(volumes))),
        slope, float(position), float(len(rows)),
        float((cutoff_time - item["end"][rows[-1]]).total_seconds() / 3600.),
        0.,
    )
    compact = {
        "close": close, "mean": mean_close,
        "open_return": values[3], "last_return": values[4],
    }
    return values, compact, item["end"][rows[-1]]


def build_perpetual_prefix_features(index, history, references, cutoff):
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
        cny, usd = states["CNYRUBF"], states["USDRUBF"]
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
        raise ValueError(f"invalid perpetual prefix matrix {result.shape}")
    return (result, names, np.asarray(available),
            np.asarray(source_at, dtype=object))


def fit_quarterly_available_hgb(features, target, maturity, dates, available):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    maturity = np.asarray(maturity, dtype=object)
    dates = np.asarray(dates, dtype=object)
    available = np.asarray(available, dtype=bool)
    probability = np.full(len(dates), np.nan)
    n_train = np.zeros(len(dates), dtype=np.int32)
    logs = []
    origins = [
        dt.date(year, month, 1)
        for year in range(2023, max(day.year for day in dates) + 1)
        for month in (1, 4, 7, 10)
    ]
    for origin in origins:
        end_month = origin.month + 3
        end = (dt.date(origin.year + 1, end_month - 12, 1)
               if end_month > 12 else dt.date(origin.year, end_month, 1))
        query = ((dates >= origin) & (dates < end) & available
                 & np.all(np.isfinite(features), axis=1))
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        train = ((dates >= MIN_TRAIN_DATE) & (dates < origin) & available
                 & np.asarray([value < cutoff for value in maturity])
                 & np.isfinite(target) & np.all(np.isfinite(features), axis=1))
        ids = np.flatnonzero(train)
        n_train[query] = len(ids)
        if len(ids) < 500 or np.unique(target[train]).size < 2:
            logs.append({"origin": str(origin), "n_train": int(len(ids)),
                         "n_query": int(query.sum()), "model_fit": False,
                         "last_target_maturity": ""})
            continue
        model = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.035, max_leaf_nodes=9,
            min_samples_leaf=42, l2_regularization=15., random_state=SEED)
        age = np.asarray([(origin - dates[i]).days for i in ids], dtype=float)
        weight = np.exp2(-age / 730.)
        model.fit(features[train], target[train].astype(int),
                  sample_weight=weight)
        probability[query] = model.predict_proba(features[query])[:, 1]
        logs.append({
            "origin": str(origin), "n_train": int(len(ids)),
            "n_positive": int(target[train].sum()),
            "n_query": int(query.sum()), "model_fit": True,
            "last_target_maturity": str(max(maturity[train])),
        })
    return probability, n_train, logs


def physical_causality_check(index, history, references, cutoff,
                             boundary=dt.date(2024, 7, 1)):
    original = build_perpetual_prefix_features(
        index, history, references, cutoff)[0]
    boundary_time = dt.datetime.combine(boundary, cutoff)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for item in rows:
            clone = dict(item)
            if item["end"] >= boundary_time:
                for key in ("open", "close", "high", "low", "volume"):
                    clone[key] *= 7.
            changed[ticker].append(clone)
    altered = build_perpetual_prefix_features(
        index, changed, references, cutoff)[0]
    past = np.asarray([row[2] <= boundary for row in index])
    np.testing.assert_array_equal(original[past], altered[past])
    if not np.any(original[~past] != altered[~past]):
        raise AssertionError(f"future corruption inert for {cutoff}")
    return True
