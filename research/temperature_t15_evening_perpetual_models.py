"""Frozen causal helpers for T15 evening perpetual-FX updates."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS
from research.after_publication_ap50_temperature_models import clipped_logit
from research.round6_moex_perpetual_hourly_features import TICKERS, _arrays
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE, SEED


BASE_CLOCK = dt.time(18, 30)
CLOCKS = tuple(dt.time(hour, 0) for hour in (20, 21, 22, 23))
PER_TICKER_NAMES = (
    "return_from_1830", "last_hour_return", "new_range",
    "new_log_volume", "new_completed_bars",
)
FEATURE_NAMES = tuple(
    f"{ticker.lower()}_{name}"
    for ticker in TICKERS for name in PER_TICKER_NAMES
) + (
    "return_divergence", "last_hour_divergence", "range_divergence",
)


def clock_name(clock: dt.time) -> str:
    return f"perp_{clock.hour:02d}{clock.minute:02d}"


def _eligible(item, day, clock):
    midnight = dt.datetime.combine(day, dt.time())
    cutoff = dt.datetime.combine(day, clock)
    start = int(np.searchsorted(item["begin"], midnight, side="left"))
    stop = int(np.searchsorted(item["end"], cutoff, side="left"))
    rows = np.arange(start, stop, dtype=int)
    if not len(rows):
        return rows
    nominal_end = np.asarray([
        item["begin"][i] + dt.timedelta(hours=1) for i in rows
    ], dtype=object)
    return rows[
        (item["begin"][rows] >= midnight)
        & (item["end"][rows] < cutoff)
        & (nominal_end <= cutoff)
    ]


def build_evening_features(panel, history, clock):
    """Return CNY/USD features using only completed bars newer than 18:30."""
    arrays = _arrays(history)
    matrix = np.zeros((len(panel), len(FEATURE_NAMES)), dtype=np.float32)
    cny_available = np.zeros(len(panel), dtype=bool)
    dual_available = np.zeros(len(panel), dtype=bool)
    cny_source = np.full(len(panel), None, dtype=object)
    dual_source = np.full(len(panel), None, dtype=object)
    for row, day in enumerate(panel.date):
        states = {}
        for ticker in TICKERS:
            item = arrays[ticker]
            base_rows = _eligible(item, day, BASE_CLOCK)
            query_rows = _eligible(item, day, clock)
            if not len(base_rows) or not len(query_rows):
                states[ticker] = None
                continue
            new_rows = query_rows[
                item["end"][query_rows] > item["end"][base_rows[-1]]]
            if not len(new_rows):
                states[ticker] = None
                continue
            anchor = float(item["close"][base_rows[-1]])
            close = float(item["close"][new_rows[-1]])
            high = float(np.max(item["high"][new_rows]))
            low = float(np.min(item["low"][new_rows]))
            last_open = float(item["open"][new_rows[-1]])
            values = np.asarray([
                10000. * np.log(close / anchor),
                10000. * np.log(close / last_open),
                10000. * np.log(high / low),
                np.log1p(float(np.sum(item["volume"][new_rows]))),
                float(len(new_rows)),
            ], dtype=float)
            states[ticker] = {
                "values": values, "source": item["end"][new_rows[-1]],
            }
        offset = 0
        for ticker in TICKERS:
            if states[ticker] is not None:
                matrix[row, offset:offset + 5] = states[ticker]["values"]
            offset += 5
        cny, usd = states["CNYRUBF"], states["USDRUBF"]
        if cny is not None:
            cny_available[row] = True
            cny_source[row] = cny["source"]
        if cny is not None and usd is not None:
            dual_available[row] = True
            matrix[row, 10:] = (
                cny["values"][:3] - usd["values"][:3])
            dual_source[row] = max(cny["source"], usd["source"])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite evening perpetual feature")
    return matrix, cny_available, dual_available, cny_source, dual_source


def candidate_features(base_probability, market, currencies, candidate):
    currencies = np.asarray(currencies)
    base = clipped_logit(base_probability)[:, None]
    dummies = np.column_stack([
        (currencies == currency).astype(float) for currency in CORRIDORS
    ])
    if candidate == "perp_cny_logit":
        chosen = market[:, :5]
    elif candidate in {"perp_dual_logit", "perp_dual_hgb"}:
        chosen = market
    else:
        raise ValueError(candidate)
    return np.column_stack([base, chosen, dummies])


def fit_quarterly_classifier(features, target, maturity, dates, available,
                             model_kind):
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
        end = (pd.Timestamp(origin).to_period("Q") + 1).start_time.date()
        query = ((dates >= origin) & (dates < end) & available
                 & np.all(np.isfinite(features), axis=1))
        if not query.any():
            continue
        cutoff = origin - dt.timedelta(days=2)
        train = ((dates >= MIN_TRAIN_DATE) & (dates < origin) & available
                 & np.asarray([value < cutoff for value in maturity])
                 & np.isfinite(target)
                 & np.all(np.isfinite(features), axis=1))
        ids = np.flatnonzero(train)
        n_train[query] = len(ids)
        threshold = 500 if model_kind == "hgb" else 200
        if len(ids) < threshold or np.unique(target[train]).size < 2:
            logs.append({
                "origin": str(origin), "n_train": int(len(ids)),
                "n_query": int(query.sum()), "model_fit": False,
                "model_kind": model_kind, "last_target_maturity": "",
            })
            continue
        age = np.asarray([(origin - dates[i]).days for i in ids], dtype=float)
        weight = np.exp2(-age / 730.)
        if model_kind == "logit":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=.5, solver="lbfgs", max_iter=1000),
            )
            model.fit(features[train], target[train].astype(int),
                      logisticregression__sample_weight=weight)
        elif model_kind == "hgb":
            model = HistGradientBoostingClassifier(
                max_iter=180, learning_rate=.035, max_leaf_nodes=7,
                min_samples_leaf=55, l2_regularization=20.,
                random_state=SEED)
            model.fit(features[train], target[train].astype(int),
                      sample_weight=weight)
        else:
            raise ValueError(model_kind)
        probability[query] = model.predict_proba(features[query])[:, 1]
        logs.append({
            "origin": str(origin), "n_train": int(len(ids)),
            "n_positive": int(target[train].sum()),
            "n_query": int(query.sum()), "model_fit": True,
            "model_kind": model_kind,
            "last_target_maturity": str(max(maturity[train])),
        })
    return probability, n_train, logs


def physical_causality_check(panel, history, clock,
                             boundary=dt.date(2025, 1, 6)):
    original = build_evening_features(panel, history, clock)[0]
    boundary_time = dt.datetime.combine(boundary, clock)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for item in rows:
            clone = dict(item)
            if item["end"] >= boundary_time:
                for key in ("open", "close", "high", "low", "volume"):
                    clone[key] *= 11.
            changed[ticker].append(clone)
    altered = build_evening_features(panel, changed, clock)[0]
    past = np.asarray([day <= boundary for day in panel.date])
    np.testing.assert_array_equal(original[past], altered[past])
    if not np.any(original[~past] != altered[~past]):
        raise AssertionError(f"future corruption inert for {clock}")
    return True
