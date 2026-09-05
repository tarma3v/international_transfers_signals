"""Strictly lagged official-MOEX FX features for packet AE."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_fx_history_2010_2026.json")
MANIFEST = Path("results/research/round6/moex/data_manifest.json")
TICKERS = ("CNYRUB_TOM", "USD000UTSTOM", "EUR_RUB__TOM")
RETURN_WINDOWS = (1, 2, 5, 10, 20)
VOL_WINDOWS = (5, 20, 60)
TRADE_WINDOWS = (1, 5, 20)
MAX_STALENESS_DAYS = 7


def load_moex_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX payload digest changed")
    result = {}
    for item in payload["instruments"]:
        columns = list(item["columns"])
        position = {name: columns.index(name) for name in columns}
        rows = []
        for raw in item["rows"]:
            close = raw[position["CLOSE"]]
            trades = raw[position["NUMTRADES"]]
            # Zero rows after a market suspension are calendar placeholders,
            # not prices. They must never create artificial zero volatility.
            if close is None or float(close) <= 0.0 or trades is None or float(trades) <= 0.0:
                continue
            rows.append({
                "date": dt.date.fromisoformat(raw[position["TRADEDATE"]]),
                "open": float(raw[position["OPEN"]]),
                "high": float(raw[position["HIGH"]]),
                "low": float(raw[position["LOW"]]),
                "close": float(close),
                "waprice": float(raw[position["WAPRICE"]]) if raw[position["WAPRICE"]] else np.nan,
                "trades": float(trades),
            })
        result[item["ticker"]] = rows
    return result, digest


def _bps_ratio(numerator, denominator):
    return float((numerator / denominator - 1.0) * 10000.0) if denominator > 0 else 0.0


def build_moex_features(index, history):
    arrays = {}
    for ticker in TICKERS:
        rows = history[ticker]
        arrays[ticker] = {
            key: np.asarray([row[key] for row in rows], dtype=object if key == "date" else float)
            for key in ("date", "open", "high", "low", "close", "waprice", "trades")
        }
    result, names = [], None
    for _currency, _position, day in index:
        values, row_names = [], []
        for ticker in TICKERS:
            item = arrays[ticker]
            end = int(np.searchsorted(item["date"], day, side="left"))
            prefix = f"moex_{ticker.lower()}"
            age = (day - item["date"][end - 1]).days if end else 999
            available = bool(end and age <= MAX_STALENESS_DAYS)
            close = item["close"][:end]
            trades = item["trades"][:end]
            for window in RETURN_WINDOWS:
                value = (
                    _bps_ratio(float(close[-1]), float(close[-1 - window]))
                    if available and len(close) > window else 0.0
                )
                values.append(value)
                row_names.append(f"{prefix}_ret_{window}")
            returns = np.diff(np.log(close)) * 10000.0 if len(close) > 1 else np.asarray([])
            for window in VOL_WINDOWS:
                value = float(np.std(returns[-window:])) if available and len(returns) else 0.0
                values.append(value)
                row_names.append(f"{prefix}_vol_{window}")
            if available:
                open_, high, low = (float(item[key][end - 1]) for key in ("open", "high", "low"))
                last_close = float(close[-1])
                waprice = float(item["waprice"][end - 1])
                daily = [
                    _bps_ratio(last_close, open_) if open_ > 0 else 0.0,
                    _bps_ratio(high, low) if low > 0 else 0.0,
                    _bps_ratio(last_close, waprice) if np.isfinite(waprice) and waprice > 0 else 0.0,
                    _bps_ratio(open_, float(close[-2])) if len(close) > 1 else 0.0,
                    float(np.log1p(trades[-1])), float(min(age, 30)),
                ]
            else:
                daily = [0.0, 0.0, 0.0, 0.0, 0.0, float(min(age, 30))]
            values.extend(daily)
            row_names.extend([
                f"{prefix}_open_close", f"{prefix}_intraday_range",
                f"{prefix}_close_wap", f"{prefix}_overnight_gap",
                f"{prefix}_log_trades", f"{prefix}_age_days",
            ])
            for window in TRADE_WINDOWS:
                value = (
                    float(np.log1p(trades[-1]) - np.log1p(trades[-1 - window]))
                    if available and len(trades) > window else 0.0
                )
                values.append(value)
                row_names.append(f"{prefix}_log_trades_change_{window}")
            values.append(float(not available))
            row_names.append(f"{prefix}_missing")
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("MOEX feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite MOEX feature")
    return matrix, names or []


def causality_check(index, history, cutoff=dt.date(2025, 6, 30)):
    full, names = build_moex_features(index, history)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["date"] >= cutoff:
                for key in ("open", "high", "low", "close", "waprice", "trades"):
                    if np.isfinite(clone[key]):
                        clone[key] *= 100.0
            changed[ticker].append(clone)
    corrupted, corrupted_names = build_moex_features(index, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != corrupted_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future MOEX values changed a past feature")
    return True
