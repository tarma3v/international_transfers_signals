"""Strictly lagged features for predeclared MOEX risk/liquidity context."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path("data/moex_market_context_2010_2026.json")
MANIFEST = Path("results/research/round6/moex_context/data_manifest.json")
TICKERS = ("IMOEX", "RGBI", "RUSFAR", "GLDRUB_TOM")
RETURN_LAGS = (1, 2, 5, 10, 20)
VOL_WINDOWS = (5, 20, 60)
LEVEL_WINDOWS = (20, 60, 120)
ACTIVITY_LAGS = (1, 5, 20)
MAX_STALENESS_DAYS = 7


def load_context_history(path=DATA):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if Path(path) == DATA and MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["payload_sha256"]
        if digest != expected:
            raise AssertionError("MOEX context payload digest changed")
    result = {}
    for item in payload["instruments"]:
        columns = list(item["columns"])
        pos = {name: columns.index(name) for name in columns}
        activity_name = "NUMTRADES" if "NUMTRADES" in pos else "VALUE"
        rows = []
        for raw in item["rows"]:
            close = raw[pos["CLOSE"]]
            activity = raw[pos[activity_name]]
            if close is None or float(close) <= 0:
                continue
            def number(name, default=np.nan):
                value = raw[pos[name]] if name in pos else None
                return float(value) if value is not None else float(default)
            rows.append({
                "date": dt.date.fromisoformat(raw[pos["TRADEDATE"]]),
                "open": number("OPEN"), "high": number("HIGH"),
                "low": number("LOW"), "close": float(close),
                "activity": float(activity) if activity is not None and float(activity) > 0 else np.nan,
                "yield": number("YIELD"),
            })
        result[item["ticker"]] = rows
    if tuple(result) != TICKERS:
        raise AssertionError(f"context instrument order changed: {tuple(result)}")
    return result, digest


def _bps(a, b):
    return float(np.log(a / b) * 10000.0) if a > 0 and b > 0 else 0.0


def build_context_features(index, history):
    arrays = {}
    for ticker in TICKERS:
        rows = history[ticker]
        arrays[ticker] = {
            key: np.asarray([row[key] for row in rows], dtype=object if key == "date" else float)
            for key in ("date", "open", "high", "low", "close", "activity", "yield")
        }
    result, names = [], None
    for _currency, _position, day in index:
        values, row_names = [], []
        for ticker in TICKERS:
            item = arrays[ticker]
            end = int(np.searchsorted(item["date"], day, side="left"))
            prefix = f"context_{ticker.lower()}"
            age = (day - item["date"][end - 1]).days if end else 999
            available = bool(end and age <= MAX_STALENESS_DAYS)
            close = item["close"][:end]
            for lag in RETURN_LAGS:
                value = _bps(float(close[-1]), float(close[-1 - lag])) if available and len(close) > lag else 0.0
                values.append(value); row_names.append(f"{prefix}_ret_{lag}")
            returns = np.diff(np.log(close)) * 10000.0 if len(close) > 1 else np.asarray([])
            for window in VOL_WINDOWS:
                value = float(np.std(returns[-window:])) if available and len(returns) >= max(3, window // 4) else 0.0
                values.append(value); row_names.append(f"{prefix}_vol_{window}")
            if available:
                open_, high, low, last = (
                    float(item[key][end - 1]) for key in ("open", "high", "low", "close")
                )
                open_close = _bps(last, open_) if np.isfinite(open_) else 0.0
                intraday = _bps(high, low) if np.isfinite(high) and np.isfinite(low) else 0.0
            else:
                open_close, intraday = 0.0, 0.0
            values.extend([open_close, intraday])
            row_names.extend([f"{prefix}_open_close", f"{prefix}_intraday_range"])
            for window in LEVEL_WINDOWS:
                recent = close[-window:]
                if available and len(recent) >= max(5, window // 4):
                    mean, std = float(np.mean(recent)), float(np.std(recent))
                    value = float((close[-1] - mean) / std) if std > 1e-12 else 0.0
                else:
                    value = 0.0
                values.append(value); row_names.append(f"{prefix}_level_z_{window}")
            activity = item["activity"][:end]
            if available and len(activity) and np.isfinite(activity[-1]):
                log_activity = float(np.log1p(activity[-1]))
            else:
                log_activity = 0.0
            values.append(log_activity); row_names.append(f"{prefix}_log_activity")
            for lag in ACTIVITY_LAGS:
                if (available and len(activity) > lag and np.isfinite(activity[-1])
                        and np.isfinite(activity[-1 - lag])):
                    value = float(np.log1p(activity[-1]) - np.log1p(activity[-1 - lag]))
                else:
                    value = 0.0
                values.append(value); row_names.append(f"{prefix}_activity_change_{lag}")
            yields = item["yield"][:end]
            current_yield = float(yields[-1]) if available and len(yields) and np.isfinite(yields[-1]) else 0.0
            values.append(current_yield); row_names.append(f"{prefix}_yield")
            for lag in (5, 20):
                if (available and len(yields) > lag and np.isfinite(yields[-1])
                        and np.isfinite(yields[-1 - lag])):
                    value = float(yields[-1] - yields[-1 - lag])
                else:
                    value = 0.0
                values.append(value); row_names.append(f"{prefix}_yield_change_{lag}")
            values.extend([float(min(age, 30)), float(not available)])
            row_names.extend([f"{prefix}_age_days", f"{prefix}_missing"])
        result.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("MOEX context feature schema changed")
    matrix = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite MOEX context feature")
    return matrix, names or []


def causality_check(index, history, cutoff=dt.date(2025, 6, 30)):
    full, names = build_context_features(index, history)
    changed = {}
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row["date"] >= cutoff:
                for key in ("open", "high", "low", "close", "activity", "yield"):
                    if np.isfinite(clone[key]):
                        clone[key] *= 100.0
            changed[ticker].append(clone)
    corrupted, changed_names = build_context_features(index, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != changed_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future context changed a past feature")
    return True
