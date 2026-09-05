"""Causal state features from targets that have already fully resolved.

At date t, an older h=5 outcome is visible only when its fifth future
publication is strictly earlier than t.  These are runtime state variables,
not leaked labels from the row being predicted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ml.data import CORRIDORS, Series
from research.extended_features import LONG_DATA, load_or_build


CACHE = Path("research/cache/round6_resolved_features.npz")
WINDOWS = (5, 10, 20, 40, 60, 120)


def _outcome_history(item: Series):
    values = item.values
    n = max(0, len(values) - 5)
    outcome = np.empty(n, dtype=float)
    margin = np.empty(n, dtype=float)
    benefit = np.empty(n, dtype=float)
    for i in range(n):
        future = values[i + 1:i + 6]
        outcome[i] = float(values[i] <= np.min(future))
        margin[i] = float(np.log(np.min(future) / values[i]) * 10000.0)
        benefit[i] = float((np.mean(future) / values[i] - 1.0) * 10000.0)
    return {
        "reach": item.dates[5:].copy(),
        "outcome": outcome,
        "margin": margin,
        "benefit": benefit,
    }


def _ew_mean(values, half_life):
    if not len(values):
        return 0.0
    take = min(len(values), max(20, int(half_life * 6)))
    part = np.asarray(values[-take:], dtype=float)
    ages = np.arange(len(part) - 1, -1, -1, dtype=float)
    weights = np.power(.5, ages / half_life)
    return float(np.dot(part, weights) / weights.sum())


def _streak(values, state):
    count = 0
    for value in values[::-1]:
        if int(value) != state:
            break
        count += 1
    return float(count)


def _state(history, day):
    reach = history["reach"]
    n = int(np.searchsorted(reach, day, side="left"))
    outcome = history["outcome"][:n]
    margin = history["margin"][:n]
    benefit = history["benefit"][:n]
    result = {}
    for lag in (1, 2, 3, 5, 10):
        result[f"resolved_y_lag_{lag}"] = float(outcome[-lag]) if len(outcome) >= lag else .5
        result[f"resolved_margin_lag_{lag}"] = float(margin[-lag]) if len(margin) >= lag else 0.0
    for window in WINDOWS:
        y_part = outcome[-window:]
        m_part = margin[-window:]
        b_part = benefit[-window:]
        result[f"resolved_rate_{window}"] = float(np.mean(y_part)) if len(y_part) else .5
        result[f"resolved_margin_mean_{window}"] = float(np.mean(m_part)) if len(m_part) else 0.0
        result[f"resolved_margin_std_{window}"] = float(np.std(m_part)) if len(m_part) else 0.0
        result[f"resolved_benefit_mean_{window}"] = float(np.mean(b_part)) if len(b_part) else 0.0
    for half_life in (5, 10, 20, 60):
        result[f"resolved_rate_ew_{half_life}"] = _ew_mean(outcome, half_life)
        result[f"resolved_margin_ew_{half_life}"] = _ew_mean(margin, half_life)
    result["resolved_rate_delta_5_20"] = result["resolved_rate_5"] - result["resolved_rate_20"]
    result["resolved_rate_delta_10_60"] = result["resolved_rate_10"] - result["resolved_rate_60"]
    result["resolved_margin_delta_5_20"] = (
        result["resolved_margin_mean_5"] - result["resolved_margin_mean_20"]
    )
    result["resolved_streak_one"] = _streak(outcome, 1)
    result["resolved_streak_zero"] = _streak(outcome, 0)
    if len(outcome) and np.any(outcome == 1):
        last = int(np.where(outcome == 1)[0][-1])
        result["resolved_since_one"] = float(len(outcome) - 1 - last)
        result["resolved_days_since_one"] = float((day - reach[last]).days)
    else:
        result["resolved_since_one"] = 120.0
        result["resolved_days_since_one"] = 180.0
    if len(outcome) >= 2:
        result["resolved_transition_rate_20"] = float(
            np.mean(np.diff(outcome[-21:]) != 0)
        )
    else:
        result["resolved_transition_rate_20"] = 0.0
    result["resolved_log_count"] = float(np.log1p(len(outcome)))
    return result


def build_resolved_features(series: dict[str, Series], index: list[tuple]):
    histories = {code: _outcome_history(series[code]) for code in CORRIDORS}
    state_cache = {}
    rows = []
    names = None
    for currency, _position, day in index:
        day_states = state_cache.get(day)
        if day_states is None:
            day_states = {code: _state(histories[code], day) for code in CORRIDORS}
            state_cache[day] = day_states
        own = day_states[currency]
        row = dict(own)
        for window in (5, 20, 60):
            values = np.asarray([
                day_states[code][f"resolved_rate_{window}"] for code in CORRIDORS
            ])
            row[f"panel_resolved_rate_mean_{window}"] = float(np.mean(values))
            row[f"panel_resolved_rate_std_{window}"] = float(np.std(values))
            row[f"panel_resolved_rate_min_{window}"] = float(np.min(values))
            row[f"panel_resolved_rate_max_{window}"] = float(np.max(values))
            row[f"panel_resolved_rate_breadth_{window}"] = float(np.mean(values >= .5))
            row[f"own_minus_panel_resolved_rate_{window}"] = float(
                own[f"resolved_rate_{window}"] - np.mean(values)
            )
        for half_life in (5, 20):
            values = np.asarray([
                day_states[code][f"resolved_rate_ew_{half_life}"] for code in CORRIDORS
            ])
            row[f"panel_resolved_rate_ew_mean_{half_life}"] = float(np.mean(values))
            row[f"panel_resolved_rate_ew_std_{half_life}"] = float(np.std(values))
        row_names = sorted(row)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("resolved feature schema changed")
        rows.append([row[name] for name in row_names])
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite resolved-outcome feature")
    return matrix, names or []


def _fingerprint():
    return hashlib.sha256(LONG_DATA.read_bytes() + b":round6-resolved-v1").hexdigest()


def load_resolved_features(rebuild=False):
    X, names, index, series = load_or_build()
    fingerprint = _fingerprint()
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=True)
        if str(cached["fingerprint"].item()) == fingerprint:
            return X, names, index, series, cached["resolved"], list(cached["resolved_names"])
    resolved, resolved_names = build_resolved_features(series, index)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, resolved=resolved,
        resolved_names=np.asarray(resolved_names, dtype=object),
        fingerprint=np.asarray(fingerprint, dtype=object),
    )
    return X, names, index, series, resolved, resolved_names


def causality_check(cut):
    _X, _names, index, series, full, full_names = load_resolved_features()
    truncated = {}
    for code, item in series.items():
        keep = np.asarray([day <= cut for day in item.dates])
        truncated[code] = Series(code, item.dates[keep].copy(), item.values[keep].copy())
    cut_index = [row for row in index if row[2] <= cut]
    rebuilt, rebuilt_names = build_resolved_features(truncated, cut_index)
    lookup = {row: i for i, row in enumerate(index)}
    expected = np.asarray([lookup[row] for row in cut_index], dtype=int)
    if full_names != rebuilt_names:
        raise AssertionError("resolved feature names changed after future truncation")
    np.testing.assert_array_equal(full[expected], rebuilt)


if __name__ == "__main__":
    *_, matrix, feature_names = load_resolved_features(rebuild=True)
    print({"rows": len(matrix), "features": len(feature_names)})
