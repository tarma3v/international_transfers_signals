"""Rich causal cross-section of the five target currencies for packet AC."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

from ml.data import CORRIDORS, Series


WINDOWS = (1, 2, 3, 5, 10, 20, 40, 60)
DEPENDENCE_WINDOWS = (10, 20, 60)
CACHE = Path("research/cache/round6_target_panel_features.npz")


def _fingerprint(series: dict[str, Series]) -> str:
    digest = hashlib.sha256()
    for code in CORRIDORS:
        item = series[code]
        digest.update(code.encode())
        digest.update("|".join(map(str, item.dates)).encode())
        digest.update(np.asarray(item.values, dtype=np.float64).tobytes())
    digest.update(str((WINDOWS, DEPENDENCE_WINDOWS)).encode())
    return digest.hexdigest()


def _return_maps(series):
    return {
        code: {
            day: float(value)
            for day, value in zip(
                series[code].dates[1:], np.diff(np.log(series[code].values)) * 10000.0,
            )
        }
        for code in CORRIDORS
    }


def _safe_dependence(own, common):
    own = np.asarray(own, dtype=float)
    common = np.asarray(common, dtype=float)
    own_std = float(np.std(own))
    common_var = float(np.var(common))
    correlation = (
        float(np.corrcoef(own, common)[0, 1])
        if own_std > 1e-12 and common_var > 1e-12 else 0.0
    )
    beta = float(np.cov(own, common, ddof=0)[0, 1] / common_var) if common_var > 1e-12 else 0.0
    residual_vol = float(np.std(own - beta * common))
    return correlation, beta, residual_vol


def build_target_panel_features(series: dict[str, Series], index: list[tuple]):
    dates = {code: np.asarray(series[code].dates, dtype=object) for code in CORRIDORS}
    returns = _return_maps(series)
    rows, names = [], None
    for target, _position, day in index:
        values, row_names = [], []
        histories = {}
        for code in CORRIDORS:
            end = int(np.searchsorted(dates[code], day, side="right"))
            histories[code] = np.asarray(series[code].values[:end], dtype=float)
        for window in WINDOWS:
            panel = {}
            for code in CORRIDORS:
                history = histories[code]
                panel[code] = (
                    float(np.log(history[-1] / history[-1 - window]) * 10000.0)
                    if len(history) > window else 0.0
                )
                values.append(panel[code])
                row_names.append(f"target_panel_{code}_ret_{window}")
            peers = np.asarray([panel[code] for code in CORRIDORS if code != target])
            target_value = panel[target]
            ordered = np.sort(np.asarray(list(panel.values()), dtype=float))
            summary = (
                float(np.mean(peers)), float(np.std(peers)), float(np.min(peers)),
                float(np.max(peers)), float(np.mean(peers > 0.0)),
                float(target_value - np.mean(peers)),
                float(np.searchsorted(ordered, target_value, side="right") / len(ordered)),
            )
            values.extend(summary)
            row_names.extend([
                f"target_panel_peer_mean_{window}",
                f"target_panel_peer_std_{window}",
                f"target_panel_peer_min_{window}",
                f"target_panel_peer_max_{window}",
                f"target_panel_peer_breadth_{window}",
                f"target_panel_relative_{window}",
                f"target_panel_rank_{window}",
            ])
        target_days = dates[target][dates[target] <= day]
        for window in DEPENDENCE_WINDOWS:
            selected_days = target_days[-window:]
            own = [returns[target].get(value, 0.0) for value in selected_days]
            common = [
                float(np.mean([returns[code].get(value, 0.0) for code in CORRIDORS]))
                for value in selected_days
            ]
            correlation, beta, residual_vol = _safe_dependence(own, common)
            values.extend([correlation, beta, residual_vol])
            row_names.extend([
                f"target_panel_corr_common_{window}",
                f"target_panel_beta_common_{window}",
                f"target_panel_residual_vol_{window}",
            ])
        rows.append(values)
        if names is None:
            names = row_names
        elif names != row_names:
            raise AssertionError("target-panel feature schema changed")
    matrix = np.asarray(rows, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite target-panel feature")
    return matrix, names or []


def load_target_panel_features(index, series, rebuild=False):
    fingerprint = _fingerprint(series)
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=True)
        if str(cached["fingerprint"].item()) == fingerprint:
            return cached["matrix"], list(cached["names"])
    matrix, names = build_target_panel_features(series, index)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, matrix=matrix, names=np.asarray(names, dtype=object),
        fingerprint=np.asarray(fingerprint, dtype=object),
    )
    return matrix, names


def causality_check(index, series, cutoff=dt.date(2025, 6, 30)):
    full, names = build_target_panel_features(series, index)
    changed = {}
    for code, item in series.items():
        values = item.values.copy()
        future = item.dates > cutoff
        values[future] *= np.linspace(2.0, 30.0, int(future.sum()))
        changed[code] = Series(code, item.dates.copy(), values)
    corrupted, corrupted_names = build_target_panel_features(changed, index)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != corrupted_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future target-panel rate changed a past feature")
    return True
