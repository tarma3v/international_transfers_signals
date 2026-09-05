"""Causal features from a target-free panel of official CBR FX rates."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.data import Series


DATA = Path("data/cbr_broad_reference_2010_2026.json")
CACHE = Path("research/cache/round6_broad_cbr_features.npz")
LAGS = (1, 2, 5, 10, 20, 60)
VOL_WINDOWS = (20, 60)


def load_reference(path: Path = DATA) -> dict[str, Series]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for iso, item in payload["currencies"].items():
        rows = item["rows"]
        result[iso] = Series(
            iso,
            np.asarray([dt.date.fromisoformat(row["date"]) for row in rows], dtype=object),
            np.asarray([float(row["rub_per_unit"]) for row in rows], dtype=float),
        )
    return result


def _return(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return np.nan
    return float(np.log(values[-1] / values[-1 - lag]) * 10000.0)


def _volatility(values: np.ndarray, window: int) -> float:
    returns = np.diff(np.log(values[-(window + 1):])) * 10000.0
    return float(np.std(returns)) if len(returns) >= max(5, window // 4) else np.nan


def _aggregate(values: np.ndarray, prefix: str, row: dict[str, float]) -> None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        row[f"{prefix}_mean"] = 0.0
        row[f"{prefix}_median"] = 0.0
        row[f"{prefix}_dispersion"] = 0.0
        row[f"{prefix}_positive_breadth"] = 0.5
        row[f"{prefix}_coverage"] = 0.0
        return
    row[f"{prefix}_mean"] = float(np.mean(finite))
    row[f"{prefix}_median"] = float(np.median(finite))
    row[f"{prefix}_dispersion"] = float(np.std(finite))
    row[f"{prefix}_positive_breadth"] = float(np.mean(finite > 0.0))
    row[f"{prefix}_coverage"] = float(len(finite) / len(values))


def build_broad_features(
    reference: dict[str, Series],
    target_series: dict[str, Series],
    index: list[tuple],
) -> tuple[np.ndarray, list[str]]:
    """Build features with an as-of join at each target publication date."""
    codes = sorted(reference)
    if "USD" not in reference:
        raise ValueError("USD required to remove the common RUB leg")
    rows: list[dict[str, float]] = []
    for target_code, target_position, day in index:
        history: dict[str, np.ndarray] = {}
        for code in codes:
            series = reference[code]
            stop = int(np.searchsorted(series.dates, day, side="right"))
            history[code] = series.values[:stop]

        row: dict[str, float] = {}
        lag_returns: dict[int, np.ndarray] = {}
        usd_returns = {lag: _return(history["USD"], lag) for lag in LAGS}
        for code in codes:
            values = history[code]
            for lag in LAGS:
                value = _return(values, lag)
                row[f"broad_{code.lower()}_ret_{lag}"] = (
                    value if np.isfinite(value) else 0.0
                )
                row[f"broad_{code.lower()}_ret_{lag}_missing"] = float(
                    not np.isfinite(value)
                )
            for window in VOL_WINDOWS:
                value = _volatility(values, window)
                row[f"broad_{code.lower()}_vol_{window}"] = (
                    value if np.isfinite(value) else 0.0
                )
            if len(values):
                series = reference[code]
                stop = int(np.searchsorted(series.dates, day, side="right"))
                row[f"broad_{code.lower()}_age_days"] = float(
                    (day - series.dates[stop - 1]).days
                )
            else:
                row[f"broad_{code.lower()}_age_days"] = 999.0

        own = target_series[target_code].values[: target_position + 1]
        for lag in LAGS:
            values = np.asarray([_return(history[code], lag) for code in codes])
            lag_returns[lag] = values
            _aggregate(values, f"broad_factor_ret_{lag}", row)
            usd_value = usd_returns[lag]
            cross = values - usd_value if np.isfinite(usd_value) else np.full(len(values), np.nan)
            _aggregate(cross, f"broad_exusd_ret_{lag}", row)
            own_return = _return(own, lag)
            finite = values[np.isfinite(values)]
            factor_mean = float(np.mean(finite)) if len(finite) else 0.0
            factor_median = float(np.median(finite)) if len(finite) else 0.0
            row[f"broad_target_minus_mean_{lag}"] = (
                own_return - factor_mean if np.isfinite(own_return) else 0.0
            )
            row[f"broad_target_minus_median_{lag}"] = (
                own_return - factor_median if np.isfinite(own_return) else 0.0
            )

        for window in VOL_WINDOWS:
            values = np.asarray([_volatility(history[code], window) for code in codes])
            _aggregate(values, f"broad_factor_vol_{window}", row)
        rows.append(row)

    names = sorted(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        bad = np.where(~np.isfinite(matrix))
        raise ValueError(f"non-finite broad CBR feature at {bad[0][0]}, {bad[1][0]}")
    return matrix, names


def _fingerprint() -> str:
    return hashlib.sha256(DATA.read_bytes()).hexdigest()


def load_broad_features(index, target_series, rebuild: bool = False):
    fingerprint = _fingerprint()
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=True)
        if (
            str(cached["fingerprint"].item()) == fingerprint
            and len(cached["matrix"]) == len(index)
        ):
            return cached["matrix"], list(cached["names"]), load_reference()
    reference = load_reference()
    matrix, names = build_broad_features(reference, target_series, index)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, matrix=matrix, names=np.asarray(names, dtype=object),
        fingerprint=np.asarray(fingerprint, dtype=object),
    )
    return matrix, names, reference


def causality_check(index, target_series, cut: dt.date) -> None:
    reference = load_reference()
    full, full_names = build_broad_features(reference, target_series, index)
    corrupted = {}
    for code, series in reference.items():
        values = series.values.copy()
        future = series.dates > cut
        if future.any():
            values[future] *= np.linspace(2.0, 50.0, int(future.sum()))
        corrupted[code] = Series(code, series.dates.copy(), values)
    changed, changed_names = build_broad_features(corrupted, target_series, index)
    past = np.asarray([row[2] <= cut for row in index])
    if full_names != changed_names:
        raise AssertionError("broad feature schema changed after future corruption")
    if not np.array_equal(full[past], changed[past]):
        diff = np.where(full[past] != changed[past])
        raise AssertionError(f"future CBR value changed past feature column {diff[1][0]}")
    if not np.any(full[~past] != changed[~past]):
        raise AssertionError("future corruption did not affect future feature rows")


if __name__ == "__main__":
    from research.round5_features import load_round5_features

    _X, _names, index, series, *_rest = load_round5_features()
    matrix, names, reference = load_broad_features(index, series, rebuild=True)
    causality_check(index, series, dt.date(2025, 6, 30))
    print({
        "rows": len(matrix), "features": len(names),
        "reference_currencies": sorted(reference), "causality_ok": True,
    })
