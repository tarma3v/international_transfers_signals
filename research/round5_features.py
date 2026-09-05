"""Target-free causal trajectory features for round five.

The ordinary feature matrix is deliberately supplemented with representations
that are invariant to the absolute volatility scale.  Every path ends at the
row date.  Random convolution kernels are fixed from a seed and never inspect
the target.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ml.data import CORRIDORS, Series
from research.extended_features import LONG_DATA, load_or_build


SEED = 20260904
PATH_LENGTH = 64
SCALE_LENGTH = 120
N_KERNELS = 96
CACHE = Path("research/cache/round5_path_features.npz")


def _robust_scale(values: np.ndarray) -> float:
    if not len(values):
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    standard = float(np.std(values))
    scale = mad if mad > 1e-9 else standard
    return scale if scale > 1e-9 else 1.0


def _pad(values: np.ndarray, length: int = PATH_LENGTH) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    take = min(length, len(values))
    if take:
        out[-take:] = values[-take:]
    return out


def _return_maps(series: dict[str, Series]):
    maps: dict[str, dict[object, float]] = {}
    all_dates: set = set()
    for code in CORRIDORS:
        values = series[code].values
        returns = np.diff(np.log(values)) * 10000.0
        maps[code] = {day: float(value) for day, value in zip(series[code].dates[1:], returns)}
        all_dates.update(maps[code])
    dates = sorted(all_dates)
    common = {
        day: float(np.mean([maps[code][day] for code in CORRIDORS if day in maps[code]]))
        for day in dates
    }
    return maps, common


def _kernels() -> list[tuple[int, np.ndarray, int, float]]:
    rng = np.random.default_rng(SEED)
    result = []
    for _ in range(N_KERNELS):
        channel = int(rng.integers(0, 3))
        length = int(rng.choice([3, 5, 7, 9, 11]))
        allowed = [d for d in (1, 2, 4, 8, 12) if (length - 1) * d < PATH_LENGTH]
        dilation = int(rng.choice(allowed))
        weights = rng.normal(size=length)
        weights -= weights.mean()
        weights /= max(float(np.linalg.norm(weights)), 1e-12)
        bias = float(rng.uniform(-1.5, 1.5))
        result.append((channel, weights, dilation, bias))
    return result


def _summaries(path: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for lag in range(1, 17):
        values.append(float(path[-lag]))
        names.append(f"{prefix}_lag_{lag}")
    for window in (3, 5, 10, 20, 40, 64):
        part = path[-window:]
        cumulative = np.cumsum(part)
        t = np.arange(window, dtype=float)
        slope = float(np.polyfit(t, cumulative, 1)[0]) if window >= 3 else 0.0
        values.extend([
            float(part.mean()), float(part.std()), float(part.sum()),
            float(np.mean(part > 0.0)), slope,
            float(np.max(cumulative) - cumulative[-1]),
            float(cumulative[-1] - np.min(cumulative)),
        ])
        names.extend([
            f"{prefix}_mean_{window}", f"{prefix}_std_{window}",
            f"{prefix}_sum_{window}", f"{prefix}_positive_{window}",
            f"{prefix}_cum_slope_{window}", f"{prefix}_off_peak_{window}",
            f"{prefix}_off_trough_{window}",
        ])
    for window in (20, 64):
        part = path[-window:]
        if float(np.std(part[:-1])) > 1e-12 and float(np.std(part[1:])) > 1e-12:
            ac1 = float(np.corrcoef(part[:-1], part[1:])[0, 1])
        else:
            ac1 = 0.0
        spectrum = np.abs(np.fft.rfft(part - part.mean())) / max(window, 1)
        values.append(ac1)
        names.append(f"{prefix}_ac1_{window}")
        for harmonic in range(1, 5):
            value = float(spectrum[harmonic]) if harmonic < len(spectrum) else 0.0
            values.append(value)
            names.append(f"{prefix}_fft_{window}_{harmonic}")
    return values, names


def _rocket(paths: np.ndarray, kernels) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    for number, (channel, weights, dilation, bias) in enumerate(kernels):
        path = paths[channel]
        width = (len(weights) - 1) * dilation + 1
        conv = np.asarray([
            float(np.dot(path[start:start + width:dilation], weights) + bias)
            for start in range(PATH_LENGTH - width + 1)
        ])
        values.extend([float(np.max(conv)), float(np.mean(conv)), float(np.mean(conv > 0.0))])
        names.extend([
            f"rocket_{number:03d}_max", f"rocket_{number:03d}_mean",
            f"rocket_{number:03d}_ppv",
        ])
    return values, names


def build_path_features(series: dict[str, Series], index: list[tuple]):
    return_maps, common = _return_maps(series)
    kernels = _kernels()
    rows = []
    feature_names = None
    raw_paths = []
    for currency, position, day in index:
        dates = series[currency].dates[: position + 1]
        own_all = np.asarray([return_maps[currency].get(d, 0.0) for d in dates[1:]], dtype=float)
        common_all = np.asarray([common.get(d, 0.0) for d in dates[1:]], dtype=float)
        residual_all = own_all - common_all
        paths = []
        for values in (own_all, common_all, residual_all):
            scale = _robust_scale(values[-SCALE_LENGTH:])
            paths.append(np.clip(_pad(values / scale), -8.0, 8.0))
        path_matrix = np.asarray(paths)
        values: list[float] = []
        names: list[str] = []
        for channel, prefix in zip(path_matrix, ("own", "common", "residual")):
            part, part_names = _summaries(channel, prefix)
            values.extend(part)
            names.extend(part_names)
        rocket, rocket_names = _rocket(path_matrix, kernels)
        values.extend(rocket)
        names.extend(rocket_names)
        rows.append(values)
        raw_paths.append(path_matrix.reshape(-1))
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise AssertionError("trajectory feature schema changed between rows")
    matrix = np.asarray(rows, dtype=np.float32)
    paths = np.asarray(raw_paths, dtype=np.float32)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(paths)):
        raise ValueError("non-finite round-five trajectory feature")
    return matrix, feature_names or [], paths


def _fingerprint() -> str:
    payload = LONG_DATA.read_bytes() + f"{SEED}:{PATH_LENGTH}:{N_KERNELS}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_round5_features(rebuild: bool = False):
    X, names, index, series = load_or_build()
    fingerprint = _fingerprint()
    if CACHE.exists() and not rebuild:
        cached = np.load(CACHE, allow_pickle=True)
        if str(cached["fingerprint"].item()) == fingerprint:
            return (
                X, names, index, series, cached["trajectory"],
                list(cached["trajectory_names"]), cached["paths"],
            )
    trajectory, trajectory_names, paths = build_path_features(series, index)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, trajectory=trajectory,
        trajectory_names=np.asarray(trajectory_names, dtype=object), paths=paths,
        fingerprint=np.asarray(fingerprint, dtype=object),
    )
    return X, names, index, series, trajectory, trajectory_names, paths


def causality_check(cut) -> None:
    """Rebuild truncated paths and require equality for every retained row."""
    _X, _names, index, series, full, full_names, full_paths = load_round5_features()
    truncated = {}
    for code, item in series.items():
        keep = np.asarray([day <= cut for day in item.dates])
        truncated[code] = Series(code, item.dates[keep].copy(), item.values[keep].copy())
    cut_index = [row for row in index if row[2] <= cut]
    cut_matrix, cut_names, cut_paths = build_path_features(truncated, cut_index)
    lookup = {row: position for position, row in enumerate(index)}
    expected = np.asarray([lookup[row] for row in cut_index], dtype=int)
    if full_names != cut_names:
        raise AssertionError("trajectory feature names change after truncation")
    if not np.array_equal(full[expected], cut_matrix):
        raise AssertionError("future values changed a past trajectory feature")
    if not np.array_equal(full_paths[expected], cut_paths):
        raise AssertionError("future values changed a past raw path")


if __name__ == "__main__":
    *_, trajectory, names, paths = load_round5_features(rebuild=True)
    print({"rows": len(trajectory), "features": len(names), "path_columns": paths.shape[1]})
