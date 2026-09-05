"""Fixed random-convolution features over 20 completed CNY sessions."""
from __future__ import annotations

import datetime as dt

import numpy as np

from research.round6_cny_waveform_features import WINDOW, build_waveform_features


SEED = 20260905
KERNELS_PER_LENGTH = 16
LENGTHS = (3, 5, 7, 9)


def kernels():
    rng = np.random.default_rng(SEED)
    result = []
    for length in LENGTHS:
        allowed = tuple(
            dilation for dilation in (1, 2, 3)
            if (length - 1) * dilation + 1 <= WINDOW
        )
        for number in range(KERNELS_PER_LENGTH):
            weights = rng.normal(size=length)
            weights -= weights.mean()
            weights /= max(float(np.linalg.norm(weights)), 1e-12)
            dilation = allowed[number % len(allowed)]
            bias = float(rng.uniform(-1.0, 1.0))
            result.append((weights, dilation, bias))
    return result


def build_rocket_features(index, history):
    waveform, _wave_names = build_waveform_features(index, history)
    paths = waveform[:, :WINDOW].astype(float)
    center = np.mean(paths, axis=1, keepdims=True)
    scale = np.std(paths, axis=1, keepdims=True)
    normalized = np.divide(
        paths - center, scale,
        out=np.zeros_like(paths), where=scale > 1e-12,
    )
    result = np.zeros((len(index), len(kernels()) * 2), dtype=np.float32)
    names = []
    for number, (weights, dilation, bias) in enumerate(kernels()):
        width = (len(weights) - 1) * dilation + 1
        responses = np.column_stack([
            normalized[:, start:start + width:dilation] @ weights + bias
            for start in range(WINDOW - width + 1)
        ])
        result[:, 2 * number] = np.max(responses, axis=1)
        result[:, 2 * number + 1] = np.mean(responses > 0.0, axis=1)
        names.extend([
            f"cny_rocket_{number:03d}_max",
            f"cny_rocket_{number:03d}_ppv",
        ])
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite CNY random-convolution feature")
    return result, names


def causality_check(index, history, cutoff=dt.date(2025, 6, 30)):
    full, names = build_rocket_features(index, history)
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
    corrupted, changed_names = build_rocket_features(index, changed)
    past = np.asarray([row[2] <= cutoff for row in index])
    if names != changed_names or not np.array_equal(full[past], corrupted[past]):
        raise AssertionError("future CNY session changed a past convolution feature")
    return True
