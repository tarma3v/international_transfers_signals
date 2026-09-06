"""Fixed cross-horizon probability consensus for AP49."""
from __future__ import annotations

import numpy as np


def geometric_consensus(probabilities):
    matrix = np.column_stack([
        np.asarray(probabilities[h], dtype=float) for h in (3, 5, 10, 20)
    ])
    result = np.full(len(matrix), np.nan)
    valid = np.isfinite(matrix).all(axis=1)
    clipped = np.clip(matrix[valid], 1e-6, 1. - 1e-6)
    result[valid] = np.exp(np.log(clipped).mean(axis=1))
    return result
