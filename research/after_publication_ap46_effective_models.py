"""AP46 joint-horizon target and deliberately season-light meta features."""
from __future__ import annotations

import numpy as np

from ml.data import CORRIDORS


EXTRA_FEATURE_NAMES = tuple('currency_' + value for value in CORRIDORS)


def currency_features(base, currencies):
    """Append only corridor identity; no annual phase or fixed regime flag."""
    base = np.asarray(base, dtype=float)
    currencies = np.asarray(currencies)
    extra = np.column_stack([
        (currencies == value).astype(float) for value in CORRIDORS
    ])
    return np.column_stack([base, extra])


def joint_survival_target(outcomes):
    """One iff the current fixing beats every future window at every unknown h."""
    matrix = np.column_stack([
        np.asarray(outcomes['y' + str(h)], dtype=float)
        for h in (3, 5, 10, 20)
    ])
    target = np.full(len(matrix), np.nan)
    valid = np.isfinite(matrix).all(axis=1)
    target[valid] = matrix[valid].min(axis=1)
    return target
