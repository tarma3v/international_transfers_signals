import numpy as np
import datetime as dt

from research.temperature_t4_premarket_models import (
    FEATURES,
    compact_features,
    fit_quarterly_hist,
)


def test_compact_premarket_features_follow_frozen_order():
    names = list(reversed(FEATURES))
    matrix = np.arange(len(names), dtype=float)[None, :]
    compact = compact_features(matrix, names)
    expected = np.array([[names.index(name) for name in FEATURES]], dtype=float)
    np.testing.assert_array_equal(compact, expected)


def test_premarket_model_predicts_unmatured_query_rows():
    dates = np.array([
        dt.date(2022, 2, 24) + dt.timedelta(days=i) for i in range(800)
    ], dtype=object)
    features = np.column_stack([
        np.sin(np.arange(len(dates)) / 10.),
        np.cos(np.arange(len(dates)) / 10.),
    ])
    target = (features[:, 0] > 0).astype(float)
    target[-30:] = np.nan
    maturity = np.array([day + dt.timedelta(days=1) for day in dates],
                        dtype=object)
    probability, _count, _logs = fit_quarterly_hist(
        features, target, maturity, dates)
    assert np.isfinite(probability[-1])
