import numpy as np

from research.temperature_t46_local_pair_fallback import (
    FEATURE_NAMES,
    MIN_TRAIN_ROWS,
    clipped_logit,
)


def test_t46_logit_transform_is_finite_and_roundtrips():
    probability = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    value = clipped_logit(probability)
    rebuilt = 1 / (1 + np.exp(-value))
    assert np.isfinite(value).all()
    np.testing.assert_allclose(rebuilt[1:-1], probability[1:-1])


def test_t46_is_compact_and_requires_dense_local_history():
    assert len(FEATURE_NAMES) == 12
    assert MIN_TRAIN_ROWS == 150
