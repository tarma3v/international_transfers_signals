import numpy as np

from research.after_publication_ap28_effective_models import (
    HARD_WEIGHT,
    hierarchical_design,
    hierarchical_weights,
)


def test_hierarchical_design_adds_only_registered_regime_columns():
    X = np.arange(12, dtype=float).reshape(4, 3)
    rolling = np.array([np.nan, .2, .7, .9])
    reserve = np.array([np.nan, .8, .6, .95])
    hard = np.array([False, True, False, False])
    design = hierarchical_design(X, rolling, reserve, hard)
    assert design.shape == (4, 6)
    np.testing.assert_array_equal(design[:, :3], X)
    np.testing.assert_allclose(design[:, 3], rolling, equal_nan=True)
    np.testing.assert_allclose(design[:, 4], reserve, equal_nan=True)
    np.testing.assert_array_equal(design[:, 5], hard.astype(float))


def test_hierarchical_weights_are_fixed_and_training_only():
    train = np.array([False, True, True, True])
    hard = np.array([True, False, True, False])
    weights = hierarchical_weights(train, hard)
    np.testing.assert_array_equal(weights, [0., 1., HARD_WEIGHT, 1.])
