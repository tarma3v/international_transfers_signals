import numpy as np

from research.after_publication_ap46_effective_models import (
    currency_features,
    joint_survival_target,
)


def test_joint_target_requires_success_at_every_unknown_horizon():
    outcomes = {
        'y3': np.array([1., 1., np.nan, 0.]),
        'y5': np.array([1., 0., 1., 0.]),
        'y10': np.array([1., 1., 1., 0.]),
        'y20': np.array([1., 1., 1., 1.]),
    }
    result = joint_survival_target(outcomes)
    np.testing.assert_allclose(result, [1., 0., np.nan, 0.], equal_nan=True)


def test_currency_features_add_only_five_one_hot_columns():
    result = currency_features(
        np.zeros((2, 3)), np.array(['TJS', 'KZT']))
    assert result.shape == (2, 8)
    np.testing.assert_array_equal(result[:, 3:].sum(axis=1), [1., 1.])
    assert result[0, 3] == 1.
    assert result[1, 7] == 1.
