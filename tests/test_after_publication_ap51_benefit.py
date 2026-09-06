import numpy as np

from research.after_publication_ap51_benefit import magnitude_metrics


def test_magnitude_metrics_compare_model_with_train_only_prior():
    result = magnitude_metrics(
        np.array([1., 2., 3.]), np.array([1., 2., 3.]),
        np.array([2., 2., 2.]))
    assert result['mae_model'] == 0.
    assert result['mae_prior'] > 0.
    assert result['positive_sign_accuracy'] == 1.
