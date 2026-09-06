import datetime as dt

import numpy as np

from research.after_publication_ap44_effective_models import augment_features


def test_meta_features_include_currency_regime_and_cycles():
    dates = np.array([dt.date(2021, 1, 1), dt.date(2023, 7, 1)], dtype=object)
    currencies = np.array(['TJS', 'KZT'])
    result = augment_features(np.zeros((2, 3)), dates, currencies)
    assert result.shape == (2, 11)
    np.testing.assert_array_equal(result[:, -1], [0., 1.])
    assert result[0, 3] == 1.
    assert result[1, 7] == 1.
