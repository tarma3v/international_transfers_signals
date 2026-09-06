import datetime as dt

import numpy as np

from research.after_publication_ap40_effective_models import optimal_stopping_router


def test_rate_floor_protects_core_opportunity_below_buffer():
    dates = np.array([dt.date(2024, 1, 1), dt.date(2024, 1, 2)], dtype=object)
    currencies = np.array(['TJS', 'TJS'])
    core = np.array([True, True])
    fallback = np.zeros(2, dtype=bool)
    probability = np.array([.9, .1])
    quality = np.zeros(2, dtype=bool)
    eligible = np.ones(2, dtype=bool)
    base = optimal_stopping_router(
        core, fallback, probability, quality, dates, currencies, eligible,
        rate_floor=0.)[0]
    protected = optimal_stopping_router(
        core, fallback, probability, quality, dates, currencies, eligible,
        rate_floor=1.1)[0]
    np.testing.assert_array_equal(base, [True, False])
    np.testing.assert_array_equal(protected, [True, True])
