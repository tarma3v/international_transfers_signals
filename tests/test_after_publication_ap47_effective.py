import datetime as dt

import numpy as np

from research.after_publication_ap40_effective_models import optimal_stopping_router


def test_rate120_guard_restores_a_model_rejected_core_below_buffer():
    dates = np.array([dt.date(2024, 1, 8)], dtype=object)
    signal, rate, _, reason, veto = optimal_stopping_router(
        core=np.array([True]), fallback=np.array([False]),
        take_probability=np.array([0.1]), fallback_quality=np.array([False]),
        dates=dates, currencies=np.array(['TJS']), eligible=np.array([True]),
        rate_floor=1.2)
    np.testing.assert_array_equal(signal, [True])
    np.testing.assert_array_equal(reason, [5])
    np.testing.assert_array_equal(veto, [False])
    assert rate[0] == 0.
