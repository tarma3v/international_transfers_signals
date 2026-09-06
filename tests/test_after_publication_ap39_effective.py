import datetime as dt

import numpy as np

from research.after_publication_ap38_effective_models import guarded_precision_router


def test_one_week_runway_preserves_then_vetoes_weak_core():
    dates = np.array([
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        dt.date(2025, 1, 8),
    ])
    currencies = np.array(['AMD'] * len(dates))
    core = np.ones(len(dates), dtype=bool)
    fallback = np.zeros(len(dates), dtype=bool)
    core_quality = np.array([True, False, False])
    fallback_quality = np.zeros(len(dates), dtype=bool)
    eligible = np.ones(len(dates), dtype=bool)
    signal, rate, _, reason, veto = guarded_precision_router(
        core, fallback, core_quality, fallback_quality,
        dates, currencies, eligible, reserve_weeks=1.)
    np.testing.assert_array_equal(signal, [True, True, False])
    np.testing.assert_array_equal(reason, [1, 4, 0])
    np.testing.assert_array_equal(veto, [False, False, True])
    assert rate[1] >= 1
    assert rate[2] >= 1
