import datetime as dt

import numpy as np

from research.after_publication_ap48_effective_models import (
    same_week_substitution_router,
)


def test_veto_is_replaced_by_same_week_fallback_without_lookahead():
    dates = np.array([
        dt.date(2024, 1, 8), dt.date(2024, 1, 9),
        dt.date(2024, 1, 15),
    ], dtype=object)
    signal, reason, pending, used = same_week_substitution_router(
        primary=np.array([False, False, False]),
        core_veto=np.array([True, False, False]),
        fallback=np.array([False, True, True]), dates=dates,
        currencies=np.array(['TJS', 'TJS', 'TJS']),
        eligible=np.array([True, True, True]))
    np.testing.assert_array_equal(signal, [False, True, False])
    np.testing.assert_array_equal(reason, [0, 2, 0])
    np.testing.assert_array_equal(pending, [False, True, False])
    np.testing.assert_array_equal(used, [0, 0, 0])


def test_same_row_substitution_and_sequential_cap_are_explicit():
    dates = np.array([dt.date(2024, 1, 8)] * 3, dtype=object)
    signal, reason, _, used = same_week_substitution_router(
        primary=np.array([False, True, True]),
        core_veto=np.array([True, False, False]),
        fallback=np.array([True, False, False]), dates=dates,
        currencies=np.array(['TJS'] * 3), eligible=np.array([True] * 3))
    np.testing.assert_array_equal(signal, [True, True, False])
    np.testing.assert_array_equal(reason, [2, 1, 0])
    np.testing.assert_array_equal(used, [0, 1, 2])
