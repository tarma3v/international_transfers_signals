import datetime as dt

import numpy as np

from research.after_publication_ap42_effective_models import (
    specialist_substitution_router,
)


def test_specialist_router_vetoes_core_and_uses_fallback_for_deficit():
    dates = np.array([
        dt.date(2024, 1, 1), dt.date(2024, 5, 1), dt.date(2024, 5, 2)],
        dtype=object)
    currencies = np.array(['TJS'] * 3)
    core = np.array([True, True, False])
    fallback = np.array([False, False, True])
    quality = np.array([True, False, False])
    eligible = np.ones(3, dtype=bool)
    signal, _, _, reason, veto = specialist_substitution_router(
        core, fallback, quality, dates, currencies, eligible,
        core_floor=0., fallback_floor=1.05)
    np.testing.assert_array_equal(signal, [True, False, True])
    np.testing.assert_array_equal(reason, [1, 0, 3])
    np.testing.assert_array_equal(veto, [False, True, False])


def test_specialist_router_never_exceeds_two_per_iso_week():
    dates = np.array([dt.date(2024, 5, d) for d in (6, 7, 8)], dtype=object)
    currencies = np.array(['TJS'] * 3)
    signal = specialist_substitution_router(
        np.ones(3, dtype=bool), np.zeros(3, dtype=bool),
        np.ones(3, dtype=bool), dates, currencies,
        np.ones(3, dtype=bool))[0]
    assert signal.sum() == 2
