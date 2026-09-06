import datetime as dt

import numpy as np

from research.after_publication_ap43_effective_models import (
    continuous_substitution_router,
)


def test_continuous_router_uses_joint_rank_substitute():
    dates = np.array([
        dt.date(2024, 1, 1), dt.date(2024, 5, 1), dt.date(2024, 5, 2)],
        dtype=object)
    currencies = np.array(['TJS'] * 3)
    core = np.array([True, True, False])
    quality = np.array([True, False, False])
    pace = np.array([.1, .1, .8])
    reserve = np.array([.1, .1, .9])
    signal, _, _, reason, veto = continuous_substitution_router(
        core, quality, pace, reserve, dates, currencies,
        np.ones(3, dtype=bool), core_floor=0.)
    np.testing.assert_array_equal(signal, [True, False, True])
    np.testing.assert_array_equal(reason, [1, 0, 3])
    np.testing.assert_array_equal(veto, [False, True, False])


def test_continuous_router_month_rescue_needs_reserve():
    dates = np.array([
        dt.date(2024, 1, 1), dt.date(2024, 5, 24), dt.date(2024, 5, 27)],
        dtype=object)
    currencies = np.array(['TJS'] * 3)
    signal = continuous_substitution_router(
        np.array([True, False, False]), np.array([True, False, False]),
        np.array([0., .1, .1]), np.array([0., .8, .6]),
        dates, currencies, np.ones(3, dtype=bool),
        core_floor=0., fallback_floor=0.)[0]
    np.testing.assert_array_equal(signal, [True, True, False])
