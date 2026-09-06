import datetime as dt

import numpy as np

from research.after_publication_ap37_effective_models import (
    EXPERTS,
    mature_support_precision,
    precision_calendar_router,
)


def test_mature_support_precision_ignores_immature_and_future_labels():
    dates = np.array([dt.date(2024, 1, day) for day in range(1, 8)])
    currencies = np.array(['AMD'] * len(dates))
    ranks = {name: np.array([.8, .8, .8, .8, .8, .8, .8])
             for name in EXPERTS}
    outcomes = {f'y{h}': np.array([1., 0., 1., 0., 1., 0., 1.])
                for h in (3, 5, 10, 20)}
    mature = np.array([dt.date(2024, 1, 2), dt.date(2024, 1, 9),
                       dt.date(2024, 1, 4), dt.date(2024, 1, 10),
                       dt.date(2024, 1, 12), dt.date(2024, 1, 13),
                       dt.date(2024, 1, 14)])
    eligible = np.ones(len(dates), dtype=bool)
    core = np.zeros(len(dates), dtype=bool)
    actual = mature_support_precision(
        ranks, outcomes, mature, dates, currencies, eligible, core)
    changed = {key: value.copy() for key, value in outcomes.items()}
    for value in changed.values():
        value[1:] = 1. - value[1:]
    corrupted = mature_support_precision(
        ranks, changed, mature, dates, currencies, eligible, core)
    for left, right in zip(actual, corrupted):
        np.testing.assert_allclose(left[:5], right[:5], equal_nan=True)
    assert actual[0][-1] == 3
    assert actual[5][-1] == 2


def test_precision_router_gates_late_week_but_preserves_silence_and_cap():
    dates = np.array([
        dt.date(2025, 1, 1),
        dt.date(2025, 3, 20),
        dt.date(2025, 3, 27),
        dt.date(2025, 3, 28),
        dt.date(2025, 4, 7),
        dt.date(2025, 4, 8),
        dt.date(2025, 4, 9),
    ])
    currencies = np.array(['AMD'] * len(dates))
    core = np.array([False, True, False, False, False, True, True])
    fallback = np.ones(len(dates), dtype=bool)
    quality = np.array([False, True, False, True, False, True, True])
    eligible = np.ones(len(dates), dtype=bool)
    signal, _, gap, reason = precision_calendar_router(
        core, fallback, quality, dates, currencies, eligible)
    np.testing.assert_array_equal(
        signal, [False, True, False, True, True, True, False])
    np.testing.assert_array_equal(reason, [0, 1, 0, 2, 3, 1, 0])
    assert gap[4] == 10
    corrupted = quality.copy()
    corrupted[5:] = ~corrupted[5:]
    changed = precision_calendar_router(
        core, fallback, corrupted, dates, currencies, eligible)
    np.testing.assert_array_equal(signal[:5], changed[0][:5])
