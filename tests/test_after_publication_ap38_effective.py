import datetime as dt

import numpy as np

from research.after_publication_ap38_effective_models import guarded_precision_router


def test_guarded_router_vetoes_only_with_rate_buffer():
    dates = np.array([
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        dt.date(2025, 1, 10),
        dt.date(2025, 1, 11),
    ])
    currencies = np.array(['AMD'] * len(dates))
    core = np.ones(len(dates), dtype=bool)
    fallback = np.zeros(len(dates), dtype=bool)
    core_quality = np.array([True, False, False, True])
    fallback_quality = np.zeros(len(dates), dtype=bool)
    eligible = np.ones(len(dates), dtype=bool)
    signal, rate, _, reason, veto = guarded_precision_router(
        core, fallback, core_quality, fallback_quality,
        dates, currencies, eligible)
    np.testing.assert_array_equal(signal, [True, False, True, True])
    np.testing.assert_array_equal(reason, [1, 0, 4, 1])
    np.testing.assert_array_equal(veto, [False, True, False, False])
    assert rate[1] >= 1
    assert rate[2] < 1


def test_guarded_router_is_prefix_invariant_and_caps_week():
    dates = np.array([dt.date(2025, 4, day) for day in range(1, 12)])
    currencies = np.array(['AMD'] * len(dates))
    core = np.ones(len(dates), dtype=bool)
    fallback = np.ones(len(dates), dtype=bool)
    quality = np.ones(len(dates), dtype=bool)
    eligible = np.ones(len(dates), dtype=bool)
    actual = guarded_precision_router(
        core, fallback, quality, quality, dates, currencies, eligible)
    corrupted = core.copy()
    corrupted[6:] = False
    changed = guarded_precision_router(
        corrupted, fallback, quality, quality, dates, currencies, eligible)
    np.testing.assert_array_equal(actual[0][:6], changed[0][:6])
    weeks = {}
    for date, fired in zip(dates, actual[0]):
        key = date.isocalendar()[:2]
        weeks[key] = weeks.get(key, 0) + int(fired)
    assert max(weeks.values()) <= 2
