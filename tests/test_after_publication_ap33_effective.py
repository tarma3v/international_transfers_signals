import datetime as dt

import numpy as np

from research.after_publication_ap33_effective_models import calendar_fallback_router


def test_calendar_router_uses_late_week_or_silence_after_warmup():
    dates = np.array([
        dt.date(2025, 1, 1),
        dt.date(2025, 3, 27),
        dt.date(2025, 3, 28),
        dt.date(2025, 4, 1),
        dt.date(2025, 4, 7),
    ])
    currencies = np.array(['AMD'] * len(dates))
    core = np.array([False, False, True, False, False])
    fallback = np.ones(len(dates), dtype=bool)
    signal, _, gap, reason = calendar_fallback_router(
        core, fallback, dates, currencies, np.ones(len(dates), dtype=bool))
    np.testing.assert_array_equal(signal, [False, True, True, False, True])
    np.testing.assert_array_equal(reason, [0, 2, 1, 0, 3])
    assert gap[1] == np.inf
    assert gap[-1] == 10


def test_calendar_router_is_prefix_invariant_and_caps_week():
    dates = np.array([dt.date(2025, 4, day) for day in range(1, 12)])
    currencies = np.array(['AMD'] * len(dates))
    core = np.ones(len(dates), dtype=bool)
    fallback = np.ones(len(dates), dtype=bool)
    eligible = np.ones(len(dates), dtype=bool)
    actual = calendar_fallback_router(core, fallback, dates, currencies, eligible)
    corrupted = core.copy()
    corrupted[6:] = False
    changed = calendar_fallback_router(
        corrupted, fallback, dates, currencies, eligible)
    np.testing.assert_array_equal(actual[0][:6], changed[0][:6])
    weeks = {}
    for date, fired in zip(dates, actual[0]):
        key = date.isocalendar()[:2]
        weeks[key] = weeks.get(key, 0) + int(fired)
    assert max(weeks.values()) <= 2
