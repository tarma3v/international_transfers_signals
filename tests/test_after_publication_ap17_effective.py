import datetime as dt

import numpy as np

from research.after_publication_ap17_effective_models import paced_month_policy


def _days(n):
    return np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)])


def test_single_policy_veto_cap_and_reason_state():
    n = 180
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    score = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    eligible[100] = False
    signal, _, _, _, reason = paced_month_policy(
        score, score, dates, currency, eligible, window=20, warmup=8)
    assert not signal[100]
    assert np.array_equal(signal, reason > 0)
    for iso in set(day.isocalendar()[:2] for day in dates):
        assert signal[[day.isocalendar()[:2] == iso for day in dates]].sum() <= 2


def test_month_rescue_is_first_signal_in_month():
    dates = _days(31)
    currency = np.array(['AMD'] * len(dates))
    primary = np.zeros(len(dates))
    reserve = np.arange(len(dates), dtype=float)
    eligible = np.ones(len(dates), dtype=bool)
    signal, _, _, _, reason = paced_month_policy(
        primary, reserve, dates, currency, eligible, window=8, warmup=4)
    rescue = np.flatnonzero(reason == 3)
    assert len(rescue) == 1 and dates[rescue[0]].day == 24
    assert signal.sum() == 1


def test_future_corruption_cannot_change_full_prefix_state():
    n = 220
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.sin(np.arange(n) / 6.)
    reserve = np.cos(np.arange(n) / 7.)
    eligible = np.ones(n, dtype=bool)
    base = paced_month_policy(
        primary, reserve, dates, currency, eligible, window=30, warmup=10)
    p2, r2 = primary.copy(), reserve.copy()
    p2[160:], r2[160:] = 1e6, -1e6
    changed = paced_month_policy(
        p2, r2, dates, currency, eligible, window=30, warmup=10)
    for left, right in zip(base, changed):
        np.testing.assert_array_equal(left[:160], right[:160])
