import datetime as dt

import numpy as np

from research.after_publication_ap16_effective_models import paced_policy


def _days(n):
    return np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)])


def test_pacing_starts_only_after_registered_84_days():
    n = 150
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.arange(n, dtype=float) % 3
    reserve = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    _, _, _, _, threshold, reason = paced_policy(
        primary, reserve, dates, currency, eligible,
        'pace365_p60', window=20, warmup=8)
    assert (threshold[:84] == .70).all()
    assert not (reason[:84] == 2).any()


def test_pacing_veto_and_weekly_cap_dominate():
    n = 180
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.arange(n, dtype=float)
    reserve = primary.copy()
    eligible = np.ones(n, dtype=bool)
    eligible[100] = False
    signal, _, _, _, _, reason = paced_policy(
        primary, reserve, dates, currency, eligible,
        'pace365_p60_month24', window=20, warmup=8)
    assert not signal[100]
    assert np.array_equal(signal, reason > 0)
    for iso in set(day.isocalendar()[:2] for day in dates):
        assert signal[[day.isocalendar()[:2] == iso for day in dates]].sum() <= 2


def test_month_reason_is_first_signal_of_month():
    n = 31
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.zeros(n)
    reserve = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    signal, _, _, _, _, reason = paced_policy(
        primary, reserve, dates, currency, eligible,
        'pace365_p60_month24', window=8, warmup=4)
    rescued = np.flatnonzero(reason == 3)
    assert len(rescued) == 1 and dates[rescued[0]].day == 24
    assert signal.sum() == 1


def test_adaptive_reproduces_registered_threshold_set():
    n = 300
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    score = np.sin(np.arange(n))
    eligible = np.ones(n, dtype=bool)
    _, _, _, rate, threshold, _ = paced_policy(
        score, score, dates, currency, eligible,
        'adaptive105_month24', window=20, warmup=8)
    assert np.isfinite(rate).all()
    assert set(np.unique(threshold)).issubset({.65, .675, .70})
    assert (threshold[:56] == .675).all()


def test_future_corruption_preserves_all_prefix_state():
    n = 220
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.sin(np.arange(n) / 6.)
    reserve = np.cos(np.arange(n) / 7.)
    eligible = np.ones(n, dtype=bool)
    base = paced_policy(
        primary, reserve, dates, currency, eligible,
        'pace365_p55_r70', window=30, warmup=10)
    p2, r2 = primary.copy(), reserve.copy()
    p2[160:], r2[160:] = 1e6, -1e6
    changed = paced_policy(
        p2, r2, dates, currency, eligible,
        'pace365_p55_r70', window=30, warmup=10)
    for left, right in zip(base, changed):
        np.testing.assert_array_equal(left[:160], right[:160])
