import datetime as dt

import numpy as np

from research.after_publication_ap15_effective_models import targeted_policy


def _days(n, start=dt.date(2026, 1, 1)):
    return np.array([start + dt.timedelta(days=i) for i in range(n)])


def test_targeted_policy_vetoes_and_caps_iso_week():
    n = 70
    dates = _days(n)
    currencies = np.array(['AMD'] * n)
    score = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    eligible[30] = False
    signal, _, _, _, reason = targeted_policy(
        score, score, dates, currencies, eligible, 'top3125', window=10, warmup=5)
    assert not signal[30]
    assert np.array_equal(signal, reason > 0)
    for iso in set(day.isocalendar()[:2] for day in dates):
        assert signal[[day.isocalendar()[:2] == iso for day in dates]].sum() <= 2


def test_month_rescue_occurs_only_when_month_is_empty():
    dates = _days(31)
    currencies = np.array(['AMD'] * len(dates))
    primary = np.zeros(len(dates))
    reserve = np.arange(len(dates), dtype=float)
    eligible = np.ones(len(dates), dtype=bool)
    signal, _, _, _, reason = targeted_policy(
        primary, reserve, dates, currencies, eligible,
        'top3125_month24', window=8, warmup=4)
    rescued = np.flatnonzero(reason == 3)
    assert len(rescued) == 1
    assert dates[rescued[0]].day == 24
    assert signal.sum() == 1


def test_existing_month_signal_prevents_month_rescue():
    dates = _days(31)
    currencies = np.array(['AMD'] * len(dates))
    primary = np.zeros(len(dates))
    primary[20] = 1000
    reserve = np.arange(len(dates), dtype=float)
    eligible = np.ones(len(dates), dtype=bool)
    _, _, _, _, reason = targeted_policy(
        primary, reserve, dates, currencies, eligible,
        'top3125_month24', window=8, warmup=4)
    assert (reason == 1).sum() == 1
    assert (reason == 3).sum() == 0


def test_narrow_adaptive_uses_only_registered_thresholds():
    n = 450
    dates = _days(n)
    currencies = np.array(['AMD'] * n)
    score = np.sin(np.arange(n))
    eligible = np.ones(n, dtype=bool)
    _, _, _, threshold, _ = targeted_policy(
        score, score, dates, currencies, eligible,
        'adaptive100', window=20, warmup=8)
    assert (threshold[:84] == .6875).all()
    assert set(np.unique(threshold[84:])).issubset({.675, .70})


def test_future_corruption_cannot_change_any_prefix_state():
    n = 150
    dates = _days(n)
    currencies = np.array(['AMD'] * n)
    primary = np.sin(np.arange(n) / 4.)
    reserve = np.cos(np.arange(n) / 5.)
    eligible = np.ones(n, dtype=bool)
    base = targeted_policy(
        primary, reserve, dates, currencies, eligible,
        'adaptive100_month24', window=20, warmup=8)
    p2, r2 = primary.copy(), reserve.copy()
    p2[100:], r2[100:] = 1e9, -1e9
    changed = targeted_policy(
        p2, r2, dates, currencies, eligible,
        'adaptive100_month24', window=20, warmup=8)
    for left, right in zip(base, changed):
        np.testing.assert_array_equal(left[:100], right[:100])
