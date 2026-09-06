import datetime as dt

import numpy as np

from research.after_publication_ap27_effective_models import (
    POLICIES,
    triple_paced_month_policy,
)


def test_backstop_policy_is_capped_and_prefix_invariant():
    n = 500
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i)
                      for i in range(n)], dtype=object)
    currencies = np.array(['AMD'] * n)
    primary = np.zeros(n)
    pace = np.zeros(n)
    backstop = np.linspace(0, 1, n)
    reserve = np.linspace(0, 1, n)
    eligible = np.ones(n, dtype=bool)
    original = triple_paced_month_policy(
        primary, pace, backstop, reserve, dates, currencies, eligible,
        .95, .70)
    assert (original[-1] == 3).any()
    for year, week in sorted(set(day.isocalendar()[:2] for day in dates)):
        mask = np.array([day.isocalendar()[:2] == (year, week) for day in dates])
        assert original[0][mask].sum() <= 2
    changed = [value.copy() for value in (primary, pace, backstop, reserve)]
    for j, value in enumerate(changed, start=1):
        value[350:] = j * 999.
    replay = triple_paced_month_policy(
        *changed, dates, currencies, eligible, .95, .70)
    for left, right in zip(original, replay):
        np.testing.assert_array_equal(left[:350], right[:350])


def test_reason_thresholds_and_known_down_veto():
    n = 500
    dates = np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i)
                      for i in range(n)], dtype=object)
    currency = np.array(['AMD'] * n)
    x = np.linspace(0, 1, n)
    eligible = np.ones(n, dtype=bool)
    eligible[120::4] = False
    result = triple_paced_month_policy(
        np.zeros(n), np.zeros(n), x, x, dates, currency, eligible, .95, .70)
    reason = result[-1]
    backstop = reason == 3
    assert (result[3][backstop] > .70).all()
    assert (result[4][backstop] > .70).all()
    assert (result[5][backstop] < .95).all()
    assert not result[0][~eligible].any()
    assert len(POLICIES) == 5
