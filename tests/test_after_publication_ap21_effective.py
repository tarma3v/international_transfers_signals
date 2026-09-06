import datetime as dt

import numpy as np

from research.after_publication_ap21_effective_models import (
    PAIRINGS,
    dual_paced_month_policy,
    pairing_scores,
)


def test_pairing_scores_are_fixed_and_complete():
    x = np.arange(100, dtype=float)
    scores = pairing_scores(x, x + 1, x + 2, x + 3, x + 4)
    assert tuple(scores) == PAIRINGS
    np.testing.assert_array_equal(scores['roll_ap18'][0], x)
    np.testing.assert_array_equal(scores['roll_cat'][1], x + 4)
    np.testing.assert_allclose(scores['roll75local25_ap1850cat50'][0], x + .25)
    np.testing.assert_allclose(scores['roll75local25_ap1850cat50'][1], x + 3.5)


def test_dual_policy_is_prefix_invariant_and_capped():
    n = 420
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    currencies = np.array(['AMD'] * n)
    primary = np.sin(np.arange(n) / 7.)
    pace = np.cos(np.arange(n) / 11.)
    reserve = np.sin(np.arange(n) / 5.)
    eligible = np.ones(n, dtype=bool)
    original = dual_paced_month_policy(
        primary, pace, reserve, dates, currencies, eligible)
    p2, f2, r2 = primary.copy(), pace.copy(), reserve.copy()
    p2[300:], f2[300:], r2[300:] = 999, -999, 777
    changed = dual_paced_month_policy(p2, f2, r2, dates, currencies, eligible)
    for left, right in zip(original, changed):
        np.testing.assert_array_equal(left[:300], right[:300])
    signal = original[0]
    for year, week in sorted(set(day.isocalendar()[:2] for day in dates)):
        mask = np.array([day.isocalendar()[:2] == (year, week) for day in dates])
        assert signal[mask].sum() <= 2


def test_known_down_veto_dominates_all_reasons():
    n = 500
    dates = np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    currency = np.array(['AMD'] * n)
    value = np.linspace(0, 1, n)
    eligible = np.ones(n, dtype=bool)
    eligible[100::3] = False
    result = dual_paced_month_policy(value, value, value, dates, currency, eligible)
    assert not result[0][~eligible].any()
