import datetime as dt

import numpy as np

from research.after_publication_ap40_effective_models import (
    build_weekly_take_target,
    optimal_stopping_router,
)


def test_weekly_take_target_uses_later_opportunities_and_latest_maturity():
    dates = np.array([
        dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 9),
        dt.date(2025, 1, 13),
    ])
    currencies = np.array(['AMD'] * len(dates))
    opportunity = np.array([True, True, True, True])
    values = np.array([.50, .75, .25, .50])
    outcomes = {f'y{h}': values.copy() for h in (3, 5, 10, 20)}
    maturity = np.array([
        dt.date(2025, 2, 1), dt.date(2025, 2, 2),
        dt.date(2025, 2, 3), dt.date(2025, 2, 9),
    ])
    target, target_maturity, utility, future_count, gain = build_weekly_take_target(
        dates, currencies, opportunity, outcomes, maturity)
    np.testing.assert_array_equal(target, [0., 1., 1., 1.])
    assert target_maturity[0] == dt.date(2025, 2, 3)
    assert target_maturity[1] == dt.date(2025, 2, 3)
    assert target_maturity[3] == dt.date(2025, 2, 9)
    np.testing.assert_array_equal(future_count, [2, 1, 0, 0])
    assert utility[0] == .5
    assert gain[0] == .25


def test_optimal_router_waits_then_takes_friday_and_rate_deficit():
    dates = np.array([
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        dt.date(2025, 1, 3),
        dt.date(2025, 1, 20),
    ])
    currencies = np.array(['AMD'] * len(dates))
    core = np.ones(len(dates), dtype=bool)
    fallback = np.zeros(len(dates), dtype=bool)
    take = np.array([.8, .2, .2, .2])
    fallback_quality = np.zeros(len(dates), dtype=bool)
    eligible = np.ones(len(dates), dtype=bool)
    signal, rate, _, reason, veto = optimal_stopping_router(
        core, fallback, take, fallback_quality, dates, currencies, eligible)
    np.testing.assert_array_equal(signal, [True, False, True, True])
    np.testing.assert_array_equal(reason, [1, 0, 4, 5])
    np.testing.assert_array_equal(veto, [False, True, False, False])
    assert rate[1] >= 1
    assert rate[-1] < 1


def test_weekly_target_is_prefix_invariant_at_week_boundary():
    dates = np.array([
        dt.date(2025, 1, 6), dt.date(2025, 1, 7),
        dt.date(2025, 1, 13), dt.date(2025, 1, 14),
    ])
    currencies = np.array(['AMD'] * len(dates))
    opportunity = np.ones(len(dates), dtype=bool)
    values = np.array([0., 1., 0., 1.])
    outcomes = {f'y{h}': values.copy() for h in (3, 5, 10, 20)}
    maturity = np.array([dt.date(2025, 2, day) for day in range(1, 5)])
    actual = build_weekly_take_target(
        dates, currencies, opportunity, outcomes, maturity)
    changed = {key: value.copy() for key, value in outcomes.items()}
    for value in changed.values():
        value[2:] = 1. - value[2:]
    corrupted = build_weekly_take_target(
        dates, currencies, opportunity, changed, maturity)
    np.testing.assert_array_equal(actual[0][:2], corrupted[0][:2])
    np.testing.assert_array_equal(actual[1][:2], corrupted[1][:2])
