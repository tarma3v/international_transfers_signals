import datetime as dt

import numpy as np
import pandas as pd

from ml.data import Series
from research.after_publication_ap11_effective import cap_opportunities, fixed_rank_cap2
from research.after_publication_ap11_effective_models import (conditional_targets, eligible_next,
    full_curve, margin_scores, margin_targets, person_period)


def test_known_next_down_is_ineligible_and_all_effective_targets_fail():
    panel = pd.DataFrame({'announced_price': [90., 110.], 'current_price': [100., 100.]})
    np.testing.assert_array_equal(eligible_next(panel), [False, True])
    outcomes = {f'y{h}': np.array([0., 1.]) for h in (3, 5, 10, 20)}
    targets = conditional_targets(outcomes)
    np.testing.assert_array_equal(targets[0], 0)


def test_person_period_starts_after_known_step_and_stops_after_failure():
    X = np.array([[10.], [20.]])
    survival = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
    design, target, rows, intervals = person_period(X, survival)
    assert list(zip(rows, intervals)) == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)]
    np.testing.assert_array_equal(target, [0, 1, 0, 0, 0, 0])
    assert design.shape[1] == X.shape[1] + 4


def test_full_curve_veto_h1_and_monotonic_projection():
    conditional = np.array([[.7, .8, .5, .6], [.9, .7, .6, .4]])
    curve = full_curve(conditional, np.array([True, False]))
    np.testing.assert_allclose(curve[0], [1, .7, .7, .5, .5])
    np.testing.assert_array_equal(curve[1], 0)


def test_margin_score_combines_known_buffer_and_unknown_minimum():
    prediction = np.array([[3., 5., -1., 2.], [2., 2., 2., 2.]])
    score = margin_scores(prediction, np.array([4., -2.]), np.array([True, False]))
    np.testing.assert_array_equal(score[0], [4., 3., 4., -1., 2.])
    assert (score[1] == -1e6).all()


def test_margin_targets_exclude_already_announced_step():
    dates = np.array([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(25)], dtype=object)
    values = np.array([100., 110., 120., 90., *([130.] * 21)])
    series = {'KZT': Series('KZT', dates, values)}
    panel = pd.DataFrame({'currency': ['KZT'], 'current_index': [0]})
    margins = margin_targets(series, panel, np.array([1.]))
    # h3 unknown remainder sees steps2=120 and3=90; the known110 is excluded.
    assert np.isclose(margins[0, 0], 1e4 * np.log(.9))


def test_cadence_filters_are_prefix_stable_and_capped():
    n = 100
    dates = np.array([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)], dtype=object)
    currencies = np.array(['KZT'] * n)
    eligible = np.ones(n, dtype=bool)
    opportunity = np.ones(n, dtype=bool)
    signal = cap_opportunities(opportunity, dates, currencies, eligible)
    prefix = cap_opportunities(opportunity[:70], dates[:70], currencies[:70], eligible[:70])
    np.testing.assert_array_equal(signal[:70], prefix)
    selected = pd.Series(signal, index=pd.to_datetime(dates))
    assert selected.groupby(selected.index.to_period('W')).sum().max() <= 2
    assert np.diff(np.flatnonzero(signal)).min() >= 2
    values = np.arange(n, dtype=float)
    rank = fixed_rank_cap2(values, dates, currencies, eligible)
    np.testing.assert_array_equal(rank[:70], fixed_rank_cap2(values[:70], dates[:70], currencies[:70], eligible[:70]))


def test_unknown_future_feature_corruption_does_not_change_rule_prefix():
    n = 120
    dates = np.array([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)], dtype=object)
    currencies = np.array(['KZT'] * n)
    eligible = np.ones(n, dtype=bool)
    values = np.sin(np.arange(n) / 7)
    baseline = fixed_rank_cap2(values, dates, currencies, eligible)
    changed = values.copy(); changed[80:] = 1e9
    np.testing.assert_array_equal(baseline[:80], fixed_rank_cap2(changed, dates, currencies, eligible)[:80])
