import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml.data import Series
from ml.targets import HORIZONS
from research.after_publication_ap4_survival import person_period
from research.after_publication_ap9_censoring import observed_followup, records, fit_one, design


def synthetic():
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(40)], dtype=object)
    values = np.ones(40)
    values[7] = .9
    series = {'TJS': Series('TJS', dates, values)}
    panel = pd.DataFrame({'date': dates[[2, 5, 9, 30]] - dt.timedelta(days=1),
                          'currency': ['TJS'] * 4, 'announced_index': [2, 5, 9, 30]})
    return panel, series


def test_followup_boundary_and_future_price_corruption():
    panel, series = synthetic()
    origin = dt.date(2022, 1, 14)  #CutoffJan12, lastallowedreceiptJan11=effectiveJan12.
    counts, failures = observed_followup(series, panel, origin)
    np.testing.assert_array_equal(counts, [9, 6, 2, 0])
    np.testing.assert_array_equal(failures, [5, 2, 0, 0])
    s = series['TJS']
    s.values[s.dates - dt.timedelta(days=1) >= origin - dt.timedelta(days=2)] = .0001
    changed = observed_followup(series, panel, origin)
    for a, b in zip((counts, failures), changed):
        np.testing.assert_array_equal(a, b)


def test_coarse_incomplete_interval_drops_both_early_failures_and_survivors():
    counts = np.array([2, 2, 3, 3])
    failures = np.array([2, 0, 2, 0])
    row, interval, target = records(counts, failures, HORIZONS)
    np.testing.assert_array_equal(row, [0, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(interval, [0, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(target, [0, 0, 0, 1, 0, 0])


def test_fine_censoring_contributes_observed_risk_only():
    rows, periods, target = records(np.array([3, 3, 0]), np.array([2, 0, 0]), range(1, 21))
    np.testing.assert_array_equal(rows, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(periods, [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(target, [0, 1, 0, 0, 0])


def test_full20_coarse_exactly_matches_original_person_period():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    failures = rng.integers(0, 21, size=40)
    y = ((failures[:, None] == 0) | (failures[:, None] > np.array(HORIZONS))).astype(float)
    original, target, rows, intervals = person_period(X, y)
    actual = records(np.full(40, 20), failures, HORIZONS, full_only=True)
    for left, right in zip(actual, (rows, intervals, target)):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(original, design(X, actual[1], actual[0], 5))


def test_likelihood_matches_observed_survival_and_failure_product():
    counts, failures = np.array([3, 4, 0]), np.array([2, 0, 0])
    row, period, y = records(counts, failures, range(1, 21))
    hazards = np.array([.1, .2, .3, .4] + [.5] * 16)
    observed = np.prod(np.where(y == 1, hazards[period], 1 - hazards[period]))
    expected = (.9 * .2) * (.9 * .8 * .7 * .6)
    assert np.isclose(observed, expected)
    assert len(y) == 6  #No invented future survival terms.


def test_constant_hazard_recovery_under_independent_censoring():
    rng = np.random.default_rng(9)
    n = 100000
    event = rng.geometric(.2, size=n)
    count = rng.integers(0, 21, size=n)
    failure = np.where(event <= count, event, 0)
    rows, intervals, target = records(count, failure, range(1, 21))
    estimate = target.mean()
    assert abs(estimate - .2) < .003
    #Unobserved steps treated as survival severely dilute the event probability.
    wrong = target.sum() / (n * 20)
    assert wrong < .05


def test_invalid_observed_failure_and_interval_layout_rejected():
    with pytest.raises(ValueError):
        records(np.array([2]), np.array([3]), HORIZONS)
    with pytest.raises(ValueError):
        records(np.array([20]), np.array([0]), [1, 3, 2])


def test_direct_h5_does_not_select_early_known_failures_before_h5():
    counts = np.array([3, 3, 5, 5, 20])
    failures = np.array([1, 0, 1, 0, 0])
    X = np.arange(10).reshape(5, 2)
    _, rows, _, target, _, _ = fit_one(X, counts, failures, 'direct_mature5', np.array([0]))
    np.testing.assert_array_equal(rows, [2, 3, 4])
    np.testing.assert_array_equal(target, [0, 1, 1])


def test_future_features_do_not_affect_fit_or_other_queries():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(460, 4))
    counts = np.r_[np.full(430, 20), np.zeros(30)].astype(int)
    failure = np.r_[rng.integers(0, 21, size=430), np.zeros(30)].astype(int)
    query = np.array([430, 431])
    curve = fit_one(X, counts, failure, 'fine_partial', query)[0]
    changed = X.copy()
    changed[432:] *= 10000
    np.testing.assert_array_equal(curve, fit_one(changed, counts, failure, 'fine_partial', query)[0])
