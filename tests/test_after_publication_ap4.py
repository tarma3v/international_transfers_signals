import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml.data import Series
from research.after_publication_ap2 import residual_mask
from research.after_publication_ap3_policy import past_percentiles, sequential_policy, utility_from_forecast
from research.after_publication_ap4 import build_policies, INCUMBENT, MODEL_NAMES
from research.after_publication_ap4_survival import (
    prediction_design, person_period, survival_from_hazards,
    fit_predict_hazard, restricted_wait_targets,
)


def test_person_period_stops_at_first_failure_not_after_each_negative_label():
    X = np.array([[10., 2.], [20., 3.], [30., 4.]])
    y = np.array([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    design, target, rows, intervals = person_period(X, y)
    np.testing.assert_array_equal(rows, [0, 0, 0, 1, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(intervals, [0, 1, 2, 0, 0, 1, 2, 3, 4])
    assert target.sum() == 2 and len(target) == 9
    np.testing.assert_array_equal(design[:, :2], X[rows])
    np.testing.assert_array_equal(design[:, 2:], np.eye(5)[intervals])


def test_incomplete_and_nonmonotone_survival_labels_are_rejected():
    for row in ([1, 1, 1, 1, np.nan], [1, 0, 1, 0, 0], [1, .5, 0, 0, 0]):
        with pytest.raises(ValueError):
            person_period(np.zeros((1, 2)), np.array([row]))


def test_cumulative_survival_is_product_not_sum_of_interval_probabilities():
    actual = survival_from_hazards(np.array([[.1, .2, .5, 0., 1.]]))
    np.testing.assert_allclose(actual, [[.9, .72, .36, .36, 0.]])
    assert (np.diff(actual, axis=1) <= 0).all()
    with pytest.raises(ValueError):
        survival_from_hazards(np.array([[1.01, .2]]))


def test_prediction_design_does_not_depend_on_future_labels_or_realized_paths():
    X = np.array([[10., 2.], [20., 3.]])
    expected = prediction_design(X)
    person_period(X, np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]))
    person_period(X, np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]))
    np.testing.assert_array_equal(expected, prediction_design(X))
    assert expected.shape == (10, 7)


def test_restricted_wait_ties_and_censoring_do_not_become_earlier_failures():
    dates = np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(40)], dtype=object)
    values = np.ones(40)
    series = {'TJS': Series('TJS', dates, values)}
    panel = pd.DataFrame({'currency': ['TJS'] * 3, 'announced_index': [4, 8, 25]})
    result = restricted_wait_targets(series, panel)
    np.testing.assert_array_equal(result, [21, 21, np.nan])
    values[7] = .9
    np.testing.assert_array_equal(restricted_wait_targets(series, panel), [3, 21, np.nan])


def test_small_sample_fallback_uses_training_risk_sets_only():
    X = np.ones((3, 2))
    y = np.array([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    pred, n_risk, failures = fit_predict_hazard(X, y, np.array([[999., -10.]]), 'logit', 60)
    np.testing.assert_allclose(pred, [[2/3, 2/3, 1/3, 1/3, 1/3]])
    assert n_risk == 9 and failures == 2


def test_local_hazard_residual_excludes_immature_and_current_quarter_rows():
    dates = np.array([dt.date(2023, 1, 3), dt.date(2023, 3, 1), dt.date(2023, 4, 3)])
    panel = pd.DataFrame({'date': dates})
    outcomes = {'mature20': np.array([dt.date(2023, 2, 3), dt.date(2023, 4, 1), dt.date(2023, 5, 5)]),
                'y20': np.ones(3)}
    origins = np.array([dt.date(2023, 1, 1), dt.date(2023, 1, 1), dt.date(2023, 4, 1)])
    mask = residual_mask(panel, outcomes, dt.date(2023, 4, 1), np.ones(3), origins)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_ap3_reproduction_and_soft_utility_prefix_invariance():
    n = 190
    dates = np.array([dt.date(2022, 7, 1) + dt.timedelta(days=i) for i in range(n)])
    currencies = np.repeat('TJS', n)
    panel = pd.DataFrame({'date': dates, 'currency': currencies})
    rng = np.random.default_rng(25)
    saved = {'score__' + k: rng.normal(size=n) for k in ('cny_last', 'market_hist', 'market_extra')}
    saved.update(scale=np.repeat(30., n), known_past_sums=np.tile([1, 3, 5, 10, 20], (n, 1)),
                 future_mean_forecast=rng.normal(size=(n, 5)))
    saved['predicted_symmetric_benefit'] = utility_from_forecast(saved['future_mean_forecast'], saved['scale'], saved['known_past_sums'])
    mix = .5 * past_percentiles(saved['score__cny_last'], currencies) + .5 * past_percentiles(saved['score__market_hist'], currencies)
    saved['signal__' + INCUMBENT] = sequential_policy(mix, dates, currencies, 'urgent_cap2')
    models = {k: rng.normal(size=n) for k in MODEL_NAMES}
    _, signals, utility = build_policies(panel, saved, models)
    assert len(signals) == 46
    np.testing.assert_array_equal(signals[INCUMBENT], saved['signal__' + INCUMBENT])
    saved['future_mean_forecast'][150:] *= -1000
    saved['predicted_symmetric_benefit'] = utility_from_forecast(saved['future_mean_forecast'], saved['scale'], saved['known_past_sums'])
    _, changed, altered_utility = build_policies(panel, saved, models)
    np.testing.assert_array_equal(utility['past_all'], altered_utility['past_all'])
    for key in signals:
        np.testing.assert_array_equal(signals[key][:150], changed[key][:150])
