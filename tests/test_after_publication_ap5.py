import datetime as dt

import numpy as np
import pandas as pd

from research.after_publication_ap5_learning import (
    available_mask, fit_map, apply_map, calibrate, delayed_weights, exponential_weights,
)


def feedback_example():
    dates = np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(12)])
    panel = pd.DataFrame({'date': dates, 'currency': ['TJS'] * len(dates)})
    maturity = dates + dt.timedelta(days=1)
    issued = np.tile(np.array([.1, .9])[None, :, None], (len(dates), 1, 5))
    y = np.ones((len(dates), 5))
    return panel, maturity, issued, y


def test_strict_maturity_and_two_day_embargo_boundary():
    p, m, issued, y = feedback_example()
    w, logs = delayed_weights(p, issued, y, m, 63, 10)
    np.testing.assert_array_equal(w[:4], np.full((4, 2), .5))
    assert w[4, 1] > .9
    assert logs[4]['last_revealed_maturity'] == '2023-01-02'
    mask = available_mask(p.date.to_numpy(), m, dt.date(2023, 1, 4), np.ones(len(p), bool))
    assert not mask.any()


def test_unrevealed_future_labels_cannot_change_earlier_weights():
    p, m, issued, y = feedback_example()
    expected, _ = delayed_weights(p, issued, y, m, 63, 30, True)
    changed = y.copy()
    changed[5:] = 0
    actual, _ = delayed_weights(p, issued, changed, m, 63, 30, True)
    np.testing.assert_array_equal(expected[:9], actual[:9])
    assert not np.allclose(expected[-1], actual[-1])


def test_same_day_currency_order_does_not_change_weights():
    p, m, issued, y = feedback_example()
    p = pd.concat([p, p.assign(currency='KZT')], ignore_index=True)
    m, issued, y = np.tile(m, 2), np.tile(issued, (2, 1, 1)), np.tile(y, (2, 1))
    y[12:] = 0
    w, _ = delayed_weights(p, issued, y, m, 63, 10, True)
    order = np.random.default_rng(42).permutation(len(p))
    other, _ = delayed_weights(p.iloc[order].reset_index(drop=True), issued[order], y[order], m[order], 63, 10, True)
    np.testing.assert_allclose(w[order], other, atol=1e-14)


def test_weights_are_finite_normalized_with_uniform_floor():
    for loss in (np.array([0., 1., .3]), np.array([1e4, 0., 1e4])):
        w = exponential_weights(loss, 30)
        assert np.isfinite(w).all() and np.isclose(w.sum(), 1.)
        assert (w >= .1 / 3).all()


def test_frozen_weights_never_use_feedback_after_freeze():
    p, m, issued, y = feedback_example()
    freeze = dt.date(2023, 1, 6)
    expected, _ = delayed_weights(p, issued, y, m, 63, 10, freeze=freeze)
    y[2:] = 0  # These feedback rows are unavailable at the frozen clock.
    actual, _ = delayed_weights(p, issued, y, m, 63, 10, freeze=freeze)
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(expected[5:], np.tile(expected[5], (7, 1)))


def test_only_issued_past_predictions_contribute_to_delayed_losses():
    p, m, issued, y = feedback_example()
    expected, _ = delayed_weights(p, issued, y, m, 252, 10)
    altered = issued.copy()
    altered[5:] = 1 - altered[5:]
    actual, _ = delayed_weights(p, altered, y, m, 252, 10)
    np.testing.assert_array_equal(expected[:9], actual[:9])


def test_local_shrinkage_uses_common_information_but_local_errors():
    p, m, issued, y = feedback_example()
    p = pd.concat([p, p.assign(currency='KZT')], ignore_index=True)
    m, issued, y = np.tile(m, 2), np.tile(issued, (2, 1, 1)), np.tile(y, (2, 1))
    y[12:] = 0
    global_w, _ = delayed_weights(p, issued, y, m, 63, 30)
    local_w, _ = delayed_weights(p, issued, y, m, 63, 30, True)
    np.testing.assert_allclose(global_w, .5)
    assert local_w[11, 1] > .5 and local_w[23, 1] < .5


def test_positive_slope_calibration_cannot_reverse_ranking():
    x = np.linspace(-2, 2, 500)
    model = fit_map(x, (x > 0).astype(float), 'logistic', .5)
    p = apply_map(model, x)
    assert model[0] == 'logistic' and (np.diff(p) >= 0).all() and p[-1] > p[0]
    inverse = apply_map(fit_map(x, (x < 0).astype(float), 'logistic', .5), x)
    assert (np.diff(inverse) >= 0).all()


def test_monthly_calibration_future_corruption_and_frozen_clock():
    rng = np.random.default_rng(21)
    n = 300
    dates = np.array([dt.date(2022, 7, 1) + dt.timedelta(days=i) for i in range(n)])
    panel = pd.DataFrame({'date': dates, 'currency': ['TJS'] * n})
    maturity = dates + dt.timedelta(days=25)
    X = rng.normal(size=(n, 1, 5))
    y = (rng.random((n, 1)) < np.array([.7, .6, .5, .4, .3])).astype(float)
    first, logs = calibrate(panel, X, y, maturity)
    changed = y.copy()
    changed[dates >= dt.date(2023, 2, 1)] = 0
    other, _ = calibrate(panel, X, changed, maturity)
    prefix = dates < dt.date(2023, 3, 1)
    for mode in first:
        np.testing.assert_array_equal(first[mode][prefix], other[mode][prefix])
        assert (np.diff(first[mode], axis=2) <= 0).all()
    np.testing.assert_array_equal(first['frozen2023'], other['frozen2023'])
    for row in logs:
        if row['last_calibration_maturity'] is not None:
            assert dt.date.fromisoformat(row['last_calibration_maturity']) < dt.date.fromisoformat(row['fit_origin']) - dt.timedelta(days=2)
