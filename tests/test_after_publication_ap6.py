import datetime as dt

import numpy as np
import pandas as pd
import pytest

from research.after_publication_ap5_learning import available_mask
from research.after_publication_ap6_learning import (
    MODEL_NAMES, fit_linear, predict_linear, ranking_pairs, local_fraction, fit_stacks,
)


def test_ap6_strict_maturity_boundary():
    now = dt.date(2023, 1, 10)
    dates = np.array([dt.date(2023, 1, 1)] * 3)
    mature = np.array([now - dt.timedelta(days=k) for k in (3, 2, 1)])
    np.testing.assert_array_equal(available_mask(dates, mature, now, np.ones(3, bool)), [True, False, False])


def test_positive_stacker_and_train_only_scaling():
    rng = np.random.default_rng(52)
    X = rng.normal(size=(350, 6))
    y = (X[:, 0] - X[:, 1] + .2 * rng.normal(size=350) > 0).astype(float)
    model = fit_linear(X, y, True)
    assert (model['coef'][1:] >= 0).all()
    np.testing.assert_allclose(model['mean'], X.mean(axis=0))
    before = predict_linear(model, X[:5])
    augmented = np.vstack((X[:5], np.full((10, 6), 1e6)))
    np.testing.assert_allclose(before, predict_linear(model, augmented)[:5])
    moved = X[:5].copy()
    moved[:, 0] += 1
    assert (predict_linear(model, moved) >= before).all()


def test_pairs_same_currency_opposite_labels_nearby_and_bounded():
    dates = np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i * 9) for i in range(50)])
    currencies = np.array(['TJS', 'KZT'] * 25)
    y = (np.arange(50) % 3 == 0).astype(float)
    pairs = ranking_pairs(dates, currencies, y)
    assert len(pairs) > 0
    for i, j in pairs:
        assert y[i] == 1 and y[j] == 0 and currencies[i] == currencies[j]
        assert abs((dates[i] - dates[j]).days) <= 60
    assert pd.Series(pairs[:, 0]).value_counts().max() <= 8
    assert ranking_pairs(dates, currencies, np.ones(50)).shape == (0, 2)


def test_pairwise_linear_uses_differences_not_intercept():
    X = np.linspace(-3, 3, 200)[:, None]
    y = (X[:, 0] > 0).astype(float)
    pairs = np.column_stack((np.arange(100, 200), np.arange(100)))
    model = fit_linear(X, y, pairs=pairs)
    assert model['coef'].shape == (1,) and model['pairwise']
    assert (np.diff(predict_linear(model, X)) > 0).all()


def test_local_shrinkage_fixed_not_selected_from_future():
    assert local_fraction(0) == 0.
    assert local_fraction(250) == .5
    assert 0. < local_fraction(60) < .5 < local_fraction(500) < 1.


@pytest.fixture(scope='module')
def synthetic_stack():
    rng = np.random.default_rng(29)
    n = 370
    dates = np.array([dt.date(2022, 7, 1) + dt.timedelta(days=i) for i in range(n)])
    panel = pd.DataFrame({'date': dates, 'currency': ['TJS'] * n})
    experts = rng.normal(size=(n, 6))
    context = rng.normal(size=(n, 3))
    equal = np.tile(np.array([.7, .6, .5, .4, .3]), (n, 1))
    latent = 1. / (1. + np.exp(-experts[:, 0]))
    y = (latent[:, None] + .1 * rng.normal(size=(n, 1)) < np.array([.75, .65, .55, .4, .25])).astype(float)
    maturity = dates + dt.timedelta(days=21)
    expected = fit_stacks(panel, experts, context, context[:, :2], equal, y, maturity)
    return panel, experts, context, equal, y, maturity, expected


def test_all_meta_models_and_residuals_future_label_corruption(synthetic_stack):
    panel, experts, context, equal, y, maturity, expected = synthetic_stack
    changed = y.copy()
    changed[panel.date >= dt.date(2023, 2, 1)] = 1 - changed[panel.date >= dt.date(2023, 2, 1)]
    actual = fit_stacks(panel, experts, context, context[:, :2], equal, changed, maturity)
    prefix = panel.date.to_numpy() < dt.date(2023, 3, 1)
    for key in MODEL_NAMES:
        np.testing.assert_array_equal(expected[0][key][prefix], actual[0][key][prefix])
    for key in ('local', 'equal'):
        np.testing.assert_array_equal(expected[4][key][prefix], actual[4][key][prefix])
    assert not np.allclose(expected[0]['local_residual'][-30:], actual[0]['local_residual'][-30:])


def test_future_features_do_not_change_past_meta_forecasts(synthetic_stack):
    panel, experts, context, equal, y, maturity, expected = synthetic_stack
    changed = experts.copy()
    changed[panel.date >= dt.date(2023, 2, 1)] *= -100
    actual = fit_stacks(panel, changed, context, context[:, :2], equal, y, maturity)
    prefix = panel.date.to_numpy() < dt.date(2023, 2, 1)
    for key in MODEL_NAMES:
        np.testing.assert_array_equal(expected[0][key][prefix], actual[0][key][prefix])


def test_meta_logs_and_local_residual_origins_are_strictly_past(synthetic_stack):
    *_, expected = synthetic_stack
    for row in expected[2]:
        origin = dt.date.fromisoformat(row['origin'])
        if row['n_train']:
            assert dt.date.fromisoformat(row['last_train_date']) < origin
            assert dt.date.fromisoformat(row['last_train_maturity']) < origin - dt.timedelta(days=2)
            if row['model'] == 'local_residual':
                assert dt.date.fromisoformat(row['last_local_origin']) < origin
    for key, values in expected[0].items():
        assert np.isfinite(values).all()
        if key != 'pairwise_context':
            assert ((values >= 0.) & (values <= 1.)).all()


def test_saved_controls_exactly_reproduce_without_changing_support():
    from research.after_publication_ap6 import BASE, build_policies
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    models = {k: np.arange(len(panel), dtype=float) for k in MODEL_NAMES}
    with np.load(BASE / 'outputs.npz') as previous:
        raw, signals = build_policies(panel, previous, models)
        assert len(raw) == len(signals) == 51
