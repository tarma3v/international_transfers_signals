import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.targets import HORIZONS, target_now_favourable, benefit_bps, benefit_forward_only
from research.after_publication_ap7_paths import (
    normalized_paths, scenario_summary, NeighborLibrary, chronological_split, SplitPathModel, forecast_paths,
)


def test_paths_match_original_targets_and_utilities_with_ties():
    values = np.r_[np.linspace(98., 100., 30), [100., 100., 101., 99.], np.linspace(100., 105., 30)]
    series = {'TJS': SimpleNamespace(values=values)}
    panel = pd.DataFrame({'currency': ['TJS'], 'announced_index': [30]})
    paths = normalized_paths(series, panel, np.array([42.]))
    past = np.array([values[30-h:30].sum() / values[30] for h in HORIZONS])
    p, f, s, clips = scenario_summary(paths, np.ones(1), 42., past)
    for j, h in enumerate(HORIZONS):
        assert p[j] == target_now_favourable(values, 30, h)
        assert f[j] == pytest.approx(benefit_forward_only(values, 30, h), abs=1e-10)
        assert s[j] == pytest.approx(benefit_bps(values, 30, h), abs=1e-10)
    assert p[0] == 1. and clips == 0


def test_benefit_is_expectation_of_scenarios_not_benefit_of_average():
    paths = np.tile(np.log([.5, 2.])[:, None] * 1e4, (1, 20))
    p, forward, _, _ = scenario_summary(paths, np.array([.5, .5]), 1., np.array(HORIZONS))
    np.testing.assert_allclose(forward, -2500.)
    assert forward[0] != pytest.approx(1e4 * (1 - 1 / 1.25))
    np.testing.assert_allclose(p, .5)


def test_neighbor_scaling_only_library_and_deterministic_ties():
    X = np.array([[1., 2.], [1., 2.], [3., 4.], [1e6, 1e6]])
    lib = NeighborLibrary(X, np.array([0, 1, 2]), 2)
    np.testing.assert_allclose(lib.mean, X[:3].mean(axis=0))
    ids, weights = lib.query(np.array([1., 2.]))
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(weights, [.5, .5])


def test_structure_outcomes_mature_before_any_estimation_day():
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(200)])
    maturity = dates + dt.timedelta(days=25)
    structure, estimation, boundary = chronological_split(np.arange(200), dates, maturity)
    assert not set(structure).intersection(estimation)
    assert (maturity[structure] < boundary - dt.timedelta(days=2)).all()
    assert (dates[estimation] >= boundary).all()
    assert len(structure) < 120 and len(estimation) == 80


def test_leaf_distribution_uses_only_estimation_paths():
    rng = np.random.default_rng(20)
    X, paths = rng.normal(size=(200, 3)), rng.normal(size=(200, 20))
    structure, estimation = np.arange(120), np.arange(140, 200)
    model = SplitPathModel(X, paths, structure, estimation, 'forest')
    scenarios, weights = model.query(X[0])
    np.testing.assert_array_equal(scenarios, paths[estimation])
    assert (weights >= 0).all() and weights.sum() == pytest.approx(1.)


def test_gaussian_fixed_antithetic_scenarios_and_train_only_mean():
    rng = np.random.default_rng(20)
    X, paths = rng.normal(size=(200, 3)), rng.normal(size=(200, 20))
    structure, estimation = np.arange(120), np.arange(140, 200)
    model = SplitPathModel(X, paths, structure, estimation, 'gaussian')
    np.testing.assert_allclose(model.mean, X[structure].mean(axis=0))
    a, w = model.query(X[0])
    model.query(X[1])
    b, _ = model.query(X[0])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose((a[:128] + a[128:]) / 2, np.tile(a.mean(axis=0), (128, 1)), atol=1e-12)
    assert w.sum() == pytest.approx(1.) and len(w) == 256


@pytest.fixture(scope='module')
def forecast_example():
    rng = np.random.default_rng(23)
    n = 275
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    panel = pd.DataFrame({'date': dates, 'currency': ['TJS'] * n})
    X = rng.normal(size=(n, 3))
    paths = np.cumsum(rng.normal(size=(n, 20)) + .1 * X[:, :1], axis=1)
    scale = np.full(n, 50.)
    past = np.tile(np.array(HORIZONS), (n, 1))
    maturity = dates + dt.timedelta(days=21)
    expected = forecast_paths(panel, X, X, paths, scale, past, maturity)
    return panel, X, paths, scale, past, maturity, expected


def test_future_paths_cannot_change_forecast_prefix(forecast_example):
    panel, X, paths, scale, past, maturity, expected = forecast_example
    changed = paths.copy()
    changed[panel.date >= dt.date(2022, 8, 1)] *= -10
    actual = forecast_paths(panel, X, X, changed, scale, past, maturity)
    prefix = panel.date.to_numpy() < dt.date(2022, 9, 1)
    for family in expected[0]:
        for metric in expected[0][family]:
            np.testing.assert_array_equal(expected[0][family][metric][prefix], actual[0][family][metric][prefix])
    for family in expected[3]:
        np.testing.assert_array_equal(expected[3][family][prefix], actual[3][family][prefix])


def test_future_inputs_cannot_change_past_queries(forecast_example):
    panel, X, paths, scale, past, maturity, expected = forecast_example
    changed = X.copy()
    changed[panel.date >= dt.date(2022, 8, 1)] *= 1e6
    actual = forecast_paths(panel, changed, changed, paths, scale, past, maturity)
    prefix = panel.date.to_numpy() < dt.date(2022, 8, 1)
    for family in expected[0]:
        for metric in expected[0][family]:
            np.testing.assert_array_equal(expected[0][family][metric][prefix], actual[0][family][metric][prefix])


def test_saved_path_controls_unchanged():
    from research.after_publication_ap7 import BASE, AP3, build_policies
    from research.after_publication_ap7_paths import FAMILIES
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    forecasts = {k: {metric: np.tile(np.linspace(.7, .2, 5), (len(panel), 1))
                    for metric in ('probability', 'forward', 'symmetric')} for k in FAMILIES}
    with np.load(BASE / 'outputs.npz') as previous, np.load(AP3 / 'outputs.npz') as ap3:
        raw, signals = build_policies(panel, previous, forecasts, np.ones(len(panel)), ap3['score__cny_last'])
        assert len(raw) == len(signals) == 70
