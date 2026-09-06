import numpy as np

from research.after_publication_ap26_effective_models import SCORES, cold_start_scores


def test_cold_start_scores_have_fixed_complete_geometry():
    n = 500
    cat = np.linspace(0, 1, n)
    y20 = np.linspace(1, 0, n)
    mean = np.sin(np.arange(n) / 30.) / 2 + .5
    future = np.cos(np.arange(n) / 20.)
    counts = np.arange(n)
    currencies = np.array(['AMD'] * n)
    scores, weights, cat_rank, future_rank = cold_start_scores(
        cat, y20, mean, future, counts, counts, counts, currencies)
    assert tuple(scores) == SCORES
    assert tuple(weights) == SCORES
    assert np.isnan(cat_rank[:40]).all() and np.isnan(future_rank[:40]).all()
    assert (weights['y20_hard100'][:100] == 0).all()
    assert (weights['y20_hard100'][100:] == 1).all()
    np.testing.assert_array_equal(scores['y20_hard100'][:100], cat[:100])
    np.testing.assert_array_equal(scores['y20_hard100'][100:], y20[100:])


def test_shrinkage_weights_are_causal_count_functions():
    counts = np.array([-1, 0, 50, 100, 200, 1000])
    n = len(counts)
    values = np.linspace(.1, .9, n)
    scores, weights, _, _ = cold_start_scores(
        1 - values, values, values, values, counts, counts, counts,
        np.array(['AMD'] * n))
    np.testing.assert_allclose(
        weights['y20_shrink100'], [0, 0, 1/3, 1/2, 2/3, 10/11])
    np.testing.assert_allclose(
        weights['y20_shrink200'], [0, 0, .2, 1/3, .5, 5/6])
    assert np.isfinite(scores['y20_shrink100']).all()
