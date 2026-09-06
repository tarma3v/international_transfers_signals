import numpy as np

from research.after_publication_ap18_effective_models import (
    fixed_blend,
    local_residual_ridge,
    residual_hist,
    residual_ridge,
    stack_logit,
)


def _sample(n=260):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(n, 6))
    base = np.clip(.5 + .15 * X[:, 0], .05, .95)
    y = (X[:, 0] + .5 * X[:, 1] > 0).astype(float)
    train = np.arange(n) < 220
    query = ~train
    return X, base, y, train, query


def test_fixed_blend_exact():
    np.testing.assert_allclose(fixed_blend([.2, .8], [.4, .6]), [.3, .7],
                               rtol=0, atol=1e-15)


def test_residual_models_are_bounded_and_ignore_query_labels():
    X, base, y, train, query = _sample()
    changed = y.copy()
    changed[query] = 1 - changed[query]
    for fit in (residual_hist, residual_ridge):
        score, correction = fit(X, y, base, train, query)
        other, other_correction = fit(X, changed, base, train, query)
        assert np.all((score >= 0) & (score <= 1))
        np.testing.assert_array_equal(score, other)
        np.testing.assert_array_equal(correction, other_correction)


def test_local_partial_pooling_bounded_and_records_counts():
    X, base, y, _, _ = _sample(500)
    train = np.arange(500) < 420
    query = ~train
    currencies = np.array(['AMD'] * 250 + ['KGS'] * 250)
    score, correction, counts = local_residual_ridge(
        X, y, base, train, query, currencies, shrink=200)
    assert np.all((score >= 0) & (score <= 1))
    assert len(score) == correction.size == query.sum()
    assert counts['AMD'] == 250 and counts['KGS'] == 170


def test_stack_logit_bounded_and_ignores_query_labels():
    X, base, y, train, query = _sample()
    score = stack_logit(X, y, base, train, query)
    changed = y.copy()
    changed[query] = 1 - changed[query]
    other = stack_logit(X, changed, base, train, query)
    assert np.all((score >= 0) & (score <= 1))
    np.testing.assert_array_equal(score, other)
