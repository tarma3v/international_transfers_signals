import numpy as np

from research.after_publication_ap19_effective_models import (
    SCORE_NAMES,
    cat_mean_utility,
    cat_multi,
    cat_pairrank,
    extra_multi,
    score_family,
)


def fixture(seed=91):
    rng = np.random.default_rng(seed)
    n = 180
    X = rng.normal(size=(n, 8))
    latent = X[:, 0] - .4 * X[:, 1] + rng.normal(scale=.8, size=n)
    cut = np.array([-.8, -.2, .25, .7])
    labels = np.column_stack([latent > value for value in cut]).astype(float)
    train = np.arange(n) < 130
    query = ~train
    currencies = np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS'] * 36)
    return X, labels, train, query, currencies


def test_multi_models_are_bounded_and_ignore_query_labels():
    X, labels, train, query, currencies = fixture()
    extra = extra_multi(X, labels, train, query)
    cat = cat_multi(X, labels, train, query)
    utility = cat_mean_utility(X, labels, train, query)
    rank = cat_pairrank(X, labels, train, query, currencies)
    changed = labels.copy()
    changed[query] = 1 - changed[query]
    np.testing.assert_allclose(extra, extra_multi(X, changed, train, query),
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(cat, cat_multi(X, changed, train, query), rtol=0, atol=0)
    np.testing.assert_allclose(utility, cat_mean_utility(X, changed, train, query), rtol=0, atol=0)
    np.testing.assert_allclose(rank, cat_pairrank(X, changed, train, query, currencies), rtol=0, atol=0)
    for value in (extra, cat, utility, rank):
        assert np.isfinite(value).all()
        assert ((value >= 0) & (value <= 1)).all()


def test_score_family_exact_fixed_blends_and_names():
    rng = np.random.default_rng(4)
    extra = rng.uniform(.1, .9, size=(12, 4))
    cat = rng.uniform(.1, .9, size=(12, 4))
    utility = rng.uniform(size=12)
    rank = rng.uniform(size=12)
    base = rng.uniform(size=12)
    base[2] = np.nan
    scores = score_family(extra, cat, utility, rank, base)
    assert tuple(scores) == SCORE_NAMES
    np.testing.assert_allclose(scores['extra_multi_mean'], extra.mean(axis=1))
    np.testing.assert_allclose(scores['cat_h5'], cat[:, 1])
    np.testing.assert_allclose(scores['cat_multi_mean'], cat.mean(axis=1))
    np.testing.assert_allclose(scores['base75_cat25'][np.isfinite(base)],
                               .75 * base[np.isfinite(base)] + .25 * cat[np.isfinite(base), 1])
    assert scores['base75_cat25'][2] == cat[2, 1]
    assert scores['base50_cat50'][2] == cat[2, 1]
    assert scores['base75_catmulti25'][2] == cat[2].mean()


def test_single_class_fallbacks_are_finite():
    X, labels, train, query, currencies = fixture()
    labels[:] = 1
    for value in (
        extra_multi(X, labels, train, query),
        cat_multi(X, labels, train, query),
        cat_mean_utility(X, labels, train, query),
        cat_pairrank(X, labels, train, query, currencies),
    ):
        assert np.isfinite(value).all()
        assert ((value >= 0) & (value <= 1)).all()
