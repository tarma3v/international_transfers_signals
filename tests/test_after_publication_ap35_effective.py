import datetime as dt

import numpy as np

from research.after_publication_ap35_effective_models import fit_distributional_cat


def synthetic_packet():
    rng = np.random.RandomState(23)
    n_train = 130
    origin = dt.date(2025, 1, 1)
    dates = np.array([
        origin - dt.timedelta(days=n_train - i) for i in range(n_train)
    ] + [origin, origin + dt.timedelta(days=1)])
    X = rng.normal(size=(n_train + 2, 5))
    X[-1] = X[-2]
    known = rng.uniform(0, 180, size=n_train + 2)
    known[-2:] = [-50, 50]
    target = -100 + 20 * X[:, 0] + rng.normal(scale=40, size=n_train + 2)
    train = np.arange(n_train + 2) < n_train
    query = ~train
    return X, target, known, train, query, dates, origin


def test_distributional_cat_is_monotone_in_explicit_anchor():
    score, stats = fit_distributional_cat(*synthetic_packet())
    assert not stats['fallback']
    assert stats['monotonic_min_delta'] >= -1e-12
    assert score[1] >= score[0]


def test_distributional_cat_ignores_query_targets():
    packet = list(synthetic_packet())
    first = fit_distributional_cat(*packet)[0]
    packet[1] = packet[1].copy()
    packet[1][-2:] = [1e9, -1e9]
    second = fit_distributional_cat(*packet)[0]
    np.testing.assert_allclose(first, second)
