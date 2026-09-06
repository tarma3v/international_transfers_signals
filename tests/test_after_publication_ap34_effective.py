import datetime as dt

import numpy as np

from research.after_publication_ap34_effective_models import fit_residual_survival


def synthetic_packet():
    rng = np.random.RandomState(17)
    n_train = 140
    origin = dt.date(2025, 1, 1)
    train_dates = np.array([
        origin - dt.timedelta(days=n_train - i) for i in range(n_train)
    ])
    dates = np.concatenate([train_dates, [origin, origin + dt.timedelta(days=1)]])
    X = rng.normal(size=(n_train + 2, 6))
    X[-1] = X[-2]
    target = -60 + 8 * X[:, 0] + rng.normal(scale=12, size=n_train + 2)
    train = np.arange(n_train + 2) < n_train
    query = ~train
    currencies = np.array(['AMD'] * (n_train + 2))
    known = np.zeros(n_train + 2)
    known[-2:] = [-20, 20]
    return X, target, train, query, dates, currencies, known, origin


def test_residual_survival_is_monotone_in_announced_anchor():
    score, detail, stats = fit_residual_survival(*synthetic_packet())
    assert not stats['fallback']
    assert np.isfinite(score).all()
    assert score[1] >= score[0]
    assert set(detail) == {
        'ridge_prediction', 'local_bias', 'local_weight',
        'global_probability', 'local_probability',
    }


def test_residual_survival_ignores_query_targets():
    packet = list(synthetic_packet())
    first = fit_residual_survival(*packet)[0]
    packet[1] = packet[1].copy()
    packet[1][-2:] = [1e9, -1e9]
    second = fit_residual_survival(*packet)[0]
    np.testing.assert_allclose(first, second)
