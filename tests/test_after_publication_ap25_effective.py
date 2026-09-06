import numpy as np

from research.after_publication_ap25_effective import hard_pool
from research.after_publication_ap25_effective_models import SPECS, specialist_target


def test_specialist_specs_are_fixed_and_unknown_horizon_only():
    assert len(SPECS) == 5
    assert all('h1' not in name for name in SPECS)
    labels = np.array([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    future = np.array([-1000., 1000.])
    train = np.array([True, True])
    mean, clip = specialist_target(labels, future, 'mean', train)
    np.testing.assert_array_equal(mean, [.5, 1.])
    assert clip is None
    y20, _ = specialist_target(labels, future, 'y20', train)
    np.testing.assert_array_equal(y20, [0., 1.])


def test_future_target_clipping_uses_training_rows_only():
    labels = np.ones((6, 4))
    future = np.array([-100., -50., 0., 50., 100., 999999.])
    train = np.array([True, True, True, True, True, False])
    target, bounds = specialist_target(labels, future, 'future5', train)
    assert bounds[1] < 100
    assert target[-1] == bounds[1] / 200.


def test_hard_pool_requires_nonprimary_high_reserve_and_eligibility():
    n = 500
    currencies = np.array(['AMD'] * n)
    rolling = np.linspace(0, 1, n)
    reserve = np.linspace(1, 0, n)
    eligible = np.ones(n, dtype=bool)
    eligible[200:250] = False
    pool, rr, zr = hard_pool(rolling, reserve, currencies, eligible)
    assert not pool[~eligible].any()
    assert (rr[pool] <= .70).all()
    assert (zr[pool] > .70).all()
    assert not pool[:40].any()  # rank warmup
