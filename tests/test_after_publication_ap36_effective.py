import datetime as dt

import numpy as np

from research.after_publication_ap36_effective_models import (
    EXPERTS,
    mature_brier_hedge,
)


def test_mature_hedge_weights_normalize_and_ignore_future():
    rng = np.random.RandomState(31)
    n = 120
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i)
                      for i in range(n)])
    currencies = np.array(['AMD'] * n)
    experts = {name: rng.normal(size=n) for name in EXPERTS}
    y20 = rng.randint(0, 2, size=n).astype(float)
    mature20 = np.array([day + dt.timedelta(days=25) for day in dates])
    eligible = np.ones(n, dtype=bool)
    actual = mature_brier_hedge(
        experts, y20, mature20, dates, currencies, eligible)
    finite = np.isfinite(actual[1]).all(axis=1)
    np.testing.assert_allclose(actual[2][finite].sum(axis=1), 1.)

    cut = 80
    bad_experts = {name: value.copy() for name, value in experts.items()}
    for value in bad_experts.values():
        value[cut:] = value[cut:] * -9 + 50
    bad_y = y20.copy()
    bad_y[cut:] = 1 - bad_y[cut:]
    changed = mature_brier_hedge(
        bad_experts, bad_y, mature20, dates, currencies, eligible)
    for left, right in zip(actual, changed):
        np.testing.assert_allclose(left[:cut], right[:cut], equal_nan=True)
