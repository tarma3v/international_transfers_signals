import datetime as dt

import numpy as np

from research.temperature_t11_causal_benefit_gate_models import (
    WEIGHTS,
    fit_quarterly_gate,
)


def test_causal_gate_prefers_better_mature_leg_and_ignores_future():
    dates = np.asarray([
        dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(1200)
    ], dtype=object)
    target = np.sin(np.arange(1200) / 20.) * 10.
    prior = target + 5.
    model = target + 1.
    maturity = np.asarray([day + dt.timedelta(days=5) for day in dates],
                          dtype=object)
    prediction, weight, count, _ = fit_quarterly_gate(
        model, prior, target, maturity, dates)
    scope = dates >= dt.date(2024, 1, 1)
    assert np.all(np.isin(weight[scope], WEIGHTS))
    assert np.nanmax(weight[scope]) == 1.
    assert np.nanmin(count[scope]) >= 500
    changed = target.copy()
    future = dates >= dt.date(2024, 1, 1)
    changed[future] += 1000.
    changed_maturity = maturity.copy()
    changed_maturity[future] = dt.date(2099, 1, 1)
    other = fit_quarterly_gate(
        model, prior, changed, changed_maturity, dates)[:2]
    past = dates < dt.date(2024, 1, 1)
    np.testing.assert_array_equal(prediction[past], other[0][past])
    np.testing.assert_array_equal(weight[past], other[1][past])
