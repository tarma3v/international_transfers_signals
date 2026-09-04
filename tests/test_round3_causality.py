"""Focused causality checks for the third research round."""
from __future__ import annotations

import datetime as dt

import numpy as np

from ml.data import Series
from research.round3_delayed_labels import delayed_features
from research.round3_online_mixture import HedgeSpec, _online_sequence


def test_delayed_features_ignore_values_after_the_row_date():
    days = np.asarray([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(40)],
                      dtype=object)
    values = 10.0 + np.sin(np.arange(40) / 3.0)
    index = [("AAA", i, day) for i, day in enumerate(days)]
    original = {"AAA": Series("AAA", days, values)}
    changed_values = values.copy()
    changed_values[21:] *= 10.0
    changed = {"AAA": Series("AAA", days, changed_values)}

    left, names = delayed_features(original, index)
    right, changed_names = delayed_features(changed, index)

    assert names == changed_names
    np.testing.assert_allclose(left[:21], right[:21], rtol=0, atol=0)


def test_online_hedge_uses_a_label_only_after_its_reach_date():
    rows = np.arange(5)
    dates = np.asarray([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(5)],
                       dtype=object)
    scores = np.asarray([
        [.9, .1], [.8, .2], [.7, .3], [.6, .4], [.5, .5],
    ])
    roles = np.asarray(["test"] * 5, dtype=object)
    currencies = np.asarray(["TJS"] * 5, dtype=object)
    reach = np.asarray([
        dates[3], dates[4], dates[4] + dt.timedelta(days=1),
        dates[4] + dt.timedelta(days=2), dates[4] + dt.timedelta(days=3),
    ], dtype=object)
    first_y = np.asarray([0., 0., 0., 0., 0.])
    changed_y = first_y.copy()
    changed_y[0] = 1.0
    spec = HedgeSpec("global", eta=10.0, rho=1.0)

    first, _ = _online_sequence(
        spec, rows, scores, roles, dates, currencies, first_y, reach,
    )
    changed, _ = _online_sequence(
        spec, rows, scores, roles, dates, currencies, changed_y, reach,
    )

    # The label of row 0 reaches on dates[3], so predictions before that date
    # must remain exactly the same and predictions from that date may change.
    np.testing.assert_allclose(first[:3], changed[:3], rtol=0, atol=0)
    assert first[3] != changed[3]

