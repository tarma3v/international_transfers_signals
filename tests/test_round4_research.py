import datetime as dt

import numpy as np

from ml.targets import build_targets
from ml.validation import target_reach_dates
from research.extended_features import load_or_build
from research.round4_research import (
    _current_columns,
    _masks,
    _publication_matrix,
)


def test_publication_matrix_uses_exactly_the_next_same_currency_row():
    X, names, index, series = load_or_build()
    matrix, feature_names, eligibility, next_rows = _publication_matrix(
        X, names, index, series, _current_columns(names)
    )
    assert matrix.shape[1] == len(feature_names)
    for row in np.where(next_rows >= 0)[0][::137]:
        currency, i, _day = index[row]
        next_currency, next_i, _next_day = index[next_rows[row]]
        assert next_currency == currency
        assert next_i == i + 1
        assert eligibility[row] == (series[currency].values[i + 1] >= series[currency].values[i])


def test_publication_gate_is_exactly_the_known_h1_event():
    X, names, index, series = load_or_build()
    _matrix, _feature_names, eligibility, _next_rows = _publication_matrix(
        X, names, index, series, _current_columns(names)
    )
    y1 = build_targets(series, index)["fav_h1"]
    valid = ~np.isnan(y1)
    assert np.array_equal(eligibility[valid], y1[valid].astype(bool))


def test_round4_training_is_purged_before_calibration_year():
    _X, _names, index, series = load_or_build()
    dates = np.asarray([day for _currency, _i, day in index], dtype=object)
    y = build_targets(series, index)["fav_h5"]
    reach = target_reach_dates(index, series, 5)
    for year in (2017, 2022, 2024, 2026):
        train, calibration, test = _masks(year, dates, reach, y)
        boundary = dt.date(year - 1, 1, 1)
        assert all(reach[row] < boundary for row in train)
        assert all(boundary <= dates[row] < dt.date(year, 1, 1) for row in calibration)
        assert all(dates[row].year == year for row in test)
