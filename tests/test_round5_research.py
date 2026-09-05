import datetime as dt

import numpy as np

from ml.data import CORRIDORS, Series
from research.round5_features import _kernels, _robust_scale, build_path_features
from research.round5_refit_calibration import _rank


def _synthetic_series(length=90):
    dates = np.asarray([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(length)])
    result = {}
    for number, code in enumerate(CORRIDORS):
        returns = .001 * np.sin(np.arange(length - 1) / (3.0 + number)) + .0002 * number
        values = (1.0 + number) * np.exp(np.r_[0.0, np.cumsum(returns)])
        result[code] = Series(code, dates.copy(), values)
    return result


def _index(series):
    return [
        (code, position, series[code].dates[position])
        for code in CORRIDORS for position in range(20, len(series[code].dates))
    ]


def test_round5_random_kernels_are_target_free_and_deterministic():
    first = _kernels()
    second = _kernels()
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]


def test_round5_robust_normalization_is_scale_invariant():
    values = np.asarray([-3.0, -1.0, 0.5, 1.0, 8.0])
    assert np.isclose(_robust_scale(values * 100.0), _robust_scale(values) * 100.0)


def test_round5_path_features_do_not_change_when_future_is_corrupted():
    series = _synthetic_series()
    index = _index(series)
    full, names, paths = build_path_features(series, index)
    cut = dt.date(2020, 3, 10)

    corrupted = {}
    for code, item in series.items():
        values = item.values.copy()
        future = item.dates > cut
        values[future] *= np.linspace(3.0, 30.0, int(future.sum()))
        corrupted[code] = Series(code, item.dates.copy(), values)
    changed, changed_names, changed_paths = build_path_features(corrupted, index)

    past_rows = np.asarray([row[2] <= cut for row in index])
    assert names == changed_names
    np.testing.assert_array_equal(full[past_rows], changed[past_rows])
    np.testing.assert_array_equal(paths[past_rows], changed_paths[past_rows])
    assert not np.array_equal(full[~past_rows], changed[~past_rows])


def test_refit_score_rank_uses_only_the_supplied_past_reference():
    reference = np.asarray([.1, .2, .4, .8])
    current = np.asarray([.05, .3, .9])
    np.testing.assert_array_equal(_rank(reference, current), [.0, .5, 1.0])
    # Values that could arrive later are not an argument and cannot change the
    # already-computed score ranks.
    before = _rank(reference, current)
    _rank(np.r_[reference, [100.0, -100.0]], current)
    np.testing.assert_array_equal(_rank(reference, current), before)
