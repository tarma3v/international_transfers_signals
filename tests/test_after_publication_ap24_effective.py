import datetime as dt

import numpy as np

from research.after_publication_ap24_effective import pair_name, ranking_pairs
from research.after_publication_ap24_effective_models import (
    SPECS,
    group_keys,
    grouped_training_rows,
    ranking_target,
)


def test_ranking_target_and_specs_exclude_h1():
    labels = np.array([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    np.testing.assert_array_equal(ranking_target(labels, 'mean'), [.5, 1.])
    np.testing.assert_array_equal(ranking_target(labels, 'y20'), [0., 1.])
    assert len(SPECS) == 5
    assert all('h1' not in name for name in SPECS)


def test_group_keys_are_currency_period_specific():
    dates = np.array([dt.date(2024, 1, 3), dt.date(2024, 2, 3),
                      dt.date(2024, 4, 3)], dtype=object)
    currencies = np.array(['AMD', 'AMD', 'AMD'])
    np.testing.assert_array_equal(
        group_keys(dates, currencies, 'quarter'),
        ['AMD-2024-01', 'AMD-2024-01', 'AMD-2024-02'])
    np.testing.assert_array_equal(
        group_keys(dates, currencies, 'month'),
        ['AMD-2024-01', 'AMD-2024-02', 'AMD-2024-04'])


def test_grouped_rows_remove_constant_groups_and_are_contiguous():
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i * 20)
                      for i in range(9)], dtype=object)
    currencies = np.array(['AMD'] * 9)
    target = np.array([0., 1., 0., 1., 0., 1., 1., 1., 1.])
    indices, group_id, stats = grouped_training_rows(
        np.ones(9, dtype=bool), target, dates, currencies, 'quarter')
    assert stats['n_groups_total'] >= stats['n_groups_kept'] >= 1
    assert len(indices) == stats['n_group_rows']
    assert np.all(np.diff(group_id) >= 0)
    for group in np.unique(group_id):
        assert np.unique(target[indices[group_id == group]]).size >= 2


def test_ranking_pairs_have_both_fixed_roles():
    predictions = {name: np.arange(20, dtype=float) + j
                   for j, name in enumerate(SPECS)}
    rolling = np.arange(20, dtype=float) + 100
    cat = np.arange(20, dtype=float) + 200
    pairs = ranking_pairs(predictions, rolling, cat)
    assert len(pairs) == 2 * len(SPECS)
    for name in SPECS:
        np.testing.assert_array_equal(pairs[pair_name(name, 'primary')][0],
                                      predictions[name])
        np.testing.assert_array_equal(pairs[pair_name(name, 'primary')][1], cat)
        np.testing.assert_array_equal(pairs[pair_name(name, 'pace')][0], rolling)
        np.testing.assert_array_equal(pairs[pair_name(name, 'pace')][1],
                                      predictions[name])
