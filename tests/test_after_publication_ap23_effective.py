import datetime as dt

import numpy as np

from research.after_publication_ap23_effective_models import (
    EXPERTS,
    PAIRINGS,
    causal_top_precision,
    competence_pairings,
    unknown_utility,
)


def test_unknown_utility_excludes_known_h1():
    outcomes = {f'y{h}': np.array([1., 0., 1.]) for h in (3, 5, 10, 20)}
    outcomes['y1'] = np.array([0., 1., 0.])
    expected = np.array([1., 0., 1.])
    np.testing.assert_array_equal(unknown_utility(outcomes), expected)
    outcomes['y1'][:] = 1 - outcomes['y1']
    np.testing.assert_array_equal(unknown_utility(outcomes), expected)


def test_top_precision_is_mature_only_and_prefix_invariant():
    n = 260
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i)
                      for i in range(n)], dtype=object)
    currencies = np.array(['AMD'] * n)
    rank = np.linspace(0, 1, n)
    utility = (np.arange(n) % 3 == 0).astype(float)
    mature = np.array([day + dt.timedelta(days=20) for day in dates], dtype=object)
    eligible = np.ones(n, dtype=bool)
    original = causal_top_precision(
        rank, utility, mature, dates, currencies, eligible, 365)
    changed_utility = utility.copy()
    changed_mature = mature.copy()
    changed_utility[180:] = 1 - changed_utility[180:]
    changed_mature[180:] = dates[180:]
    changed = causal_top_precision(
        rank, changed_utility, changed_mature, dates, currencies, eligible, 365)
    for left, right in zip(original, changed):
        np.testing.assert_array_equal(left[:180], right[:180])
    # The current label cannot enter its own competence score even if marked old.
    one = utility.copy()
    one[140] = 1 - one[140]
    altered = causal_top_precision(rank, one, mature, dates, currencies, eligible, 365)
    np.testing.assert_array_equal(original[0][:141], altered[0][:141])


def test_competence_pairings_are_complete_bounded_and_prefix_invariant():
    rng = np.random.default_rng(23)
    days = 220
    dates = np.repeat(np.array([dt.date(2023, 1, 1) + dt.timedelta(days=i)
                                for i in range(days)], dtype=object), 5)
    currencies = np.tile(np.array(['AMD', 'KGS', 'KZT', 'TJS', 'UZS']), days)
    experts = {name: rng.normal(size=len(dates)) for name in EXPERTS}
    outcomes = {f'y{h}': rng.integers(0, 2, size=len(dates)).astype(float)
                for h in (3, 5, 10, 20)}
    mature = np.array([day + dt.timedelta(days=25) for day in dates], dtype=object)
    eligible = np.ones(len(dates), dtype=bool)
    result = competence_pairings(
        experts, outcomes, mature, dates, currencies, eligible)
    pairs, _, _, _, weights, _ = result
    assert tuple(pairs) == PAIRINGS
    assert set(np.unique(weights['primary_hard730'])) <= {0., 1.}
    for name, value in weights.items():
        assert np.isfinite(value).all(), name
        assert ((value >= 0) & (value <= 1)).all(), name

    cut = 700
    changed_experts = {name: value.copy() for name, value in experts.items()}
    changed_outcomes = {name: value.copy() for name, value in outcomes.items()}
    changed_mature = mature.copy()
    for j, value in enumerate(changed_experts.values(), start=1):
        value[cut:] = j * 999.
    for value in changed_outcomes.values():
        value[cut:] = 1 - value[cut:]
    changed_mature[cut:] = dates[cut:]
    changed = competence_pairings(
        changed_experts, changed_outcomes, changed_mature, dates, currencies, eligible)
    for name in PAIRINGS:
        for left, right in zip(pairs[name], changed[0][name]):
            np.testing.assert_array_equal(left[:cut], right[:cut])
    for name in weights:
        np.testing.assert_array_equal(weights[name][:cut], changed[4][name][:cut])
