import datetime as dt

import numpy as np

from research.after_publication_ap12_effective_models import (
    causal_rank_cap2,
    causal_rank_percentile,
    compact_feature_indices,
    first_failure_class,
)


def test_first_failure_class_uses_unknown_intervals_only():
    y = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    np.testing.assert_equal(first_failure_class(y), [0, 1, 2, 3, 4, np.nan])


def test_compact_features_contain_signal_regime_currency_and_market_age():
    names = ['junk', 'known_change_z', 'after2022', 'currency_AMD',
             'CNY_source_age_days', 'announced_ret5', 'effective_vol20']
    selected = [names[i] for i in compact_feature_indices(names)]
    assert selected == names[1:]


def test_causal_rank_controller_vetoes_and_caps_each_week():
    dates = np.array([dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(14)])
    values = np.arange(14, dtype=float)
    currencies = np.array(['AMD'] * 14)
    eligible = np.ones(14, dtype=bool)
    eligible[5] = False
    fired = causal_rank_cap2(values, dates, currencies, eligible, rate=.5,
                             minimum_gap=0, window=4, warmup=2)
    assert not fired[5]
    for iso in set(d.isocalendar()[:2] for d in dates):
        assert fired[[d.isocalendar()[:2] == iso for d in dates]].sum() <= 2


def test_causal_rank_prefix_is_unchanged_by_future_corruption():
    dates = np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(20)])
    values = np.sin(np.arange(20))
    currencies = np.array(['AMD'] * 20)
    eligible = np.ones(20, dtype=bool)
    base = causal_rank_cap2(values, dates, currencies, eligible, .3, 0, window=6, warmup=3)
    changed = values.copy()
    changed[12:] = 1e6
    corrupted = causal_rank_cap2(changed, dates, currencies, eligible, .3, 0, window=6, warmup=3)
    np.testing.assert_array_equal(base[:12], corrupted[:12])
    p0 = causal_rank_percentile(values, currencies, window=6, warmup=3)
    p1 = causal_rank_percentile(changed, currencies, window=6, warmup=3)
    np.testing.assert_array_equal(p0[:12], p1[:12])


def test_gap2_requires_one_full_day_between_signals():
    dates = np.array([dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(10)])
    values = np.arange(10, dtype=float)
    currencies = np.array(['AMD'] * 10)
    eligible = np.ones(10, dtype=bool)
    fired = causal_rank_cap2(values, dates, currencies, eligible, 1., 2, window=3, warmup=1)
    chosen = dates[fired]
    assert all((b - a).days >= 2 for a, b in zip(chosen, chosen[1:]))
