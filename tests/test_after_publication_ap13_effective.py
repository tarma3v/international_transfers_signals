import datetime as dt

import numpy as np

from research.after_publication_ap13_effective_models import (
    brier365_router,
    fit_meta_router,
    reserve_policy,
)


def test_brier_router_uses_only_strictly_mature_past_rows():
    dates = np.array([dt.date(2026, 1, 1), dt.date(2026, 1, 1),
                      dt.date(2026, 1, 10), dt.date(2026, 1, 10)])
    mature = np.array([dt.date(2026, 1, 5), dt.date(2026, 1, 10),
                       dt.date.max, dt.date.max], dtype=object)
    y = np.array([1., 0., 1., 0.])
    extra = np.array([.9, .9, .8, .8])
    local = np.array([.1, .1, .2, .2])
    eligible = np.ones(4, dtype=bool)
    score, weights, counts = brier365_router(
        dates, mature, y, eligible, extra, local, minimum=1, eta=20.)
    assert counts[2] == counts[3] == 1
    np.testing.assert_array_equal(weights[2], weights[3])
    assert weights[2, 0] > weights[2, 1]
    assert score[2] > .5


def test_brier_router_prefix_ignores_future_outcome_corruption():
    dates = np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)])
    mature = np.array([d + dt.timedelta(days=2) for d in dates], dtype=object)
    y = np.arange(8, dtype=float) % 2
    extra = np.linspace(.1, .8, 8)
    local = 1 - extra
    eligible = np.ones(8, dtype=bool)
    base = brier365_router(dates, mature, y, eligible, extra, local, minimum=1)[0]
    changed = y.copy()
    changed[5:] = 1 - changed[5:]
    other = brier365_router(dates, mature, changed, eligible, extra, local, minimum=1)[0]
    np.testing.assert_array_equal(base[:7], other[:7])


def test_meta_router_is_a_convex_expert_blend_and_train_only():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(240, 3))
    y = (X[:, 0] > 0).astype(float)
    extra = np.clip(.2 + .6 * y, 0, 1)
    local = 1 - extra
    train = np.zeros(240, dtype=bool)
    train[:220] = True
    query = ~train
    score, weight, n, preferred = fit_meta_router(X, y, train, query, extra, local, minimum=20)
    assert n == 220 and preferred == 220
    assert np.all((weight >= 0) & (weight <= 1))
    lo = np.minimum(extra[query], local[query])
    hi = np.maximum(extra[query], local[query])
    assert np.all((score >= lo) & (score <= hi))


def test_reserve_policy_caps_week_and_vetoes():
    dates = np.array([dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(14)])
    primary = np.arange(14, dtype=float)
    reserve = primary.copy()
    currencies = np.array(['AMD'] * 14)
    eligible = np.ones(14, dtype=bool)
    eligible[7] = False
    fired = reserve_policy(primary, reserve, dates, currencies, eligible,
                           'reserve7', window=4, warmup=2)
    assert not fired[7]
    for iso in set(d.isocalendar()[:2] for d in dates):
        assert fired[[d.isocalendar()[:2] == iso for d in dates]].sum() <= 2


def test_month24_rescue_is_causal_and_once_month_has_signal_no_rescue_needed():
    dates = np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(31)])
    primary = np.zeros(31)
    reserve = np.arange(31, dtype=float)
    currencies = np.array(['AMD'] * 31)
    eligible = np.ones(31, dtype=bool)
    fired = reserve_policy(primary, reserve, dates, currencies, eligible,
                           'month24', window=5, warmup=2)
    assert fired.sum() == 1
    assert dates[fired][0].day == 24


def test_reserve_policy_prefix_does_not_change_from_future_scores():
    dates = np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(20)])
    primary = np.sin(np.arange(20))
    reserve = np.cos(np.arange(20))
    currencies = np.array(['AMD'] * 20)
    eligible = np.ones(20, dtype=bool)
    base = reserve_policy(primary, reserve, dates, currencies, eligible,
                          'reserve7', window=6, warmup=3)
    changed = primary.copy()
    changed[12:] = 1e6
    other = reserve_policy(changed, reserve, dates, currencies, eligible,
                           'reserve7', window=6, warmup=3)
    np.testing.assert_array_equal(base[:12], other[:12])
