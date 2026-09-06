import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml.data import Series
from ml.targets import HORIZONS
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_features import market_features, session_state
from research.after_publication_ap3_policy import sequential_policy
from research.after_publication_ap8 import CLOCKS, FAMILIES, clock_policies, fit_clock, freeze_policies


def example():
    day = dt.date(2024, 1, 9)
    begins = pd.date_range('2024-01-09 17:30', periods=13, freq='10min')
    values = np.arange(len(begins), dtype=float) + 10
    frame = pd.DataFrame({'begin': begins, 'end': begins + pd.Timedelta(minutes=9, seconds=59),
                         **{c: values for c in ('open', 'close', 'high', 'low')}})
    dates = np.array([dt.date(2024, 1, d) for d in (6, 9, 10, 11)], dtype=object)
    series = {c: Series(c, dates, np.array([8., 9., 10., 999.])) for c in ('TJS', 'CNY')}
    panel = pd.DataFrame([{'date': day, 'currency': 'TJS', 'announced_index': 2, 'announced_price': 10.}])
    return panel, series, frame


def test_default_cutoff_keeps_old_behavior():
    p, s, f = example()
    pd.testing.assert_frame_equal(market_features(p, s, {'CNYRUB_TOM': f}, 20),
        market_features(p, s, {'CNYRUB_TOM': f}, 20, dt.time(18, 30)))


@pytest.mark.parametrize('clock', list(CLOCKS))
def test_unavailable_candles_and_cbr_cannot_change_earlier_snapshot(clock):
    p, s, f = example()
    cutoff = CLOCKS[clock]
    stop = pd.Timestamp(dt.datetime.combine(p.date.iloc[0], cutoff))
    available = (f.end + pd.Timedelta(minutes=20) < stop) & (f.begin + pd.Timedelta(minutes=30) <= stop)
    original = market_features(p, s, {'CNYRUB_TOM': f}, 20, cutoff)
    changed = f.copy()
    changed.loc[~available, ['open', 'close', 'high', 'low']] *= 10000
    s['CNY'].values[-1] = .001
    s['TJS'].values[-1] = .001
    pd.testing.assert_frame_equal(original, market_features(p, s, {'CNYRUB_TOM': changed}, 20, cutoff))
    assert original.cny_announced_price.iloc[0] == 10.


def test_new_bar_requires_both_nominal_and_actual_end_delay():
    p, _, frame = example()
    row = frame.iloc[[3]].copy()  #18:00 nominal completion18:10, delayed18:30.
    row['end'] = pd.Timestamp('2024-01-09 18:00:01')
    day = p.date.iloc[0]
    assert session_state(row, day, dt.time(18, 29, 59), 20)['n'] == 0
    assert session_state(row, day, dt.time(18, 30), 20)['n'] == 1
    row['end'] = pd.Timestamp('2024-01-09 18:10')
    assert session_state(row, day, dt.time(18, 30), 20)['n'] == 0


def test_missing_next_day_has_no_overnight_carry():
    p, s, frame = example()
    p['date'] = dt.date(2024, 1, 13)
    for cutoff in CLOCKS.values():
        market = market_features(p, s, {'CNYRUB_TOM': frame}, 20, cutoff)
        assert market.cny_n.iloc[0] == 0 and market.cny_last_missing.iloc[0] == 1
        assert market.cny_quality.iloc[0] == 0


def test_frozen_controls_copy_entire_scores_and_policy_prefix_is_causal():
    rng = np.random.default_rng(8)
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(200)])
    cur = np.repeat('TJS', len(dates))
    panel = pd.DataFrame({'date': dates, 'currency': cur})
    parts = clock_policies(panel, rng.normal(size=200), rng.uniform(size=200), rng.uniform(size=(200, 5)))
    raw = {'t1830_' + k: v for k, v in parts.items()}
    frozen = freeze_policies(raw)
    assert len(frozen) == 6
    for clock in ('1850', '1930'):
        for family in FAMILIES:
            np.testing.assert_array_equal(frozen[f't{clock}_frozen1830_{family}'], raw['t1830_' + family])
    original = sequential_policy(parts['cny_hist50'], dates, cur, 'urgent_cap2')
    changed = parts['cny_hist50'].copy()
    changed[150:] = 1000
    np.testing.assert_array_equal(original[:150], sequential_policy(changed, dates, cur, 'urgent_cap2')[:150])


def test_models_never_train_on_immature_or_future_outcomes():
    rng = np.random.default_rng(8)
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(510)], dtype=object)
    panel = pd.DataFrame({'date': dates})
    X = rng.normal(size=(len(dates), 4))
    failure = rng.integers(1, 22, size=len(dates))
    outcomes = {f'y{h}': (failure > h).astype(float) for h in HORIZONS}
    outcomes['mature20'] = dates + dt.timedelta(days=28)
    origin = dt.date(2023, 4, 1)
    tr = matured_mask(panel, outcomes, origin)
    assert tr.sum() >= 400 and outcomes['mature20'][tr].max() < origin - dt.timedelta(days=2)
    direct, hazard, logs = fit_clock(panel, X, X, outcomes, [origin])
    corrupted = {k: v.copy() for k, v in outcomes.items()}
    for h in HORIZONS:
        corrupted[f'y{h}'][~tr] = 999  #Would violate survival design if used.
    direct2, hazard2, logs2 = fit_clock(panel, X, X, corrupted, [origin])
    np.testing.assert_array_equal(direct, direct2)
    np.testing.assert_array_equal(hazard, hazard2)
    assert logs == logs2
    assert logs[0]['n_train'] == tr.sum()
