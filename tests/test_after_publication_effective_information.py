import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml.data import CORRIDORS, Series
from research.after_publication_effective_information import past_features, fit_models, policies, SIMPLE
from research.after_publication_panel import build_features, REFERENCES
from research.after_publication_ap4 import INCUMBENT
from research.after_publication_effective_market_control import effective_market


def series_fixture():
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(330)], dtype=object)
    result = {}
    for j, c in enumerate(dict.fromkeys(CORRIDORS + REFERENCES)):
        values = np.exp(.0002 * np.arange(len(dates)) + .015 * np.sin(np.arange(len(dates)) / (9 + j))) * (10 + j)
        result[c] = Series(c, dates, values)
    return result


def test_every_currency_future_announcements_absent_from_past_features():
    series = series_fixture()
    panel, _, names = build_features(series)
    panel = panel.iloc[:5]
    day = panel.date.iloc[0]
    original = past_features(series, panel, names)
    changed = {c: Series(c, s.dates, np.where(s.dates > day, s.values * 8, s.values)) for c, s in series.items()}
    np.testing.assert_array_equal(past_features(changed, panel, names), original)
    assert (original[:, names.index('known_change_z')] == 0).all()
    assert (original[:, names.index('next_effective_gap')] == 0).all()


def test_same_schema_and_effective_history_preserved():
    series = series_fixture()
    panel, full, names = build_features(series)
    old = past_features(series, panel.iloc[:10], names)
    invariant = [i for i, n in enumerate(names) if n.startswith(('effective_', 'currency_', 'dow_', 'annual_'))]
    np.testing.assert_array_equal(old[:, invariant], full[:10, invariant])
    for i, n in enumerate(names):
        if n.startswith('announced_'):
            np.testing.assert_array_equal(old[:, i], old[:, names.index(n.replace('announced_', 'effective_', 1))])


def test_past_features_ignore_panel_announced_metadata():
    series = series_fixture()
    panel, _, names = build_features(series)
    panel = panel.iloc[:5].copy()
    before = past_features(series, panel, names)
    panel['announced_price'] = 1e99
    panel['announced_index'] = 999999
    panel['announced_effective_date'] = '2099-01-01'
    np.testing.assert_array_equal(before, past_features(series, panel, names))


def test_gate_veto_and_prefix_stability():
    n = 300
    dates = np.array([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)], dtype=object)
    price = np.where(np.arange(n) % 3 == 0, 90., 110.)
    panel = pd.DataFrame({'date': dates, 'currency': ['KZT'] * n, 'announced_price': price, 'current_price': 100.})
    rng = np.random.default_rng(4)
    scores = {f'{info}_{model}': rng.random(n) for info in ('past', 'announced', 'market')
              for model in ('hist', 'logit', 'survival_h5', 'survival_mean')}
    X = (price - 100)[:, None]
    previous = {'signal__' + INCUMBENT: np.zeros(n, dtype=bool)}
    _, signals = policies(panel, scores, X, ['known_change_z'], previous)
    _, prefix = policies(panel.iloc[:200], {k: v[:200] for k, v in scores.items()}, X[:200], ['known_change_z'],
                         {'signal__' + INCUMBENT: previous['signal__' + INCUMBENT][:200]})
    for key, fired in signals.items():
        np.testing.assert_array_equal(fired[:200], prefix[key])
        if '_gated_' in key or key in (SIMPLE, 'known_change_z_urgent_cap2'):
            assert not fired[price < 100].any()


def test_all_three_models_ignore_future_training_information():
    n = 730
    rng = np.random.default_rng(18)
    dates = np.array([dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(n)], dtype=object)
    panel = pd.DataFrame({'date': dates})
    X = rng.normal(size=(n, 4))
    y = np.cumprod(rng.random((n, 5)) > .15, axis=1).astype(float)
    out = {'y' + str(h): y[:, j] for j, h in enumerate((1, 3, 5, 10, 20))}
    out['mature20'] = dates + dt.timedelta(days=28)
    cap = dict(out, mature20=dates + dt.timedelta(days=29))
    origin = dt.date(2023, 7, 1)
    a, curves_a, _ = fit_models(panel, {'past': X}, out, cap, [origin])
    changed_X = X.copy()
    changed_X[dates >= dt.date(2023, 10, 1)] = 1e6
    changed_out = {k: v.copy() for k, v in out.items()}
    immature = cap['mature20'] >= origin - dt.timedelta(days=2)
    for key in ('y1', 'y3', 'y5', 'y10', 'y20'):
        changed_out[key][immature] = 0.
    b, curves_b, _ = fit_models(panel, {'past': changed_X}, changed_out, cap, [origin])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(curves_a['past'], curves_b['past'])


def test_market_control_ignores_unavailable_cbr_values_and_future_bars():
    series = series_fixture()
    panel, _, _ = build_features(series)
    panel = panel.iloc[:5].copy()
    day = panel.date.iloc[0]
    begin = pd.to_datetime([str(day) + ' 10:00:00', str(day) + ' 18:00:00', str(day) + ' 18:10:00'])
    f = pd.DataFrame({'begin': begin, 'end': begin + pd.Timedelta(minutes=9, seconds=59),
                      'open': [15., 16., 90.], 'close': [15., 16., 90.],
                      'low': [14., 15., 89.], 'high': [16., 17., 91.]})
    frames = {'CNYRUB_TOM': f, **{c + 'RUB_TOM': f for c in CORRIDORS}}
    a = effective_market(panel, series, frames)
    changed = {c: Series(c, s.dates, np.where(s.dates > day, s.values * 13, s.values)) for c, s in series.items()}
    future = f.copy()
    future.loc[2, ['open', 'close', 'high', 'low']] *= 100
    b = effective_market(panel, changed, {k: future for k in frames})
    pd.testing.assert_frame_equal(a, b)
    assert (a.cny_n == 2).all()


def test_market_control_ignores_announced_panel_prices():
    series = series_fixture()
    panel, _, _ = build_features(series)
    panel = panel.iloc[:5].copy()
    empty = pd.DataFrame({k: pd.Series(dtype='datetime64[ns]' if k in ('begin', 'end') else float)
                          for k in ('begin', 'end', 'open', 'close', 'high', 'low')})
    a = effective_market(panel, series, {'CNYRUB_TOM': empty})
    panel['announced_price'] = 1e99
    panel['announced_index'] = 999999
    pd.testing.assert_frame_equal(a, effective_market(panel, series, {'CNYRUB_TOM': empty}))
