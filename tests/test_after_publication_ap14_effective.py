import datetime as dt

import numpy as np

from research.after_publication_ap14_effective_models import (
    fixed_scores,
    light_cadence_policy,
)


def _days(n):
    return np.array([dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)])


def test_fixed_blend_is_exact_and_outcome_free():
    ap13 = {
        'prediction__extra_roll2': np.array([.2, .8]),
        'prediction__local_extra': np.array([.4, .6]),
    }
    ap12 = {'prediction__extra_h5': np.array([.1, .9])}
    scores = fixed_scores(ap13, ap12)
    np.testing.assert_allclose(scores['blend_roll2_local'], [.3, .7], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(scores['extra_ap12'], [.1, .9])


def test_light_policy_vetoes_and_caps_every_iso_week():
    n = 35
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    score = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    eligible[22] = False
    signal, _, _, _, _ = light_cadence_policy(
        score, score, dates, currency, eligible, 'top35', window=8, warmup=4)
    assert not signal[22]
    for iso in set(day.isocalendar()[:2] for day in dates):
        assert signal[[day.isocalendar()[:2] == iso for day in dates]].sum() <= 2


def test_silence_reserve_waits_from_first_observation():
    n = 45
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.zeros(n)
    reserve = np.arange(n, dtype=float)
    eligible = np.ones(n, dtype=bool)
    signal, _, _, _, reason = light_cadence_policy(
        primary, reserve, dates, currency, eligible, 'silence14_r80', window=8, warmup=4)
    reserve_ids = np.flatnonzero(reason == 2)
    assert len(reserve_ids) >= 1
    assert dates[reserve_ids[0]] - dates[0] >= dt.timedelta(days=14)
    assert np.array_equal(signal, reason > 0)


def test_adaptive_threshold_uses_only_prior_selected_dates():
    n = 90
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    score = np.sin(np.arange(n) / 3.)
    eligible = np.ones(n, dtype=bool)
    _, _, _, threshold, _ = light_cadence_policy(
        score, score, dates, currency, eligible, 'adaptive105', window=12, warmup=5)
    assert np.all(threshold[:56] == .675)
    assert set(np.unique(threshold[56:])).issubset({.65, .675, .70})


def test_future_score_corruption_cannot_change_policy_prefix():
    n = 70
    dates = _days(n)
    currency = np.array(['AMD'] * n)
    primary = np.sin(np.arange(n))
    reserve = np.cos(np.arange(n))
    eligible = np.ones(n, dtype=bool)
    base = light_cadence_policy(
        primary, reserve, dates, currency, eligible, 'adaptive105', window=12, warmup=5)
    changed_primary = primary.copy()
    changed_reserve = reserve.copy()
    changed_primary[48:] = 1e9
    changed_reserve[48:] = -1e9
    other = light_cadence_policy(
        changed_primary, changed_reserve, dates, currency, eligible,
        'adaptive105', window=12, warmup=5)
    for left, right in zip(base, other):
        np.testing.assert_array_equal(left[:48], right[:48])
