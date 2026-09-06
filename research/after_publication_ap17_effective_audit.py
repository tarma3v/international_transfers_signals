"""Independent AP17 single-policy state, success and uncertainty audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap13_effective import clustering_stats
from research.after_publication_ap17_effective import (
    ANCHOR,
    AP1_CAP,
    AP12,
    AP12_CONTROL,
    AP14_SELECTED,
    AP15_SELECTED,
    AP16_SELECTED,
    BASE,
    CANDIDATE,
    FROZEN_CONTROLS,
    HAZARD,
    LOCAL_CONTROL,
    NEAR_CONTROL,
    OUT,
    RESERVE_CONTROL,
    ROLLING_CONTROL,
    SIMPLE,
    STRICT_CONTROL,
    build_policies,
)
from research.after_publication_ap17_effective_models import paced_month_policy
from research.after_publication_panel import build_outcomes


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    with np.load(OUT / 'outputs.npz') as result, np.load(BASE / 'outputs.npz') as source, \
            np.load(AP12 / 'outputs.npz') as base:
        saved = {key: result[key] for key in result.files}
        old = {key: source[key] for key in source.files}
        ap12 = {key: base[key] for key in base.files}
    for key in ('dates', 'currencies', 'early', 'later', 'groups', 'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)], outcomes[kind + str(h)])
    score = ap12['prediction__extra_h5']
    np.testing.assert_array_equal(saved['score__extra_ap12'], score)
    reserve = ap12['score__known70_hazard30']
    eligible = saved['eligible_next'].astype(bool)
    signals, detail = build_policies(panel, score, reserve, eligible, old)
    for key, signal in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], signal)
        assert not signal[~eligible].any()
    for name, value in detail.items():
        np.testing.assert_array_equal(saved[f'{name}__{CANDIDATE}'], value)
    signal = signals[CANDIDATE]
    reason = detail['reason']
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    assert np.array_equal(signal, reason > 0)
    assert set(np.unique(reason)).issubset({0, 1, 2, 3})
    assert (detail['primary_rank'][reason == 1] > .70).all()
    paced = reason == 2
    assert (detail['primary_rank'][paced] > .55).all()
    assert (detail['reserve_rank'][paced] > .70).all()
    assert (detail['trailing_rate'][paced] < 1.).all()
    assert all(day.day >= 24 for day in dates[reason == 3])
    for i in np.flatnonzero(reason == 3):
        same_month_before = ((currencies == currencies[i])
                             & np.array([(day.year, day.month) ==
                                         (dates[i].year, dates[i].month)
                                         for day in dates])
                             & (np.arange(len(dates)) < i))
        assert not signal[same_month_before].any()

    cut = np.searchsorted(dates, dt.date(2025, 1, 1), side='left')
    p2, r2 = score.copy(), reserve.copy()
    p2[cut:] = np.nan_to_num(p2[cut:], nan=0.) + 1000
    r2[cut:] = np.nan_to_num(r2[cut:], nan=0.) - 1000
    rebuilt = paced_month_policy(p2, r2, dates, currencies, eligible)
    for left, right in zip((signal, detail['primary_rank'], detail['reserve_rank'],
                            detail['trailing_rate'], reason), rebuilt):
        np.testing.assert_array_equal(left[:cut], right[:cut])

    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert selection['selected'] == CANDIDATE
    assert bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selection_horizons'] == [3, 5, 10, 20]
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    row = later.loc[CANDIDATE]
    clusters = clustering_stats(panel, signals, saved['later'], np.isfinite(outcomes['y5']))
    assert row.min_rate >= 1 and row.max_rate <= 2
    assert row.min_lift > 2.35
    assert clusters.loc[CANDIDATE, 'empty_complete_months'] == 0

    controls = [AP16_SELECTED, AP12_CONTROL, NEAR_CONTROL, STRICT_CONTROL,
                AP15_SELECTED, AP14_SELECTED, ROLLING_CONTROL, LOCAL_CONTROL,
                RESERVE_CONTROL, ANCHOR, HAZARD, AP1_CAP]
    common_audit(OUT, [CANDIDATE], [(control, [CANDIDATE]) for control in controls])
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in controls for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    late_h5 = pd.read_csv(OUT / 'retrospective_all_horizons.csv').query('h == 5').set_index('candidate')
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected': CANDIDATE, 'strict_all_unknown_h_success': True,
        'h5_lift': float(late_h5.loc[CANDIDATE, 'adjusted_lift']),
        'h5_rate': float(late_h5.loc[CANDIDATE, 'frequency']),
        'h5_currency_rate_min': float(late_h5.loc[CANDIDATE, 'currency_rate_min']),
        'min_unknown_h_lift': float(row.min_lift),
        'min_unknown_h_rate': float(row.min_rate),
        'empty_complete_months': 0,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'frozen_ap12_score_unchanged': True,
        'single_policy_state_rebuilt': True,
        'pace_threshold_rate_and_reserve_checked': True,
        'month_rescue_first_in_currency_month': True,
        'future_score_prefix_invariance': True,
        'known_down_veto': True, 'weekly_cap_checked': True,
        'early_selection_rebuilt': CANDIDATE,
        'h1_excluded_from_selection': True,
        'strict_all_unknown_h_success': True,
        'historical_receipts_certified': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
