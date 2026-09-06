"""Independent AP16 pacing state, selection, cadence and uncertainty audit."""
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
from research.after_publication_ap16_effective import (
    ANCHOR,
    AP1_CAP,
    AP12,
    AP12_CONTROL,
    AP14_ADAPTIVE,
    AP14_SELECTED,
    AP15_SELECTED,
    BASE,
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
    fresh_key,
)
from research.after_publication_ap16_effective_models import POLICY_KINDS, paced_policy
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
    signals, diagnostics = build_policies(panel, score, reserve, eligible, old)
    for key, signal in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], signal)
        assert not signal[~eligible].any()
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            np.testing.assert_array_equal(saved[f'{name}__{candidate}'], value)
        assert np.array_equal(signals[candidate], detail['reason'] > 0)
        assert set(np.unique(detail['reason'])).issubset({0, 1, 2, 3})
        assert np.isin(detail['threshold'], [.55, .60, .65, .675, .70]).all()
        assert (detail['primary_rank'][detail['reason'] == 1] > .70).all()
        assert all(day.day >= 24 for day in dates[detail['reason'] == 3])
        if 'p55_r70' in candidate:
            pace = detail['reason'] == 2
            assert (detail['primary_rank'][pace] > .55).all()
            assert (detail['reserve_rank'][pace] > .70).all()
            assert (detail['trailing_rate'][pace] < 1.).all()
        elif 'pace' in candidate:
            pace = detail['reason'] == 2
            assert (detail['primary_rank'][pace] > .60).all()
            assert (detail['trailing_rate'][pace] < 1.).all()
        for i in np.flatnonzero(detail['reason'] == 3):
            same_month_before = ((currencies == currencies[i])
                                 & np.array([(d.year, d.month) ==
                                             (dates[i].year, dates[i].month)
                                             for d in dates])
                                 & (np.arange(len(dates)) < i))
            assert not signals[candidate][same_month_before].any()

    cut = np.searchsorted(dates, dt.date(2025, 1, 1), side='left')
    for kind in POLICY_KINDS:
        p2, r2 = score.copy(), reserve.copy()
        p2[cut:] = np.nan_to_num(p2[cut:], nan=0.) + 1000
        r2[cut:] = np.nan_to_num(r2[cut:], nan=0.) - 1000
        rebuilt = paced_policy(p2, r2, dates, currencies, eligible, kind)
        candidate = fresh_key(kind)
        original = diagnostics[candidate]
        for left, right in zip((signals[candidate], original['primary_rank'],
                                original['reserve_rank'], original['trailing_rate'],
                                original['threshold'], original['reason']), rebuilt):
            np.testing.assert_array_equal(left[:cut], right[:cut])

    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    fresh = [fresh_key(kind) for kind in POLICY_KINDS]
    passing = early.loc[fresh]
    passing = passing[passing.joint_early_pass].sort_values(
        ['min_lift', 'mean_lift', 'calendar_gap_max'],
        ascending=[False, False, True], kind='stable')
    assert selection['selected'] == passing.index[0]
    assert len(passing) == selection['fresh_joint_early_pass_count'] == 5
    assert selection['selection_horizons'] == [3, 5, 10, 20]

    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    strict_all_h = later.loc[fresh]
    strict_all_h = strict_all_h[(strict_all_h.min_rate >= 1)
                                & (strict_all_h.max_rate <= 2)
                                & (strict_all_h.empty_complete_months == 0)
                                & (strict_all_h.min_lift > 2.35)]
    late_h5 = pd.read_csv(OUT / 'retrospective_all_horizons.csv').query('h == 5').set_index('candidate')
    clusters = clustering_stats(panel, signals, saved['later'], np.isfinite(outcomes['y5']))
    strict_h5 = [key for key in fresh
                 if late_h5.loc[key, 'currency_rate_min'] >= 1
                 and clusters.loc[key, 'empty_complete_months'] == 0]

    diagnostics_to_report = list(dict.fromkeys([selection['selected'], *fresh]))
    controls = [AP12_CONTROL, NEAR_CONTROL, AP14_ADAPTIVE, STRICT_CONTROL,
                AP14_SELECTED, AP15_SELECTED, ROLLING_CONTROL, LOCAL_CONTROL,
                RESERVE_CONTROL, ANCHOR, HAZARD, AP1_CAP]
    common_audit(OUT, diagnostics_to_report,
                 [(control, diagnostics_to_report) for control in controls])
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in diagnostics_to_report
        for control in controls
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'early_selected': selection['selected'],
        'strict_all_unknown_h_successes': list(strict_all_h.index),
        'strict_h5_successes': strict_h5,
        'selected_h5_lift': float(late_h5.loc[selection['selected'], 'adjusted_lift']),
        'selected_h5_rate': float(late_h5.loc[selection['selected'], 'frequency']),
        'selected_h5_currency_rate_min': float(
            late_h5.loc[selection['selected'], 'currency_rate_min']),
        'selected_min_unknown_h_lift': float(later.loc[selection['selected'], 'min_lift']),
        'selected_min_unknown_h_rate': float(later.loc[selection['selected'], 'min_rate']),
        'selected_empty_complete_months': int(
            clusters.loc[selection['selected'], 'empty_complete_months']),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'frozen_ap12_score_unchanged': True,
        'all_five_pacing_states_rebuilt': True,
        'pace_reason_threshold_and_rate_checked': True,
        'month_rescue_first_in_currency_month': True,
        'future_score_prefix_invariance_all_five': True,
        'known_down_veto_all_policies': True,
        'early_selection_rebuilt': selection['selected'],
        'h1_excluded_from_selection': True,
        'strict_all_unknown_h_success_count': len(strict_all_h),
        'strict_h5_success_count': len(strict_h5),
        'historical_receipts_certified': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
