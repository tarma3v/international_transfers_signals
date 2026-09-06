"""Independent AP14 score, controller, causality and uncertainty audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap13_effective import clustering_stats
from research.after_publication_ap14_effective import (
    ANCHOR,
    AP1_CAP,
    AP12,
    BASE,
    EXTRA_CONTROL,
    FROZEN_CONTROLS,
    HAZARD,
    LOCAL_CONTROL,
    OUT,
    RESERVE_CONTROL,
    ROLLING_CONTROL,
    ROUTER_CONTROL,
    SIMPLE,
    build_policies,
)
from research.after_publication_ap14_effective_models import (
    POLICY_KINDS,
    SCORE_NAMES,
    fixed_scores,
    light_cadence_policy,
)
from research.after_publication_panel import build_outcomes


NEAR_CADENCE = 'extra_ap12_silence14_r80_cap2'
ADAPTIVE = 'extra_ap12_adaptive105_cap2'
STRICT_CADENCE = 'extra_ap12_top35_cap2'


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

    scores = fixed_scores(old, ap12)
    for key, score in scores.items():
        np.testing.assert_array_equal(saved['score__' + key], score)
    np.testing.assert_allclose(
        scores['blend_roll2_local'],
        .5 * scores['extra_roll2'] + .5 * scores['local_extra'],
        rtol=0, atol=0)

    eligible = saved['eligible_next'].astype(bool)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, scores, reserve, eligible, old)
    for key, signal in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], signal)
        assert not signal[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            np.testing.assert_array_equal(saved[f'{name}__{candidate}'], value)
        assert set(np.unique(detail['reason'])).issubset({0, 1, 2})
        assert np.array_equal(signal := signals[candidate], detail['reason'] > 0)
        assert np.isin(detail['threshold'], [.65, .675, .70]).all()
        if '_top325_' in candidate:
            assert (detail['threshold'] == .675).all()
        if '_top35_' in candidate:
            assert (detail['threshold'] == .65).all()
        if '_silence' in candidate:
            assert (detail['threshold'] == .70).all()
            assert not (detail['reason'] == 2)[~eligible].any()

    # Explicit prefix-invariance check on every score/controller pair.
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    cut = np.searchsorted(dates, __import__('datetime').date(2025, 1, 1), side='left')
    for score_name in SCORE_NAMES:
        for kind in POLICY_KINDS:
            corrupt_primary = scores[score_name].copy()
            corrupt_reserve = reserve.copy()
            corrupt_primary[cut:] = np.nan_to_num(corrupt_primary[cut:], nan=0.) + 1000
            corrupt_reserve[cut:] = np.nan_to_num(corrupt_reserve[cut:], nan=0.) - 1000
            rebuilt = light_cadence_policy(
                corrupt_primary, corrupt_reserve, dates, currencies, eligible, kind)
            original = diagnostics[f'{score_name}_{kind}_cap2']
            for left, right in zip((signals[f'{score_name}_{kind}_cap2'],
                                    original['primary_rank'], original['reserve_rank'],
                                    original['threshold'], original['reason']), rebuilt):
                np.testing.assert_array_equal(left[:cut], right[:cut])

    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    fresh = [f'{score}_{kind}_cap2' for score in SCORE_NAMES for kind in POLICY_KINDS]
    passing = early.loc[fresh]
    passing = passing[passing.joint_early_pass].sort_values(
        ['min_lift', 'mean_lift', 'calendar_gap_max'],
        ascending=[False, False, True], kind='stable')
    assert len(passing) == selection['fresh_joint_early_pass_count']
    assert selection['selected'] == passing.index[0]
    assert selection['selection_horizons'] == [3, 5, 10, 20]
    assert not selection['used_registered_rate_cap_fallback']

    late_h5 = pd.read_csv(OUT / 'retrospective_all_horizons.csv').query('h == 5')
    best_late = str(late_h5[late_h5.candidate.isin(fresh)].sort_values(
        'adjusted_lift', ascending=False).iloc[0].candidate)
    late_summary = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    strict = late_summary.loc[fresh]
    strict = strict[(strict.min_rate >= 1) & (strict.max_rate <= 2)
                    & (strict.empty_complete_months == 0)].sort_values(
                        ['min_lift', 'mean_lift'], ascending=False, kind='stable')
    assert len(strict) and strict.index[0] == STRICT_CADENCE

    diagnostic = [selection['selected'], best_late, NEAR_CADENCE, ADAPTIVE, STRICT_CADENCE]
    controls = [EXTRA_CONTROL, ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL,
                ROUTER_CONTROL, ANCHOR, HAZARD, AP1_CAP]
    common_audit(OUT, diagnostic,
                 [(control, diagnostic) for control in controls])
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in diagnostic
        for control in controls
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)

    # Operational cadence is a property of decision dates, not of h20 maturity;
    # preserve both the strict all-h scorecard gate and the h5 near-cadence view.
    clusters = clustering_stats(panel, signals, saved['later'], np.isfinite(outcomes['y5']))
    h5 = late_h5.set_index('candidate')
    diagnostics_out = {
        'early_selected': selection['selected'],
        'best_late_fresh_h5': best_late,
        'near_cadence_candidate': NEAR_CADENCE,
        'near_cadence_h5_lift': float(h5.loc[NEAR_CADENCE, 'adjusted_lift']),
        'near_cadence_h5_rate': float(h5.loc[NEAR_CADENCE, 'frequency']),
        'near_cadence_h5_currency_rate_min': float(h5.loc[NEAR_CADENCE, 'currency_rate_min']),
        'near_cadence_empty_complete_months': int(clusters.loc[NEAR_CADENCE, 'empty_complete_months']),
        'strict_all_h_cadence_candidate': STRICT_CADENCE,
        'strict_all_h_min_lift': float(late_summary.loc[STRICT_CADENCE, 'min_lift']),
        'strict_all_h_h5_lift': float(h5.loc[STRICT_CADENCE, 'adjusted_lift']),
        'adaptive_candidate': ADAPTIVE,
        'adaptive_h5_lift': float(h5.loc[ADAPTIVE, 'adjusted_lift']),
    }
    (OUT / 'late_diagnostics.json').write_text(json.dumps(diagnostics_out, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'frozen_scores_unchanged': True,
        'fixed_blend_exact': True,
        'all_signals_and_controller_state_rebuilt': True,
        'future_score_prefix_invariance_all_20_policies': True,
        'known_down_veto_all_policies': True,
        'weekly_cap_checked_by_common_audit': True,
        'early_selection_rebuilt': selection['selected'],
        'h1_excluded_from_selection': True,
        'best_late_fresh_h5': best_late,
        'strict_all_h_cadence': STRICT_CADENCE,
        'historical_receipts_certified': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
