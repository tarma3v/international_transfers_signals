"""Independent AP35 distributional refit, policy and corruption audit."""
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
from research.after_publication_ap3 import weekly_max
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap35_effective import (
    AP13,
    AP21_STRICT,
    AP23_BEST,
    AP26,
    AP26_BEST,
    AP32_CANDIDATE,
    AP33_CANDIDATE,
    AP34_CANDIDATE,
    BASE,
    CANDIDATE,
    DROP_FEATURES,
    FEATURE_META,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    fit_all,
)
from research.after_publication_panel import build_outcomes


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates = panel.date.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    known_change = 10000 * np.log(
        panel.announced_price.to_numpy() / panel.current_price.to_numpy())
    target = outcomes['floor20'] - known_change
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    loaded = []
    for folder in (OUT, BASE, AP26, AP13):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap26, ap13 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    names = json.loads(FEATURE_META.read_text())['feature_names']
    compact = ap13['compact_feature_indices'].astype(int)
    model_indices = np.array(
        [i for i in compact if names[i] not in DROP_FEATURES], dtype=int)
    X = ap13['features'][:, model_indices]
    same(saved['features'], X)
    np.testing.assert_array_equal(saved['model_feature_indices'], model_indices)
    same(saved['known_change'], known_change)
    same(saved['residual_floor20_target'], target)
    eligible = saved['eligible_next'].astype(bool)

    score, refit_log = fit_all(panel, X, target, cap, eligible, known_change)
    same(saved['model_score'], score)
    stored_log = pd.read_csv(OUT / 'training_log.csv')
    pd.testing.assert_frame_equal(
        stored_log, refit_log, check_dtype=False, check_exact=False,
        rtol=1e-12, atol=1e-12)
    assert (refit_log.monotonic_min_delta >= -1e-12).all()

    rolling = ap26['expert__rolling']
    reserve = ap26['reserve_score']
    fallback = old['signal__' + AP23_BEST].astype(bool)
    same(saved['input__rolling'], rolling)
    same(saved['input__reserve'], reserve)
    np.testing.assert_array_equal(saved['input__fallback_signal'], fallback)
    signals, policy_detail = build_policies(
        panel, score, rolling, reserve, fallback, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in policy_detail.items():
        same(saved['policy__' + name], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_X = X.copy()
    bad_target = target.copy()
    bad_known = known_change.copy()
    bad_X[cut:] = bad_X[cut:] * -13 + 211
    bad_target[cut:] = np.where(np.isfinite(bad_target[cut:]),
                                bad_target[cut:] * -7 + 999, np.nan)
    bad_known[cut:] = bad_known[cut:] * -3 + 517
    bad_score, bad_log = fit_all(
        panel, bad_X, bad_target, cap, eligible, bad_known)
    same(score[:cut], bad_score[:cut])
    prior_log = refit_log[pd.to_datetime(refit_log.origin).dt.date < cut_day]
    bad_prior_log = bad_log[pd.to_datetime(bad_log.origin).dt.date < cut_day]
    pd.testing.assert_frame_equal(
        prior_log.reset_index(drop=True), bad_prior_log.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    bad_fallback = fallback.copy()
    bad_fallback[cut:] = ~bad_fallback[cut:]
    bad_signals, bad_policy = build_policies(
        panel, bad_score, rolling, reserve, bad_fallback, eligible, old)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for name in policy_detail:
        same(policy_detail[name][:cut], bad_policy[name][:cut])

    common_audit(OUT, [CANDIDATE],
                 [(control, [CANDIDATE]) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (CANDIDATE if passed else AP33_CANDIDATE)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in (AP33_CANDIDATE, AP34_CANDIDATE, AP32_CANDIDATE,
                        AP23_BEST, AP26_BEST, AP21_STRICT)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    row = later.loc[CANDIDATE]
    max_week = weekly_max(panel, signals[CANDIDATE], saved['later'])
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0
                  and max_week <= 2)
    scope = saved['later'].astype(bool)
    final_reason = saved['policy__final_reason']
    source_reason = saved['policy__source_reason']
    final_names = ('none', 'new_core', 'ap23_late_week', 'ap23_silence10')
    source_names = ('none', 'rolling_primary', 'distributional_pace', 'month_rescue')
    counts = pd.concat([
        pd.DataFrame({
            'level': level,
            'reason': reason_names,
            'count': [int((scope & (reason == value)).sum())
                      for value in range(len(reason_names))],
        })
        for level, reason_names, reason in (
            ('final', final_names, final_reason),
            ('source', source_names, source_reason),
        )
    ], ignore_index=True)
    counts.to_csv(OUT / 'reason_counts.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_mean_lift': float(row.mean_lift),
        'late_min_rate': float(row.min_rate),
        'late_max_weekly_signals': int(max_week),
        'late_strict_success': strict,
        'late_reason_counts': {
            level: dict(zip(frame.reason, frame['count']))
            for level, frame in counts.groupby('level')
        },
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'all_17_distributional_catboost_fits_refit_exact': True,
        'expanded_labels_recency_weights_and_monotonicity_exact': True,
        'nested_source_and_calendar_policy_state_exact': True,
        'future_feature_label_and_source_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto_inherited_and_checked': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
