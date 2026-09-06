"""Independent AP28 hierarchy, refit, state and uncertainty audit."""
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
from research.after_publication_ap28_effective import (
    AP13,
    AP17_CONTROL,
    AP18_CONTROL,
    AP21_STRICT,
    AP23_BEST,
    AP25,
    AP26_BEST,
    AP27_SELECTED,
    AP27_STRICT,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    candidate_key,
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
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    loaded = []
    for folder in (OUT, BASE, AP25, AP13):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap25, ap13 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    X = ap13['features']
    target = ap13['conditional_labels'][:, 3]
    eligible = old['eligible_next'].astype(bool)
    rolling = old['input__primary']
    reserve = old['input__reserve']
    hard_pool = ap25['hard_pool'].astype(bool)
    rolling_rank = ap25['hard_pool_rolling_rank']
    reserve_rank = ap25['hard_pool_reserve_rank']
    for key, value in {
        'features': X, 'target_y20': target, 'hard_pool': hard_pool,
        'hard_pool_rolling_rank': rolling_rank,
        'hard_pool_reserve_rank': reserve_rank,
        'expert__rolling': rolling, 'reserve_score': reserve,
    }.items():
        same(saved[key], value)
    prediction, logs = fit_all(
        panel, X, target, eligible, hard_pool, rolling_rank, reserve_rank, cap)
    same(saved['prediction__hier_y20_weight4'], prediction)
    saved_logs = pd.read_csv(OUT / 'training_log.csv')
    rebuilt_logs = pd.DataFrame(logs)
    pd.testing.assert_frame_equal(
        saved_logs, rebuilt_logs, check_dtype=False, check_exact=False,
        rtol=1e-12, atol=1e-12)
    signals, diagnostics = build_policies(
        panel, prediction, rolling, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in diagnostics.items():
        same(saved[f'{name}__{candidate_key()}'], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_X = X.copy()
    bad_target = target.copy()
    bad_pool = hard_pool.copy()
    bad_rolling_rank = rolling_rank.copy()
    bad_reserve_rank = reserve_rank.copy()
    bad_X[cut:] = 777.
    bad_target[cut:] = 1. - bad_target[cut:]
    bad_pool[cut:] = ~bad_pool[cut:]
    bad_rolling_rank[cut:] = .999
    bad_reserve_rank[cut:] = .001
    bad_prediction, _ = fit_all(
        panel, bad_X, bad_target, eligible, bad_pool, bad_rolling_rank,
        bad_reserve_rank, cap)
    same(prediction[:cut], bad_prediction[:cut])
    bad_signals, bad_diagnostics = build_policies(
        panel, bad_prediction, rolling, reserve, eligible, old)
    np.testing.assert_array_equal(
        signals[candidate_key()][:cut], bad_signals[candidate_key()][:cut])
    for name in diagnostics:
        same(diagnostics[name][:cut], bad_diagnostics[name][:cut])

    fresh = [candidate_key()]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[candidate_key(), 'joint_early_pass'])
    assert selection['selected'] == (candidate_key() if passed else AP21_STRICT)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate_key(), control, block)
        for control in (AP27_SELECTED, AP27_STRICT, AP26_BEST, AP23_BEST,
                        AP21_STRICT, AP18_CONTROL, AP17_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    row = later.loc[candidate_key()]
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_min_rate': float(row.min_rate),
        'late_strict_success': strict,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'hard_pool_and_regime_features_exact': True,
        'all_17_hierarchical_fits_refit_exact': True,
        'all_policy_states_rebuilt': True,
        'future_feature_label_regime_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
