"""Independent AP25 hard-pool, refit, state and uncertainty audit."""
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
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap25_effective import (
    AP12,
    AP13,
    AP17_CONTROL,
    AP18_CONTROL,
    AP21_STRICT,
    AP23_BEST,
    AP24_BEST,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    candidate_key,
    fit_all,
    hard_pool,
)
from research.after_publication_ap25_effective_models import SPECS
from research.after_publication_panel import build_outcomes


BEST_LIFT = candidate_key('pace_cat_y20_gate_full')
BEST_H5 = candidate_key('pace_cat_future5_gate_full')


def same(left, right):
    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12,
                               equal_nan=True)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    folders = (OUT, BASE, AP12, AP13)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap12, ap13 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    X = ap13['features']
    labels = ap13['conditional_labels']
    eligible = saved['eligible_next'].astype(bool)
    rolling = old['expert__rolling']
    reserve = ap12['score__known70_hazard30']
    pool, rolling_rank, reserve_rank = hard_pool(
        rolling, reserve, currencies, eligible)
    np.testing.assert_array_equal(saved['expert__rolling'], rolling)
    np.testing.assert_array_equal(saved['reserve_score'], reserve)
    np.testing.assert_array_equal(saved['hard_pool'], pool)
    same(saved['hard_pool_rolling_rank'], rolling_rank)
    same(saved['hard_pool_reserve_rank'], reserve_rank)
    assert (rolling_rank[pool] <= .70).all() and (reserve_rank[pool] > .70).all()
    assert not pool[~eligible].any()

    clean, clean_logs = fit_all(
        panel, X, labels, outcomes['forward5'], eligible, pool, cap)
    for name in SPECS:
        same(saved['prediction__' + name], clean[name])
    logged = pd.read_csv(OUT / 'training_log.csv')
    rebuilt = pd.DataFrame(clean_logs)
    assert len(logged) == len(rebuilt) == 17 * len(SPECS)
    for column in ('origin', 'family', 'n_shared_train', 'n_pool_train',
                   'n_query', 'shared_mask_sha256', 'pool_mask_sha256',
                   'last_shared_mature20', 'family_kind', 'target_kind',
                   'n_specialist_train', 'fallback'):
        np.testing.assert_array_equal(logged[column].to_numpy(),
                                      rebuilt[column].to_numpy())
    for origin_text in sorted(logged.origin.unique()):
        origin = dt.date.fromisoformat(origin_text)
        shared = matured_mask(panel, cap, origin)
        train_pool = shared & pool & eligible & np.isfinite(labels).all(axis=1)
        group = logged[logged.origin == origin_text]
        assert (group.n_shared_train == shared.sum()).all()
        assert (group.n_pool_train == train_pool.sum()).all()
        assert (group.last_shared_mature20 == str(max(cap['mature20'][shared]))).all()

    signals, diagnostics = build_policies(
        panel, clean, rolling, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)
        reason = detail['reason']
        assert np.array_equal(signals[candidate], reason > 0)
        assert (detail['primary_rank'][reason == 1] > .70).all()
        pace = reason == 2
        assert (detail['pace_rank'][pace] > .55).all()
        assert (detail['reserve_rank'][pace] > .70).all()
        assert (detail['trailing_rate'][pace] < 1.).all()

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_X, bad_labels = X.copy(), labels.copy()
    bad_future = outcomes['forward5'].copy()
    bad_X[cut:] = np.arange(X.shape[1], dtype=float) * -555.
    finite = np.isfinite(bad_labels[cut:])
    bad_labels[cut:][finite] = 1 - bad_labels[cut:][finite]
    bad_future[cut:] = 999999.
    bad_predictions, _ = fit_all(
        panel, bad_X, bad_labels, bad_future, eligible, pool, cap)
    for name in SPECS:
        same(clean[name][:cut], bad_predictions[name][:cut])
    bad_signals, bad_detail = build_policies(
        panel, bad_predictions, rolling, reserve, eligible, old)
    for name in SPECS:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in SPECS]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert not early.loc[fresh, 'joint_early_pass'].any()
    assert selection['selected'] == AP21_STRICT
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    best = str(later.loc[fresh].sort_values(
        ['min_lift', 'mean_lift'], ascending=False).index[0])
    assert best == BEST_LIFT
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in (BEST_LIFT, BEST_H5)
        for control in (AP23_BEST, AP24_BEST, AP21_STRICT, AP18_CONTROL,
                        AP17_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    strict = []
    for name in fresh:
        row = later.loc[name]
        if (row.min_lift > 2.4 and row.min_rate >= 1 and row.max_rate <= 2
                and row.empty_complete_months == 0):
            strict.append(name)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_control_after_no_fresh_early_pass': selection['selected'],
        'fresh_early_pass_count': 0,
        'best_later_min_lift': best,
        'best_later_min_lift_value': float(later.loc[best, 'min_lift']),
        'best_h5_specialist': BEST_H5,
        'strict_later_successes': strict,
        'specialists_improve_accuracy_but_miss_rate_floor': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'hard_pool_and_both_causal_ranks_exact': True,
        'all_85_specialist_fits_rebuilt': True,
        'all_masks_clipping_and_scores_exact': True,
        'all_policy_states_rebuilt': True,
        'future_feature_label_benefit_corruption_prefix_invariant': True,
        'no_fresh_early_joint_pass_confirmed': True,
        'specialist_accuracy_rate_tradeoff_confirmed': True,
        'known_down_veto': True, 'h1_excluded_from_training_and_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
