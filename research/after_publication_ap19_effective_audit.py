"""Independent AP19 CatBoost/ExtraTrees refit and causality audit."""
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
from research.after_publication_ap19_effective import (
    AP12,
    AP12_CONTROL,
    AP13,
    AP17_CONTROL,
    AP18_CONTROL,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    SCORE_NAMES,
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
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan

    with np.load(OUT / 'outputs.npz') as result, \
            np.load(BASE / 'outputs.npz') as source, \
            np.load(AP12 / 'outputs.npz') as ap12_source, \
            np.load(AP13 / 'outputs.npz') as ap13_source:
        saved = {key: result[key] for key in result.files}
        old = {key: source[key] for key in source.files}
        ap12 = {key: ap12_source[key] for key in ap12_source.files}
        ap13 = {key: ap13_source[key] for key in ap13_source.files}
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
    base = ap12['prediction__extra_h5']
    prediction, raw, logs = fit_all(panel, X, labels, eligible, cap, base)
    for name in SCORE_NAMES:
        same(saved['prediction__' + name], prediction[name])
    for name, value in raw.items():
        same(saved['raw__' + name], value)
    logged = pd.read_csv(OUT / 'training_log.csv')
    rebuilt = pd.DataFrame(logs)
    assert len(logged) == len(rebuilt) == 17 * len(SCORE_NAMES)
    stable = ['origin', 'family', 'n_shared_train', 'n_eligible_train',
              'n_query', 'shared_mask_sha256', 'train_mask_sha256',
              'last_shared_mature20']
    pd.testing.assert_frame_equal(logged[stable].astype(str),
                                  rebuilt[stable].astype(str))

    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, prediction, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)

    # Corrupt all future model inputs and labels. Refit from scratch and require
    # every score and state before the cutoff to remain unchanged.
    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 1), side='left'))
    X_bad, labels_bad, base_bad = X.copy(), labels.copy(), base.copy()
    X_bad[cut:] = X_bad[cut:] * -23. + 9876.
    labels_bad[cut:] = 1. - labels_bad[cut:]
    base_bad[cut:] = np.nan_to_num(base_bad[cut:], nan=.5)[::-1]
    pred_bad, _, _ = fit_all(panel, X_bad, labels_bad, eligible, cap, base_bad)
    for name in SCORE_NAMES:
        same(prediction[name][:cut], pred_bad[name][:cut])
    signal_bad, detail_bad = build_policies(panel, pred_bad, reserve, eligible, old)
    for name in SCORE_NAMES:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], signal_bad[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], detail_bad[key][field][:cut])

    accuracy = []
    y5 = labels[:, 1]
    for name in SCORE_NAMES:
        score = prediction[name]
        finite = np.isfinite(score)
        assert ((score[finite] >= 0) & (score[finite] <= 1)).all()
        for period in ('early', 'later'):
            mask = saved[period] & eligible & np.isfinite(y5) & finite
            accuracy.append({
                'model': name, 'period': period, 'n': int(mask.sum()),
                'squared_error_h5': float(np.mean((score[mask] - y5[mask]) ** 2)),
                'mean_score': float(score[mask].mean()),
                'event_rate': float(y5[mask].mean()),
            })
    pd.DataFrame(accuracy).to_csv(OUT / 'score_accuracy.csv', index=False)

    fresh = [candidate_key(name) for name in SCORE_NAMES]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    fresh_later = later.loc[fresh]
    best_min = str(fresh_later.sort_values('min_lift', ascending=False).index[0])
    best_mean = str(fresh_later.sort_values('mean_lift', ascending=False).index[0])
    focus = list(dict.fromkeys([selected, best_min, best_mean]))
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in focus
        for control in (AP18_CONTROL, AP17_CONTROL, AP12_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selected,
        'best_later_fresh_min': best_min,
        'best_later_fresh_mean': best_mean,
        'selected_min_unknown_h_lift': float(later.loc[selected, 'min_lift']),
        'best_fresh_min_unknown_h_lift': float(later.loc[best_min, 'min_lift']),
        'best_fresh_mean_unknown_h_lift': float(later.loc[best_mean, 'mean_lift']),
        'strict_target_gt_2_4': bool(later.loc[best_min, 'min_lift'] > 2.4),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'quarterly_origins_checked': 17,
        'fit_log_rows_checked': len(logged),
        'independent_catboost_and_extratrees_refit': True,
        'all_raw_and_combined_scores_rebuilt': True,
        'all_policy_states_rebuilt': True,
        'future_feature_label_and_base_corruption_prefix_invariant': True,
        'all_scores_bounded': True,
        'known_down_veto': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
