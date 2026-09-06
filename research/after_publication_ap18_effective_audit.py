"""Independent AP18 refit, causality, policy and uncertainty audit."""
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
from research.after_publication_ap18_effective import (
    AP12,
    AP12_CONTROL,
    AP13,
    AP17_CONTROL,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    SCORE_NAMES,
    SIMPLE,
    build_policies,
    candidate_key,
    fit_all,
)
from research.after_publication_panel import build_outcomes


def _same_with_nan(left, right):
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
    y = ap13['conditional_labels'][:, 1]
    eligible = saved['eligible_next'].astype(bool)
    base = ap12['prediction__extra_h5']
    recent = ap13['prediction__extra_roll2']
    prediction, corrections, compact, logs = fit_all(
        panel, X, y, eligible, cap, base, recent)
    np.testing.assert_array_equal(saved['compact_feature_indices'], compact)
    for name in SCORE_NAMES:
        _same_with_nan(saved['prediction__' + name], prediction[name])
    for name, value in corrections.items():
        _same_with_nan(saved['correction__' + name], value)

    logged = pd.read_csv(OUT / 'training_log.csv')
    rebuilt = pd.DataFrame(logs)
    assert len(logged) == len(rebuilt) == 17 * len(SCORE_NAMES)
    stable_columns = ['origin', 'family', 'n_shared_train', 'n_eligible_train',
                      'n_query', 'shared_mask_sha256', 'train_mask_sha256',
                      'last_shared_mature20']
    pd.testing.assert_frame_equal(logged[stable_columns].astype(str),
                                  rebuilt[stable_columns].astype(str))

    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, prediction, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            _same_with_nan(saved[f'{name}__{candidate}'], value)

    # Physically corrupt every future input channel and refit. Earlier quarterly
    # predictions must remain bit-for-bit/allclose unchanged.
    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 1), side='left'))
    X_bad, y_bad = X.copy(), y.copy()
    base_bad, recent_bad = base.copy(), recent.copy()
    X_bad[cut:] = X_bad[cut:] * -17. + 12345.
    y_bad[cut:] = 1. - y_bad[cut:]
    base_bad[cut:] = np.nan_to_num(base_bad[cut:], nan=.5)[::-1]
    recent_bad[cut:] = np.nan_to_num(recent_bad[cut:], nan=.5)[::-1]
    pred_bad, _, compact_bad, _ = fit_all(
        panel, X_bad, y_bad, eligible, cap, base_bad, recent_bad)
    np.testing.assert_array_equal(compact_bad, compact)
    for name in SCORE_NAMES:
        _same_with_nan(prediction[name][:cut], pred_bad[name][:cut])
    signal_bad, detail_bad = build_policies(
        panel, pred_bad, reserve, eligible, old)
    for name in SCORE_NAMES:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], signal_bad[key][:cut])
        for field in diagnostics[key]:
            _same_with_nan(diagnostics[key][field][:cut],
                           detail_bad[key][field][:cut])

    accuracy = []
    for name in SCORE_NAMES:
        probability = prediction[name]
        finite = np.isfinite(probability)
        assert ((probability[finite] >= 0) & (probability[finite] <= 1)).all()
        for period in ('early', 'later'):
            mask = saved[period] & eligible & np.isfinite(y) & finite
            accuracy.append({
                'model': name, 'period': period, 'n': int(mask.sum()),
                'brier_h5': float(np.mean((probability[mask] - y[mask]) ** 2)),
                'mean_probability': float(probability[mask].mean()),
                'event_rate': float(y[mask].mean()),
            })
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)

    fresh = [candidate_key(name) for name in SCORE_NAMES]
    controls = list(FROZEN_CONTROLS)
    common_audit(OUT, fresh, [(control, fresh) for control in controls])
    selection = json.loads((OUT / 'selection.json').read_text())
    selected = selection['selected']
    late_h5 = pd.read_csv(OUT / 'retrospective_all_horizons.csv').query('h == 5')
    best_late = str(late_h5[late_h5.candidate.isin(fresh)].sort_values(
        'adjusted_lift', ascending=False).iloc[0].candidate)
    focus = list(dict.fromkeys([selected, best_late]))
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in focus
        for control in (AP17_CONTROL, AP12_CONTROL, *controls[2:])
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)

    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    selected_row = later.loc[selected]
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected': selected, 'best_late_fresh_h5': best_late,
        'selected_min_unknown_h_lift': float(selected_row.min_lift),
        'selected_mean_unknown_h_lift': float(selected_row.mean_lift),
        'selected_strict_target_gt_2_4': bool(selected_row.min_lift > 2.4),
        'selected_improves_ap17_point_min': bool(
            selected_row.min_lift > later.loc[AP17_CONTROL, 'min_lift']),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'quarterly_origins_checked': 17,
        'fit_log_rows_checked': len(logged),
        'independent_deterministic_refit': True,
        'all_predictions_and_corrections_rebuilt': True,
        'all_policy_states_rebuilt': True,
        'future_feature_label_and_expert_corruption_prefix_invariant': True,
        'all_predictions_bounded': True,
        'known_down_veto': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
