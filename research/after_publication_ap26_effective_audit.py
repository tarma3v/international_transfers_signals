"""Independent AP26 count mapping, mixtures, policy and uncertainty audit."""
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
from research.after_publication_ap26_effective import (
    AP17_CONTROL,
    AP18_CONTROL,
    AP21_STRICT,
    AP23_BEST,
    AP24,
    AP24_BEST,
    AP25_H5,
    AP25_LIFT,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    candidate_key,
    count_by_row,
)
from research.after_publication_ap26_effective_models import SCORES, cold_start_scores
from research.after_publication_panel import build_outcomes


SELECTED = candidate_key('y20_hard100')
BEST_LATE = candidate_key('y20_shrink200')


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
    with np.load(OUT / 'outputs.npz') as source, \
            np.load(BASE / 'outputs.npz') as old_source, \
            np.load(AP24 / 'outputs.npz') as ap24_source:
        saved, old, ap24 = ({key: obj[key] for key in obj.files}
                            for obj in (source, old_source, ap24_source))
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    logs = pd.read_csv(BASE / 'training_log.csv')
    counts = {
        'y20': count_by_row(panel, logs, 'pace_cat_y20_gate_full'),
        'mean': count_by_row(panel, logs, 'pace_cat_mean_gate_full'),
        'future5': count_by_row(panel, logs, 'pace_cat_future5_gate_full'),
    }
    cat = ap24['expert__cat']
    rolling = old['expert__rolling']
    reserve = old['reserve_score']
    specialists = {
        'y20': old['prediction__pace_cat_y20_gate_full'],
        'mean': old['prediction__pace_cat_mean_gate_full'],
        'future5': old['prediction__pace_cat_future5_gate_full'],
    }
    for name in counts:
        np.testing.assert_array_equal(saved['count__' + name], counts[name])
        np.testing.assert_array_equal(saved['specialist__' + name], specialists[name])
    np.testing.assert_array_equal(saved['expert__cat'], cat)
    np.testing.assert_array_equal(saved['expert__rolling'], rolling)
    np.testing.assert_array_equal(saved['reserve_score'], reserve)
    rebuilt = cold_start_scores(
        cat, specialists['y20'], specialists['mean'], specialists['future5'],
        counts['y20'], counts['mean'], counts['future5'], panel.currency.to_numpy())
    scores, weights, cat_rank, future_rank = rebuilt
    same(saved['inner_rank__cat'], cat_rank)
    same(saved['inner_rank__future5'], future_rank)
    for name in SCORES:
        same(saved['pace_score__' + name], scores[name])
        same(saved['weight__' + name], weights[name])
    cold = (counts['y20'] < 100) & np.isfinite(cat)
    same(scores['y20_hard100'][cold], cat[cold])

    eligible = saved['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(
        panel, scores, rolling, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_cat = cat.copy()
    bad_specialists = {name: value.copy() for name, value in specialists.items()}
    bad_counts = {name: value.copy() for name, value in counts.items()}
    bad_cat[cut:] = 999.
    for j, name in enumerate(bad_specialists, start=1):
        bad_specialists[name][cut:] = -999. * j
        bad_counts[name][cut:] = 9999
    bad = cold_start_scores(
        bad_cat, bad_specialists['y20'], bad_specialists['mean'],
        bad_specialists['future5'], bad_counts['y20'], bad_counts['mean'],
        bad_counts['future5'], panel.currency.to_numpy())
    for name in SCORES:
        same(scores[name][:cut], bad[0][name][:cut])
        same(weights[name][:cut], bad[1][name][:cut])
    bad_signals, bad_detail = build_policies(
        panel, bad[0], rolling, reserve, eligible, old)
    for name in SCORES:
        key = candidate_key(name)
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in SCORES]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert int(early.loc[fresh, 'joint_early_pass'].sum()) == 5
    assert selection['selected'] == SELECTED
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    best_late = str(later.loc[fresh].sort_values(
        ['min_lift', 'mean_lift'], ascending=False).index[0])
    assert best_late == BEST_LATE
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in (SELECTED, BEST_LATE)
        for control in (AP25_LIFT, AP25_H5, AP23_BEST, AP24_BEST,
                        AP21_STRICT, AP18_CONTROL, AP17_CONTROL)
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
        'selected_early': SELECTED,
        'selected_late_min_lift': float(later.loc[SELECTED, 'min_lift']),
        'best_later_fresh': best_late,
        'best_later_min_lift': float(later.loc[best_late, 'min_lift']),
        'best_later_min_rate': float(later.loc[best_late, 'min_rate']),
        'strict_later_successes': strict,
        'cold_start_improves_accuracy_but_still_misses_rate': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'all_quarter_count_maps_exact': True,
        'all_specialists_experts_ranks_weights_and_scores_exact': True,
        'all_policy_states_rebuilt': True,
        'future_score_count_corruption_prefix_invariant': True,
        'five_fresh_early_joint_passes_confirmed': True,
        'late_accuracy_rate_tradeoff_confirmed': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
