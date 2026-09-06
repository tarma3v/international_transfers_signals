"""Independent AP20 expert geometry, state and uncertainty audit."""
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
from research.after_publication_ap20_effective import (
    AP12,
    AP12_CONTROL,
    AP17_CONTROL,
    AP18,
    AP18_CONTROL,
    AP19_CAT,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    SCORE_NAMES,
    build_policies,
    candidate_key,
)
from research.after_publication_ap20_effective_models import expert_scores
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
            np.load(AP18 / 'outputs.npz') as ap18_source, \
            np.load(AP12 / 'outputs.npz') as ap12_source:
        saved = {key: result[key] for key in result.files}
        old = {key: source[key] for key in source.files}
        ap18 = {key: ap18_source[key] for key in ap18_source.files}
        ap12 = {key: ap12_source[key] for key in ap12_source.files}
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    left = ap18['prediction__full_recent50']
    right = old['prediction__cat_mean_utility']
    np.testing.assert_array_equal(saved['expert__ap18'], left)
    np.testing.assert_array_equal(saved['expert__cat_utility'], right)
    scores, left_rank, right_rank = expert_scores(
        left, right, panel.currency.to_numpy())
    same(saved['rank__ap18'], left_rank)
    same(saved['rank__cat_utility'], right_rank)
    for name, value in scores.items():
        same(saved['score__' + name], value)
    eligible = saved['eligible_next'].astype(bool)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, scores, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)

    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 1), side='left'))
    left_bad, right_bad = left.copy(), right.copy()
    left_bad[cut:] = 9999
    right_bad[cut:] = -9999
    bad_scores, _, _ = expert_scores(left_bad, right_bad, panel.currency.to_numpy())
    bad_signals, bad_detail = build_policies(
        panel, bad_scores, reserve, eligible, old)
    for name in SCORE_NAMES:
        key = candidate_key(name)
        same(scores[name][:cut], bad_scores[name][:cut])
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in SCORE_NAMES]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    best_min = str(later.loc[fresh].sort_values('min_lift', ascending=False).index[0])
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for candidate in dict.fromkeys([selected, best_min])
        for control in (AP19_CAT, AP18_CONTROL, AP17_CONTROL, AP12_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selected, 'best_later_fresh_min': best_min,
        'selected_min_unknown_h_lift': float(later.loc[selected, 'min_lift']),
        'best_fresh_min_unknown_h_lift': float(later.loc[best_min, 'min_lift']),
        'strict_target_gt_2_4': bool(later.loc[best_min, 'min_lift'] > 2.4),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'both_frozen_experts_exact': True, 'all_eight_scores_rebuilt': True,
        'all_policy_states_rebuilt': True,
        'future_expert_corruption_prefix_invariant': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
