"""Independent AP22 consensus ranks, state and negative-result audit."""
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
from research.after_publication_ap22_effective import (
    AP12,
    AP12_CONTROL,
    AP13,
    AP17_CONTROL,
    AP18_CONTROL,
    AP19,
    AP21_STRICT,
    BASE,
    FROZEN_CONTROLS,
    OUT,
    PAIRINGS,
    build_policies,
    candidate_key,
)
from research.after_publication_ap22_effective_models import consensus_pairings
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
    folders = (OUT, BASE, AP12, AP13, AP19)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap12, ap13, ap19 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    experts = {
        'rolling': ap13['prediction__extra_roll2'],
        'local': ap13['prediction__local_extra'],
        'cat': ap19['prediction__cat_mean_utility'],
        'ap12': ap12['prediction__extra_h5'],
    }
    for name, value in experts.items():
        np.testing.assert_array_equal(saved['expert__' + name], value)
    pairs, rolling_rank, local_rank = consensus_pairings(
        experts['rolling'], experts['local'], experts['cat'], experts['ap12'],
        panel.currency.to_numpy())
    same(saved['rank__rolling'], rolling_rank)
    same(saved['rank__local'], local_rank)
    for name, (primary, pace) in pairs.items():
        same(saved['primary_score__' + name], primary)
        same(saved['pace_score__' + name], pace)
    eligible = saved['eligible_next'].astype(bool)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, pairs, reserve, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            same(saved[f'{name}__{candidate}'], value)

    dates = panel.date.to_numpy()
    cut = int(np.searchsorted(dates, dt.date(2025, 1, 1), side='left'))
    bad = {name: value.copy() for name, value in experts.items()}
    for j, value in enumerate(bad.values(), start=1):
        value[cut:] = j * 1111 * (-1 if j % 2 else 1)
    bad_pairs, bad_rr, bad_lr = consensus_pairings(
        bad['rolling'], bad['local'], bad['cat'], bad['ap12'],
        panel.currency.to_numpy())
    same(rolling_rank[:cut], bad_rr[:cut])
    same(local_rank[:cut], bad_lr[:cut])
    bad_signals, bad_detail = build_policies(
        panel, bad_pairs, reserve, eligible, old)
    for name in PAIRINGS:
        key = candidate_key(name)
        for original, changed in zip(pairs[name], bad_pairs[name]):
            same(original[:cut], changed[:cut])
        np.testing.assert_array_equal(signals[key][:cut], bad_signals[key][:cut])
        for field in diagnostics[key]:
            same(diagnostics[key][field][:cut], bad_detail[key][field][:cut])

    fresh = [candidate_key(name) for name in PAIRINGS]
    common_audit(OUT, fresh, [(control, fresh) for control in FROZEN_CONTROLS])
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    assert not early.loc[fresh, 'joint_early_pass'].any()
    assert selected == AP21_STRICT
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    best_fresh = str(later.loc[fresh].sort_values('min_lift', ascending=False).index[0])
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, best_fresh, control, block)
        for control in (AP21_STRICT, AP18_CONTROL, AP17_CONTROL, AP12_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_control_after_no_fresh_early_pass': selected,
        'fresh_early_pass_count': 0,
        'best_later_fresh': best_fresh,
        'best_later_fresh_min_lift': float(later.loc[best_fresh, 'min_lift']),
        'negative_result': True,
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'events_and_targets_rebuilt': len(panel),
        'all_four_experts_exact': True, 'both_causal_ranks_exact': True,
        'all_eight_pairings_and_states_rebuilt': True,
        'future_all_expert_corruption_prefix_invariant': True,
        'no_fresh_early_joint_pass_confirmed': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
