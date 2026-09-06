"""Independent audit of AP41's single frequency-floor change."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap3 import weekly_max
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap41_effective import (
    BASE,
    CANDIDATE,
    CONTROLS,
    OUT,
    build_policies,
)


def same(left, right):
    np.testing.assert_allclose(
        left, right, rtol=1e-12, atol=1e-12, equal_nan=True)


def main():
    metadata = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates = panel.date.to_numpy()
    with np.load(OUT / 'outputs.npz') as source:
        saved = {key: source[key] for key in source.files}
    with np.load(BASE / 'outputs.npz') as source:
        old = {key: source[key] for key in source.files}
    signals, diagnostics = build_policies(panel, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
    for key, value in diagnostics.items():
        if value.dtype.kind in 'biu':
            np.testing.assert_array_equal(saved['policy__' + key], value)
        else:
            same(saved['policy__' + key], value)

    cut_day = dt.date(2025, 1, 6)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad = {key: value.copy() for key, value in old.items()}
    future = np.arange(len(dates)) >= cut
    for key in ('input__core_signal', 'input__fallback_signal',
                'input__fallback_quality'):
        bad[key][future] = ~bad[key][future].astype(bool)
    bad['take_probability'][future] = 1. - bad['take_probability'][future]
    bad_signals, bad_diagnostics = build_policies(panel, bad)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])

    common_audit(OUT, [CANDIDATE],
                 [(control, [CANDIDATE]) for control in CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (
        CANDIDATE if passed else AP37_CANDIDATE)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in CONTROLS for block in (20, 50)])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    row = later.loc[CANDIDATE]
    max_week = weekly_max(panel, signals[CANDIDATE], saved['later'])
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0
                  and max_week <= 2)
    leader = old['signal__' + AP37_CANDIDATE].astype(bool)
    scope = saved['later'].astype(bool)
    comparison = pd.DataFrame({
        'change': ('same_selected', 'removed_from_ap37', 'added_vs_ap37'),
        'count': (
            int((scope & signals[CANDIDATE] & leader).sum()),
            int((scope & ~signals[CANDIDATE] & leader).sum()),
            int((scope & signals[CANDIDATE] & ~leader).sum()),
        )})
    comparison.to_csv(OUT / 'decision_changes.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'late_min_lift': float(row.min_lift),
        'late_mean_lift': float(row.mean_lift),
        'late_min_rate': float(row.min_rate),
        'late_max_weekly_signals': int(max_week),
        'late_strict_success': strict,
        'late_core_veto_count': int((scope & diagnostics['core_veto']).sum()),
        'late_decision_changes_vs_ap37': dict(zip(
            comparison.change, comparison['count'])),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'ap40_predictions_inherited_exact': True,
        'rate110_router_state_and_cap_exact': True,
        'future_input_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto_inherited_and_checked': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
