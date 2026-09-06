"""Independent AP29 source, state, prefix and uncertainty audit."""
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
from research.after_publication_ap29_effective import (
    AP17_CONTROL,
    AP18_CONTROL,
    AP21_STRICT,
    AP23,
    AP23_BEST,
    AP26,
    AP26_BEST,
    AP27_SELECTED,
    AP27_STRICT,
    BASE,
    CANDIDATE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
    policy_inputs,
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
    for folder in (OUT, BASE, AP26, AP23):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap26, ap23 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    inputs = policy_inputs(old, ap26, ap23)
    for key, value in inputs.items():
        same(saved['input__' + key], value)
    eligible = saved['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(panel, inputs, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in diagnostics.items():
        same(saved[f'{name}__{CANDIDATE}'], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_inputs = {key: value.copy() for key, value in inputs.items()}
    for j, key in enumerate(bad_inputs, start=1):
        bad_inputs[key][cut:] = 999. * j
    bad_signals, bad_detail = build_policies(
        panel, bad_inputs, eligible, old)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for name in diagnostics:
        same(diagnostics[name][:cut], bad_detail[name][:cut])

    common_audit(OUT, [CANDIDATE],
                 [(control, [CANDIDATE]) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (CANDIDATE if passed else AP21_STRICT)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in (AP27_SELECTED, AP27_STRICT, AP26_BEST, AP23_BEST,
                        AP21_STRICT, AP18_CONTROL, AP17_CONTROL)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    row = later.loc[CANDIDATE]
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0)
    reason = saved[f'reason__{CANDIDATE}']
    scope = saved['later'].astype(bool)
    counts = pd.DataFrame({
        'reason': ('none', 'primary', 'pace', 'backstop', 'month'),
        'count': [int((scope & (reason == value)).sum()) for value in range(5)],
    })
    counts.to_csv(OUT / 'reason_counts.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_min_rate': float(row.min_rate),
        'late_strict_success': strict,
        'late_reason_counts': dict(zip(counts.reason, counts['count'])),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'all_four_inputs_and_ranks_exact': True,
        'policy_state_and_reasons_rebuilt': True,
        'future_all_input_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto': True, 'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
