"""Independent AP38 dual-precision, guarded-router and prefix audit."""
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
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap37_effective_models import (
    EXPERTS,
    mature_pool_support_precision,
    mature_support_precision,
)
from research.after_publication_ap38_effective import (
    BASE,
    CANDIDATE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
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
    currencies = panel.currency.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    loaded = []
    for folder in (OUT, BASE):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    eligible = saved['eligible_next'].astype(bool)
    core = old['input__core_signal'].astype(bool)
    fallback = old['input__fallback_signal'].astype(bool)
    ranks = {name: old['rank__' + name] for name in EXPERTS}
    np.testing.assert_array_equal(saved['input__core_signal'], core)
    np.testing.assert_array_equal(saved['input__fallback_signal'], fallback)
    for name in EXPERTS:
        same(saved['rank__' + name], ranks[name])
    fallback_precision = mature_support_precision(
        ranks, outcomes, cap['mature20'], dates, currencies, eligible, core)
    core_precision = mature_pool_support_precision(
        ranks, outcomes, cap['mature20'], dates, currencies, eligible, core)
    precision_keys = ('support_stratum', 'quality_gate', 'overall_precision',
                      'global_stratum_precision', 'local_stratum_precision',
                      'overall_count', 'global_stratum_count',
                      'local_stratum_count')
    for prefix, values in (('fallback', fallback_precision),
                           ('core', core_precision)):
        for key, value in zip(precision_keys, values):
            saved_value = saved[prefix + '__' + key]
            if value.dtype.kind in 'biu':
                np.testing.assert_array_equal(saved_value, value)
            else:
                same(saved_value, value)
    signals, diagnostics = build_policies(
        panel, core, fallback, core_precision[1], fallback_precision[1],
        eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in diagnostics.items():
        if value.dtype.kind in 'biu':
            np.testing.assert_array_equal(saved['policy__' + name], value)
        else:
            same(saved['policy__' + name], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_ranks = {name: value.copy() for name, value in ranks.items()}
    for value in bad_ranks.values():
        value[cut:] = 1. - value[cut:]
    bad_outcomes = {key: value.copy() for key, value in outcomes.items()}
    immature_at_cut = np.asarray(cap['mature20'], dtype=object) >= (
        cut_day - dt.timedelta(days=2))
    for h in (3, 5, 10, 20):
        key = 'y' + str(h)
        finite = np.isfinite(bad_outcomes[key])
        change = finite & (np.arange(len(dates)) >= cut | immature_at_cut)
        bad_outcomes[key][change] = 1. - bad_outcomes[key][change]
    bad_core = core.copy()
    bad_fallback = fallback.copy()
    bad_core[cut:] = ~bad_core[cut:]
    bad_fallback[cut:] = ~bad_fallback[cut:]
    bad_fallback_precision = mature_support_precision(
        bad_ranks, bad_outcomes, cap['mature20'], dates, currencies,
        eligible, bad_core)
    bad_core_precision = mature_pool_support_precision(
        bad_ranks, bad_outcomes, cap['mature20'], dates, currencies,
        eligible, bad_core)
    bad_signals, bad_detail = build_policies(
        panel, bad_core, bad_fallback, bad_core_precision[1],
        bad_fallback_precision[1], eligible, old)
    for actual_values, bad_values in ((fallback_precision, bad_fallback_precision),
                                      (core_precision, bad_core_precision)):
        for actual, corrupted in zip(actual_values, bad_values):
            if actual.dtype.kind in 'biu':
                np.testing.assert_array_equal(actual[:cut], corrupted[:cut])
            else:
                same(actual[:cut], corrupted[:cut])
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for name in diagnostics:
        same(diagnostics[name][:cut], bad_detail[name][:cut])

    common_audit(OUT, [CANDIDATE],
                 [(control, [CANDIDATE]) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (CANDIDATE if passed else AP37_CANDIDATE)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in FROZEN_CONTROLS
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    row = later.loc[CANDIDATE]
    max_week = weekly_max(panel, signals[CANDIDATE], saved['later'])
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0
                  and max_week <= 2)
    reason = saved['policy__reason']
    scope = saved['later'].astype(bool)
    names = ('none', 'quality_core', 'precision_late_week', 'silence10',
             'rate_guard_core')
    counts = pd.DataFrame({
        'reason': names,
        'count': [int((scope & (reason == value)).sum())
                  for value in range(len(names))],
    })
    counts.to_csv(OUT / 'reason_counts.csv', index=False)
    leader = old['signal__' + AP37_CANDIDATE].astype(bool)
    comparison = pd.DataFrame({
        'change': ('same_selected', 'removed_from_ap37', 'added_vs_ap37'),
        'count': (
            int((scope & signals[CANDIDATE] & leader).sum()),
            int((scope & ~signals[CANDIDATE] & leader).sum()),
            int((scope & signals[CANDIDATE] & ~leader).sum()),
        ),
    })
    comparison.to_csv(OUT / 'decision_changes.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_mean_lift': float(row.mean_lift),
        'late_min_rate': float(row.min_rate),
        'late_max_weekly_signals': int(max_week),
        'late_strict_success': strict,
        'late_core_veto_count': int((scope & diagnostics['core_veto']).sum()),
        'late_reason_counts': dict(zip(counts.reason, counts['count'])),
        'late_decision_changes_vs_ap37': dict(zip(
            comparison.change, comparison['count'])),
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'three_frozen_expert_ranks_exact': True,
        'both_mature_precision_pools_counts_and_shrinkage_exact': True,
        'guarded_router_veto_rate_silence_reason_and_cap_exact': True,
        'future_expert_label_and_source_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto_inherited_and_checked': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
