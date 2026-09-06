"""Independent audit of the AP40 causal weekly optimal-stopping model."""
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
from research.after_publication_ap37_effective_models import EXPERTS
from research.after_publication_ap40_effective import (
    AP37,
    BASE,
    CANDIDATE,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
)
from research.after_publication_ap40_effective_models import (
    build_stopping_features,
    build_weekly_take_target,
    fit_quarterly_logit,
)
from research.after_publication_panel import build_outcomes


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
    currencies = panel.currency.to_numpy()
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan

    loaded = []
    for folder in (OUT, BASE, AP37):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap37 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(
                saved[kind + str(h)], outcomes[kind + str(h)])

    eligible = old['eligible_next'].astype(bool)
    core = old['input__core_signal'].astype(bool)
    fallback = old['input__fallback_signal'].astype(bool)
    opportunity = eligible & (core | fallback)
    ranks = {name: old['rank__' + name] for name in EXPERTS}
    leader = old['signal__' + AP37_CANDIDATE].astype(bool)
    quality = ap37['quality_gate'].astype(bool)
    features = build_stopping_features(
        panel, ranks, core, fallback,
        ap37['local_stratum_precision'], ap37['overall_precision'],
        ap37['policy__trailing_rate'], ap37['policy__days_since_signal'],
        leader)
    target = build_weekly_take_target(
        dates, currencies, opportunity, outcomes, cap['mature20'])
    take_y, take_maturity, utility, future_count, wait_gain = target
    prediction, logs = fit_quarterly_logit(
        features, take_y, take_maturity, dates, opportunity)
    signals, diagnostics = build_policies(
        panel, core, fallback, prediction, quality, eligible, old)

    same(saved['features'], features)
    np.testing.assert_array_equal(saved['opportunity'], opportunity)
    same(saved['take_target'], take_y)
    np.testing.assert_array_equal(
        saved['take_target_maturity'],
        np.asarray([str(value) if value is not None else ''
                    for value in take_maturity]))
    same(saved['utility'], utility)
    np.testing.assert_array_equal(
        saved['future_opportunity_count'], future_count)
    same(saved['wait_gain'], wait_gain)
    same(saved['take_probability'], prediction)
    saved_log = pd.read_csv(OUT / 'training_log.csv').fillna('')
    rebuilt_log = pd.DataFrame(logs).fillna('')
    pd.testing.assert_frame_equal(saved_log, rebuilt_log, check_dtype=False)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in diagnostics.items():
        if value.dtype.kind in 'biu':
            np.testing.assert_array_equal(saved['policy__' + name], value)
        else:
            same(saved['policy__' + name], value)

    # Monday boundary means no previous ISO-week target can reference a
    # corrupted opportunity. Everything from this date onward is adversarial.
    cut_day = dt.date(2025, 1, 6)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    future = np.arange(len(dates)) >= cut
    bad_ranks = {name: value.copy() for name, value in ranks.items()}
    for value in bad_ranks.values():
        value[future] = 1. - value[future]
    bad_outcomes = {key: value.copy() for key, value in outcomes.items()}
    for h in (3, 5, 10, 20):
        key = 'y' + str(h)
        finite = future & np.isfinite(bad_outcomes[key])
        bad_outcomes[key][finite] = 1. - bad_outcomes[key][finite]
    bad_core = core.copy()
    bad_fallback = fallback.copy()
    bad_leader = leader.copy()
    bad_quality = quality.copy()
    bad_local = ap37['local_stratum_precision'].copy()
    bad_overall = ap37['overall_precision'].copy()
    bad_rate = ap37['policy__trailing_rate'].copy()
    bad_gap = ap37['policy__days_since_signal'].copy()
    for value in (bad_core, bad_fallback, bad_leader, bad_quality):
        value[future] = ~value[future]
    for value in (bad_local, bad_overall, bad_rate, bad_gap):
        value[future] = np.where(np.isfinite(value[future]),
                                 value[future] + 17., -17.)
    bad_opportunity = eligible & (bad_core | bad_fallback)
    bad_features = build_stopping_features(
        panel, bad_ranks, bad_core, bad_fallback,
        bad_local, bad_overall, bad_rate, bad_gap, bad_leader)
    bad_target = build_weekly_take_target(
        dates, currencies, bad_opportunity, bad_outcomes, cap['mature20'])
    bad_y, bad_maturity = bad_target[:2]
    bad_prediction, _ = fit_quarterly_logit(
        bad_features, bad_y, bad_maturity, dates, bad_opportunity)
    bad_signals, bad_diagnostics = build_policies(
        panel, bad_core, bad_fallback, bad_prediction, bad_quality,
        eligible, old)
    same(features[:cut], bad_features[:cut])
    same(take_y[:cut], bad_y[:cut])
    np.testing.assert_array_equal(take_maturity[:cut], bad_maturity[:cut])
    same(prediction[:cut], bad_prediction[:cut])
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for name in diagnostics:
        same(diagnostics[name][:cut], bad_diagnostics[name][:cut])

    common_audit(
        OUT, [CANDIDATE],
        [(control, [CANDIDATE]) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (
        CANDIDATE if passed else AP37_CANDIDATE)
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
    reason_names = (
        'none', 'model_or_unavailable_core', 'precision_late_week_fallback',
        'silence_fallback', 'friday_core_guard', 'rate_deficit_core_guard',
        'silence_core_guard', 'month24_core_guard')
    scope = saved['later'].astype(bool)
    reason = saved['policy__reason']
    counts = pd.DataFrame({
        'reason': reason_names,
        'count': [int((scope & (reason == value)).sum())
                  for value in range(len(reason_names))],
    })
    counts.to_csv(OUT / 'reason_counts.csv', index=False)
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
        'late_max_rate': float(row.max_rate),
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
        'weekly_target_and_latest_label_maturity_exact': True,
        'twenty_current_only_features_exact': True,
        'quarterly_logistic_predictions_exact': True,
        'causal_router_state_and_cap_exact': True,
        'future_feature_label_and_policy_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto_inherited_and_checked': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
    }, indent=2))


if __name__ == '__main__':
    main()
