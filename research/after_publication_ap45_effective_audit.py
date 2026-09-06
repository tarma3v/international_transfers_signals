"""Independent rebuild and causality audits for AP42--AP45."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap3 import weekly_max
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap37_effective import CANDIDATE as AP37_CANDIDATE
from research.after_publication_ap42_effective import (
    AP38 as AP42_AP38,
    BASE as AP42_BASE,
    CANDIDATE as AP42_CANDIDATE,
    CONTROLS as AP42_CONTROLS,
    OUT as AP42_OUT,
    build_policies as build_ap42,
)
from research.after_publication_ap43_effective import (
    AP23 as AP43_AP23,
    AP38 as AP43_AP38,
    BASE as AP43_BASE,
    CANDIDATE as AP43_CANDIDATE,
    CONTROLS as AP43_CONTROLS,
    OUT as AP43_OUT,
    build_policies as build_ap43,
)
from research.after_publication_ap44_effective import (
    AP37 as AP44_AP37,
    BASE as AP44_BASE,
    CANDIDATE as AP44_CANDIDATE,
    CONTROLS as AP44_CONTROLS,
    OUT as AP44_OUT,
    build_policies as build_ap44,
)
from research.after_publication_ap44_effective_models import (
    augment_features,
    fit_quarterly_meta_cat,
)
from research.after_publication_ap45_effective import (
    AP38 as AP45_AP38,
    BASE as AP45_BASE,
    CANDIDATE as AP45_CANDIDATE,
    CONTROLS as AP45_CONTROLS,
    OUT as AP45_OUT,
    build_policies as build_ap45,
)
from research.after_publication_panel import build_outcomes


def loadz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def same(left, right):
    np.testing.assert_allclose(
        left, right, rtol=1e-12, atol=1e-12, equal_nan=True)


def panel_at(folder):
    panel = pd.read_csv(folder / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    return panel


def verify_sources(folder):
    metadata = json.loads((folder / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path


def exact_saved(saved, signals, diagnostics):
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
    for key, value in diagnostics.items():
        if value.dtype.kind in 'biu':
            np.testing.assert_array_equal(saved['policy__' + key], value)
        else:
            same(saved['policy__' + key], value)


def finish(folder, candidate, controls, panel, saved, signals, diagnostics,
           extra_checks):
    common_audit(folder, [candidate],
                 [(control, [candidate]) for control in controls])
    selection = json.loads((folder / 'selection.json').read_text())
    early = pd.read_csv(folder / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[candidate, 'joint_early_pass'])
    assert selection['selected'] == (
        candidate if passed else AP37_CANDIDATE)
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, candidate, control, block)
        for control in controls for block in (20, 50)])
    sensitivity.to_csv(folder / 'selected_block_sensitivity.csv', index=False)
    later = pd.read_csv(folder / 'retrospective_summary.csv').set_index('candidate')
    row = later.loc[candidate]
    max_week = weekly_max(panel, signals[candidate], saved['later'])
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0
                  and max_week <= 2)
    scope = saved['later'].astype(bool)
    leader = saved['signal__' + AP37_CANDIDATE].astype(bool)
    changes = pd.DataFrame({
        'change': ('same_selected', 'removed_from_ap37', 'added_vs_ap37'),
        'count': (
            int((scope & signals[candidate] & leader).sum()),
            int((scope & ~signals[candidate] & leader).sum()),
            int((scope & signals[candidate] & ~leader).sum()),
        )})
    changes.to_csv(folder / 'decision_changes.csv', index=False)
    (folder / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_mean_lift': float(row.mean_lift),
        'late_min_rate': float(row.min_rate),
        'late_max_rate': float(row.max_rate),
        'late_max_weekly_signals': int(max_week),
        'late_strict_success': strict,
        'late_core_veto_count': int(
            (scope & diagnostics['core_veto']).sum()),
        'late_decision_changes_vs_ap37': dict(zip(
            changes.change, changes['count'])),
    }, indent=2))
    checks = {
        'source_hashes_verified': True,
        'policy_state_and_cap_exact': True,
        'future_input_corruption_prefix_invariant': True,
        'early_selection_rebuilt': True,
        'known_down_veto_inherited_and_checked': True,
        'h1_excluded_from_selection': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
        **extra_checks,
    }
    (folder / 'audit_checks.json').write_text(json.dumps(checks, indent=2))


def audit_ap42(cut, future):
    verify_sources(AP42_OUT)
    panel = panel_at(AP42_OUT)
    saved = loadz(AP42_OUT / 'outputs.npz')
    old = loadz(AP42_BASE / 'outputs.npz')
    ap38 = loadz(AP42_AP38 / 'outputs.npz')
    signals, diagnostics = build_ap42(panel, old, ap38)
    exact_saved(saved, signals, diagnostics)
    bad_old = {key: value.copy() for key, value in old.items()}
    bad_ap38 = {key: value.copy() for key, value in ap38.items()}
    for key in ('input__core_signal', 'input__fallback_signal'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    bad_ap38['core__quality_gate'][future] = ~bad_ap38[
        'core__quality_gate'][future].astype(bool)
    bad_signals, bad_diagnostics = build_ap42(panel, bad_old, bad_ap38)
    np.testing.assert_array_equal(
        signals[AP42_CANDIDATE][:cut], bad_signals[AP42_CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])
    finish(AP42_OUT, AP42_CANDIDATE, AP42_CONTROLS, panel, saved,
           signals, diagnostics, {'ap38_quality_and_ap23_binary_exact': True})


def audit_ap43(cut, future):
    verify_sources(AP43_OUT)
    panel = panel_at(AP43_OUT)
    saved = loadz(AP43_OUT / 'outputs.npz')
    old = loadz(AP43_BASE / 'outputs.npz')
    ap23 = loadz(AP43_AP23 / 'outputs.npz')
    ap38 = loadz(AP43_AP38 / 'outputs.npz')
    result = build_ap43(panel, old, ap23, ap38)
    signals, diagnostics = result[:2]
    exact_saved(saved, signals, diagnostics)
    bad_old = {key: value.copy() for key, value in old.items()}
    bad_ap23 = {key: value.copy() for key, value in ap23.items()}
    bad_ap38 = {key: value.copy() for key, value in ap38.items()}
    bad_old['input__core_signal'][future] = ~bad_old[
        'input__core_signal'][future].astype(bool)
    for key in [value for value in ap23 if value.startswith(
            ('pace_rank__', 'reserve_rank__'))]:
        bad_ap23[key][future] = 1. - bad_ap23[key][future]
    bad_ap38['core__quality_gate'][future] = ~bad_ap38[
        'core__quality_gate'][future].astype(bool)
    bad_signals, bad_diagnostics = build_ap43(
        panel, bad_old, bad_ap23, bad_ap38)[:2]
    np.testing.assert_array_equal(
        signals[AP43_CANDIDATE][:cut], bad_signals[AP43_CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])
    finish(AP43_OUT, AP43_CANDIDATE, AP43_CONTROLS, panel, saved,
           signals, diagnostics,
           {'ap38_quality_and_ap23_continuous_ranks_exact': True})


def audit_ap44(cut, future):
    verify_sources(AP44_OUT)
    panel = panel_at(AP44_OUT)
    dates = panel.date.to_numpy()
    saved = loadz(AP44_OUT / 'outputs.npz')
    old = loadz(AP44_BASE / 'outputs.npz')
    series = load(DATA)
    cap = build_outcomes(series, panel, 'publication')
    features = augment_features(old['features'], dates, panel.currency.to_numpy())
    probability, logs, importance = fit_quarterly_meta_cat(
        features, old['y20'], cap['mature20'], dates,
        old['eligible_next'].astype(bool))
    same(saved['meta_features'], features)
    same(saved['meta_y20_probability'], probability)
    same(saved['meta_feature_importance'], importance)
    pd.testing.assert_frame_equal(
        pd.read_csv(AP44_OUT / 'training_log.csv').fillna(''),
        pd.DataFrame(logs).fillna(''), check_dtype=False)
    signals, diagnostics = build_ap44(panel, old, probability)
    exact_saved(saved, signals, diagnostics)

    bad_features = features.copy()
    bad_features[future] = np.where(
        np.isfinite(bad_features[future]), bad_features[future] + 17., -17.)
    bad_target = old['y20'].copy()
    finite = future & np.isfinite(bad_target)
    bad_target[finite] = 1. - bad_target[finite]
    bad_eligible = old['eligible_next'].astype(bool).copy()
    bad_eligible[future] = ~bad_eligible[future]
    bad_probability = fit_quarterly_meta_cat(
        bad_features, bad_target, cap['mature20'], dates, bad_eligible)[0]
    same(probability[:cut], bad_probability[:cut])
    bad_old = {key: value.copy() for key, value in old.items()}
    for key in ('input__core_signal', 'input__fallback_signal',
                'input__fallback_quality'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    bad_signals, bad_diagnostics = build_ap44(
        panel, bad_old, bad_probability)
    np.testing.assert_array_equal(
        signals[AP44_CANDIDATE][:cut], bad_signals[AP44_CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])
    finish(AP44_OUT, AP44_CANDIDATE, AP44_CONTROLS, panel, saved,
           signals, diagnostics, {
               'augmented_features_exact': True,
               'seventeen_quarterly_meta_cat_fits_exact': True,
               'mature_y20_targets_and_importances_exact': True,
           })


def audit_ap45(cut, future):
    verify_sources(AP45_OUT)
    panel = panel_at(AP45_OUT)
    saved = loadz(AP45_OUT / 'outputs.npz')
    old = loadz(AP45_BASE / 'outputs.npz')
    ap38 = loadz(AP45_AP38 / 'outputs.npz')
    signals, diagnostics = build_ap45(panel, old, ap38)
    exact_saved(saved, signals, diagnostics)
    bad_old = {key: value.copy() for key, value in old.items()}
    bad_ap38 = {key: value.copy() for key, value in ap38.items()}
    for key in ('input__core_signal', 'input__fallback_signal',
                'input__fallback_quality'):
        bad_old[key][future] = ~bad_old[key][future].astype(bool)
    bad_old['meta_y20_probability'][future] = 1. - bad_old[
        'meta_y20_probability'][future]
    bad_ap38['core__quality_gate'][future] = ~bad_ap38[
        'core__quality_gate'][future].astype(bool)
    bad_signals, bad_diagnostics = build_ap45(panel, bad_old, bad_ap38)
    np.testing.assert_array_equal(
        signals[AP45_CANDIDATE][:cut], bad_signals[AP45_CANDIDATE][:cut])
    for key in diagnostics:
        same(diagnostics[key][:cut], bad_diagnostics[key][:cut])
    finish(AP45_OUT, AP45_CANDIDATE, AP45_CONTROLS, panel, saved,
           signals, diagnostics,
           {'ap38_ap44_dual_gate_and_rate120_exact': True})


def main():
    dates = panel_at(AP45_OUT).date.to_numpy()
    cut_day = dt.date(2025, 1, 6)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    future = np.arange(len(dates)) >= cut
    audit_ap42(cut, future)
    audit_ap43(cut, future)
    audit_ap44(cut, future)
    audit_ap45(cut, future)


if __name__ == '__main__':
    main()
