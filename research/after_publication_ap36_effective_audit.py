"""Independent AP36 mature-Hedge, nested-policy and corruption audit."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap3 import weekly_max
from research.after_publication_ap11_effective_audit import selected_bootstrap
from research.after_publication_ap36_effective import (
    AP21_STRICT,
    AP23_BEST,
    AP26,
    AP26_BEST,
    AP32_CANDIDATE,
    AP33_CANDIDATE,
    AP34,
    AP34_CANDIDATE,
    AP35_CANDIDATE,
    BASE,
    CANDIDATE,
    EXPERTS,
    FROZEN_CONTROLS,
    OUT,
    build_policies,
)
from research.after_publication_ap36_effective_models import mature_brier_hedge
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
    for folder in (OUT, BASE, AP34, AP26):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    saved, old, ap34, ap26 = loaded
    for key in ('dates', 'currencies', 'early', 'later', 'groups',
                'eligible_next'):
        np.testing.assert_array_equal(saved[key], old[key])
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            np.testing.assert_array_equal(saved[kind + str(h)],
                                          outcomes[kind + str(h)])
    np.testing.assert_array_equal(
        saved['mature20'], np.array([str(day) for day in cap['mature20']]))
    eligible = saved['eligible_next'].astype(bool)
    experts = {
        'ap26_y20': ap26['pace_score__y20_shrink200'],
        'ridge_survival': ap34['model_score'],
        'distributional_cat': old['model_score'],
    }
    assert tuple(experts) == EXPERTS
    hedge = mature_brier_hedge(
        experts, outcomes['y20'], cap['mature20'], dates,
        panel.currency.to_numpy(), eligible)
    score, ranks, weights, losses, local_count, global_count = hedge
    same(saved['hedge_score'], score)
    for j, name in enumerate(EXPERTS):
        same(saved['expert__' + name], experts[name])
        same(saved['rank__' + name], ranks[:, j])
        same(saved['weight__' + name], weights[:, j])
        same(saved['loss__' + name], losses[:, j])
        np.testing.assert_array_equal(saved['local_count__' + name], local_count[:, j])
        np.testing.assert_array_equal(saved['global_count__' + name], global_count[:, j])

    rolling = ap26['expert__rolling']
    reserve = ap26['reserve_score']
    fallback = old['signal__' + AP23_BEST].astype(bool)
    signals, policy_detail = build_policies(
        panel, score, rolling, reserve, fallback, eligible, old)
    for key, value in signals.items():
        np.testing.assert_array_equal(saved['signal__' + key], value)
        assert not value[~eligible].any()
    for name, value in policy_detail.items():
        same(saved['policy__' + name], value)

    cut_day = dt.date(2025, 1, 1)
    cut = int(np.searchsorted(dates, cut_day, side='left'))
    bad_experts = {name: value.copy() for name, value in experts.items()}
    for value in bad_experts.values():
        value[cut:] = value[cut:] * -9 + 37
    bad_y20 = outcomes['y20'].copy()
    bad_y20[cut:] = np.where(np.isfinite(bad_y20[cut:]),
                             1 - bad_y20[cut:], np.nan)
    bad_hedge = mature_brier_hedge(
        bad_experts, bad_y20, cap['mature20'], dates,
        panel.currency.to_numpy(), eligible)
    for left, right in zip(hedge, bad_hedge):
        same(left[:cut], right[:cut])
    bad_fallback = fallback.copy()
    bad_fallback[cut:] = ~bad_fallback[cut:]
    bad_signals, bad_policy = build_policies(
        panel, bad_hedge[0], rolling, reserve, bad_fallback, eligible, old)
    np.testing.assert_array_equal(
        signals[CANDIDATE][:cut], bad_signals[CANDIDATE][:cut])
    for name in policy_detail:
        same(policy_detail[name][:cut], bad_policy[name][:cut])

    common_audit(OUT, [CANDIDATE],
                 [(control, [CANDIDATE]) for control in FROZEN_CONTROLS])
    selection = json.loads((OUT / 'selection.json').read_text())
    early = pd.read_csv(OUT / 'early_summary.csv').set_index('candidate')
    passed = bool(early.loc[CANDIDATE, 'joint_early_pass'])
    assert selection['selected'] == (CANDIDATE if passed else AP33_CANDIDATE)
    later = pd.read_csv(OUT / 'retrospective_summary.csv').set_index('candidate')
    sensitivity = pd.concat([
        selected_bootstrap(panel, saved, CANDIDATE, control, block)
        for control in (AP33_CANDIDATE, AP34_CANDIDATE, AP35_CANDIDATE,
                        AP32_CANDIDATE, AP23_BEST, AP26_BEST, AP21_STRICT)
        for block in (20, 50)
    ])
    sensitivity.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    weight_rows = []
    for period, scope in (('early', saved['early']), ('late', saved['later'])):
        for currency in CORRIDORS:
            use = scope & (panel.currency.to_numpy() == currency)
            for j, name in enumerate(EXPERTS):
                weight_rows.append({
                    'period': period,
                    'currency': currency,
                    'expert': name,
                    'mean_weight': float(np.nanmean(weights[use, j])),
                    'mean_loss': float(np.nanmean(losses[use, j])),
                    'last_local_count': int(local_count[use, j][-1]),
                })
    pd.DataFrame(weight_rows).to_csv(OUT / 'hedge_weight_summary.csv', index=False)
    row = later.loc[CANDIDATE]
    max_week = weekly_max(panel, signals[CANDIDATE], saved['later'])
    strict = bool(row.min_lift > 2.4 and row.min_rate >= 1
                  and row.max_rate <= 2 and row.empty_complete_months == 0
                  and max_week <= 2)
    scope = saved['later'].astype(bool)
    final_reason = saved['policy__final_reason']
    source_reason = saved['policy__source_reason']
    final_names = ('none', 'hedge_core', 'ap23_late_week', 'ap23_silence10')
    source_names = ('none', 'rolling_primary', 'hedge_pace', 'month_rescue')
    counts = pd.concat([
        pd.DataFrame({
            'level': level,
            'reason': reason_names,
            'count': [int((scope & (reason == value)).sum())
                      for value in range(len(reason_names))],
        })
        for level, reason_names, reason in (
            ('final', final_names, final_reason),
            ('source', source_names, source_reason),
        )
    ], ignore_index=True)
    counts.to_csv(OUT / 'reason_counts.csv', index=False)
    (OUT / 'late_diagnostics.json').write_text(json.dumps({
        'selected_early': selection['selected'],
        'fresh_early_joint_pass': passed,
        'late_min_lift': float(row.min_lift),
        'late_mean_lift': float(row.mean_lift),
        'late_min_rate': float(row.min_rate),
        'late_max_weekly_signals': int(max_week),
        'late_strict_success': strict,
        'late_reason_counts': {
            level: dict(zip(frame.reason, frame['count']))
            for level, frame in counts.groupby('level')
        },
    }, indent=2))
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True,
        'events_and_targets_rebuilt': len(panel),
        'three_expert_causal_ranks_exact': True,
        'all_mature_brier_losses_counts_and_weights_exact': True,
        'nested_source_and_calendar_policy_state_exact': True,
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
