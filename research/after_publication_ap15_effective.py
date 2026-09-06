"""AP15-E: targeted causal closure of the remaining cadence gap."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap15_effective_models import POLICY_KINDS, targeted_policy
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap15_effective')
BASE = OUT.parent / 'ap14_effective'
AP12 = OUT.parent / 'ap12_effective'
UNKNOWN_H = (3, 5, 10, 20)
AP12_CONTROL = 'extra_h5_r30_nogap_cap2'
NEAR_CONTROL = 'extra_ap12_silence14_r80_cap2'
ADAPTIVE_CONTROL = 'extra_ap12_adaptive105_cap2'
STRICT_CONTROL = 'extra_ap12_top35_cap2'
SELECTED_CONTROL = 'extra_roll2_silence21_r70_cap2'
ROLLING_CONTROL = 'extra_roll2_primary_cap2'
LOCAL_CONTROL = 'local_extra_primary_cap2'
RESERVE_CONTROL = 'extra_roll2_reserve7_cap2'
ANCHOR = 'known_change_z_urgent_cap2'
HAZARD = 'hazard_hist_h5_urgent_cap2'
AP1_CAP = 'ap1_change_z_r25_exact_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP12_CONTROL, NEAR_CONTROL, ADAPTIVE_CONTROL, STRICT_CONTROL,
    SELECTED_CONTROL, ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL,
    ANCHOR, HAZARD, AP1_CAP,
)


def fresh_key(kind):
    return f'extra_ap12_{kind}_cap2'


def build_policies(panel, score, reserve, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for kind in POLICY_KINDS:
        key = fresh_key(kind)
        result = targeted_policy(score, reserve, dates, currencies, eligible, kind)
        signals[key] = result[0]
        diagnostics[key] = {
            'primary_rank': result[1], 'reserve_rank': result[2],
            'threshold': result[3], 'reason': result[4],
        }
    for key in (*FROZEN_CONTROLS, SIMPLE):
        signals[key] = old['signal__' + key]
    assert len(signals) == len(POLICY_KINDS) + len(FROZEN_CONTROLS) + 1
    assert all(not signal[~eligible].any() for signal in signals.values())
    return signals, diagnostics


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([
        {'candidate': key, **row}
        for key, signal in signals.items()
        for row in scorecard(panel, outcomes, signal, early, groups)
    ])
    uncertainty = benefit_bootstrap(panel, outcomes, signals, early)
    summary = unknown_summary(frame)
    summary['min_benefit_lower_ci'] = uncertainty[
        uncertainty.h.isin(UNKNOWN_H)].groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[key], early)
                                     for key in summary.index]
    forward = frame[frame.h.isin(UNKNOWN_H)].pivot(
        index='candidate', columns='h', values='forward_bps')
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    summary = summary.join(clustering_stats(
        panel, signals, early, np.isfinite(outcomes['y5'])))
    summary['rate_cap_pass'] = (
        (summary.min_rate >= 1) & (summary.max_rate <= 2)
        & (summary.max_weekly_signals <= 2))
    summary['joint_early_pass'] = (
        summary.rate_cap_pass & (summary.min_lift >= 1.3)
        & (summary.min_benefit_lower_ci > 0)
        & (summary.early_forward_ratio_min >= .8)
        & (summary.empty_complete_months == 0))
    fresh = [fresh_key(kind) for kind in POLICY_KINDS]
    passing = summary.loc[fresh]
    passing = passing[passing.joint_early_pass].sort_values(
        ['min_lift', 'mean_lift', 'calendar_gap_max'],
        ascending=[False, False, True], kind='stable')
    fallback = False
    if len(passing):
        selected = str(passing.index[0])
    else:
        fallback = True
        passing = summary.loc[fresh]
        passing = passing[passing.rate_cap_pass].sort_values(
            ['min_lift', 'mean_lift', 'calendar_gap_max'],
            ascending=[False, False, True], kind='stable')
        selected = str(passing.index[0]) if len(passing) else AP12_CONTROL
    selection = {
        'selected': selected, 'selected_simple': SIMPLE,
        'fresh_candidates': len(fresh),
        'fresh_joint_early_pass_count': int(summary.loc[fresh].joint_early_pass.sum()),
        'used_registered_rate_cap_fallback': fallback,
        'selection_year': 2023, 'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True,
        'h1_known_validity_only': True,
        'reference': 'today-effective CBR', 'fresh_holdout': False,
    }
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    uncertainty.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def addition_stats(signals, diagnostics, early, later):
    rows = []
    for candidate in (fresh_key(kind) for kind in POLICY_KINDS):
        for baseline_name, baseline in (
                ('ap12_top30', signals[AP12_CONTROL]),
                ('ap14_near', signals[NEAR_CONTROL])):
            for period, scope in (('early', early), ('later', later)):
                current, old = signals[candidate] & scope, baseline & scope
                rows.append({
                    'candidate': candidate, 'baseline': baseline_name, 'period': period,
                    'baseline_signals': int(old.sum()), 'signals': int(current.sum()),
                    'added': int((current & ~old).sum()),
                    'removed': int((old & ~current).sum()),
                    'net': int(current.sum() - old.sum()),
                    'primary_reason': int((scope & (diagnostics[candidate]['reason'] == 1)).sum()),
                    'silence_reason': int((scope & (diagnostics[candidate]['reason'] == 2)).sum()),
                    'month_reason': int((scope & (diagnostics[candidate]['reason'] == 3)).sum()),
                })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    with np.load(BASE / 'outputs.npz') as source, np.load(AP12 / 'outputs.npz') as base:
        old = {key: source[key] for key in source.files}
        ap12 = {key: base[key] for key in base.files}
    score = old['score__extra_ap12']
    np.testing.assert_array_equal(score, ap12['prediction__extra_h5'])
    reserve = ap12['score__known70_hazard30']
    eligible = old['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(panel, score, reserve, eligible, old)

    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([
        {'candidate': key, **row}
        for key, signal in signals.items()
        for row in scorecard(panel, outcomes, signal, old['later'], old['groups'])
    ])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    addition_stats(signals, diagnostics, old['early'], old['later']).to_csv(
        OUT / 'signal_additions.csv', index=False)

    arrays = {key: old[key] for key in ('dates', 'currencies', 'early', 'later',
                                        'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays['score__extra_ap12'] = score
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)

    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               AP12 / 'metadata.json', AP12 / 'outputs.npz',
               Path('research/after_publication_ap15_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP15-E', 'n_rows': len(panel),
        'n_fresh_policies': len(POLICY_KINDS),
        'n_frozen_controls': len(FROZEN_CONTROLS),
        'selection_horizons': list(UNKNOWN_H), 'selected': selection['selected'],
        'reference': 'today-effective CBR', 'h1_known_after_receipt': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }, indent=2))
    print(final[final.h == 5].sort_values(
        'adjusted_lift', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
