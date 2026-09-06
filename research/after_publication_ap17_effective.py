"""AP17-E: selected causal pacing plus a single empty-month rescue."""
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
from research.after_publication_ap17_effective_models import paced_month_policy
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap17_effective')
BASE = OUT.parent / 'ap16_effective'
AP12 = OUT.parent / 'ap12_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'extra_ap12_pace365_p55_r70_month24_cap2'
AP16_SELECTED = 'extra_ap12_pace365_p55_r70_cap2'
AP12_CONTROL = 'extra_h5_r30_nogap_cap2'
NEAR_CONTROL = 'extra_ap12_silence14_r80_cap2'
STRICT_CONTROL = 'extra_ap12_top35_cap2'
AP15_SELECTED = 'extra_ap12_silence14_month24_cap2'
AP14_SELECTED = 'extra_roll2_silence21_r70_cap2'
ROLLING_CONTROL = 'extra_roll2_primary_cap2'
LOCAL_CONTROL = 'local_extra_primary_cap2'
RESERVE_CONTROL = 'extra_roll2_reserve7_cap2'
ANCHOR = 'known_change_z_urgent_cap2'
HAZARD = 'hazard_hist_h5_urgent_cap2'
AP1_CAP = 'ap1_change_z_r25_exact_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP16_SELECTED, AP12_CONTROL, NEAR_CONTROL, STRICT_CONTROL, AP15_SELECTED,
    AP14_SELECTED, ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL,
    ANCHOR, HAZARD, AP1_CAP,
)


def build_policies(panel, score, reserve, eligible, old):
    result = paced_month_policy(
        score, reserve, panel.date.to_numpy(), panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: result[0]}
    for key in (*FROZEN_CONTROLS, SIMPLE):
        signals[key] = old['signal__' + key]
    assert all(not signal[~eligible].any() for signal in signals.values())
    diagnostics = {
        'primary_rank': result[1], 'reserve_rank': result[2],
        'trailing_rate': result[3], 'reason': result[4],
    }
    return signals, diagnostics


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

    early_frame = pd.DataFrame([{'candidate': key, **row}
                                for key, signal in signals.items()
                                for row in scorecard(panel, outcomes, signal,
                                                     old['early'], old['groups'])])
    early_uncertainty = benefit_bootstrap(panel, outcomes, signals, old['early'])
    early_summary = unknown_summary(early_frame)
    early_summary['min_benefit_lower_ci'] = early_uncertainty[
        early_uncertainty.h.isin(UNKNOWN_H)].groupby('candidate').ci_lo.min()
    early_summary['max_weekly_signals'] = [weekly_max(
        panel, signals[key], old['early']) for key in early_summary.index]
    forward = early_frame[early_frame.h.isin(UNKNOWN_H)].pivot(
        index='candidate', columns='h', values='forward_bps')
    early_summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    early_summary = early_summary.join(clustering_stats(
        panel, signals, old['early'], np.isfinite(outcomes['y5'])))
    early_summary['joint_early_pass'] = (
        (early_summary.min_lift >= 1.3) & (early_summary.min_rate >= 1)
        & (early_summary.max_rate <= 2) & (early_summary.min_benefit_lower_ci > 0)
        & (early_summary.max_weekly_signals <= 2)
        & (early_summary.early_forward_ratio_min >= .8)
        & (early_summary.empty_complete_months == 0))
    assert bool(early_summary.loc[CANDIDATE, 'joint_early_pass'])
    selection = {
        'selected': CANDIDATE, 'selected_simple': SIMPLE,
        'fresh_candidates': 1, 'fresh_joint_early_pass_count': 1,
        'selection_year': 2023, 'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True, 'h1_known_validity_only': True,
        'reference': 'today-effective CBR', 'fresh_holdout': False,
    }
    early_frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    early_uncertainty.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    early_summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)

    final = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal,
                                               old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    rows = []
    for baseline_name, baseline in (('ap16_selected', signals[AP16_SELECTED]),
                                    ('ap14_near', signals[NEAR_CONTROL]),
                                    ('ap12_top30', signals[AP12_CONTROL])):
        for period, scope in (('early', old['early']), ('later', old['later'])):
            current, previous = signals[CANDIDATE] & scope, baseline & scope
            rows.append({
                'candidate': CANDIDATE, 'baseline': baseline_name, 'period': period,
                'baseline_signals': int(previous.sum()), 'signals': int(current.sum()),
                'added': int((current & ~previous).sum()),
                'removed': int((previous & ~current).sum()),
                'net': int(current.sum() - previous.sum()),
                'primary_reason': int((scope & (diagnostics['reason'] == 1)).sum()),
                'pace_reason': int((scope & (diagnostics['reason'] == 2)).sum()),
                'month_reason': int((scope & (diagnostics['reason'] == 3)).sum()),
            })
    pd.DataFrame(rows).to_csv(OUT / 'signal_additions.csv', index=False)

    arrays = {key: old[key] for key in ('dates', 'currencies', 'early', 'later',
                                        'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays['score__extra_ap12'] = score
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    arrays.update({f'{name}__{CANDIDATE}': value
                   for name, value in diagnostics.items()})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               AP12 / 'metadata.json', AP12 / 'outputs.npz',
               Path('research/after_publication_ap17_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP17-E', 'n_rows': len(panel), 'n_fresh_policies': 1,
        'n_frozen_controls': len(FROZEN_CONTROLS),
        'selection_horizons': list(UNKNOWN_H), 'selected': CANDIDATE,
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
