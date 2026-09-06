"""AP23-E: mature-only competence routing under the frozen AP21 policy."""
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
from research.after_publication_ap21_effective_models import dual_paced_month_policy
from research.after_publication_ap23_effective_models import (
    EXPERTS,
    PAIRINGS,
    competence_pairings,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap23_effective')
BASE = OUT.parent / 'ap22_effective'
AP12 = OUT.parent / 'ap12_effective'
AP18 = OUT.parent / 'ap18_effective'
UNKNOWN_H = (3, 5, 10, 20)
AP21_SELECTED = 'roll75local25_ap1850cat50_dual_pace_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP21_H5 = 'local_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
AP12_CONTROL = 'extra_h5_r30_nogap_cap2'
ROLLING_CONTROL = 'extra_roll2_primary_cap2'
LOCAL_CONTROL = 'local_extra_primary_cap2'
RESERVE_CONTROL = 'extra_roll2_reserve7_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP21_SELECTED, AP21_STRICT, AP21_H5, AP18_CONTROL, AP17_CONTROL,
    AP12_CONTROL, ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL,
)


def candidate_key(name):
    return f'{name}_competence_dual_month24_cap2'


def build_policies(panel, pairs, reserve, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for name in PAIRINGS:
        key = candidate_key(name)
        result = dual_paced_month_policy(
            pairs[name][0], pairs[name][1], reserve, dates, currencies, eligible)
        signals[key] = result[0]
        diagnostics[key] = {
            'primary_rank': result[1], 'pace_rank': result[2],
            'reserve_rank': result[3], 'trailing_rate': result[4],
            'reason': result[5],
        }
    for key in (*FROZEN_CONTROLS, SIMPLE):
        signals[key] = old['signal__' + key]
    assert all(not signal[~eligible].any() for signal in signals.values())
    return signals, diagnostics


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, early, groups)])
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
    summary['rate_cap_pass'] = ((summary.min_rate >= 1) & (summary.max_rate <= 2)
                                & (summary.max_weekly_signals <= 2))
    summary['joint_early_pass'] = (
        summary.rate_cap_pass & (summary.min_lift >= 1.3)
        & (summary.min_benefit_lower_ci > 0)
        & (summary.early_forward_ratio_min >= .8)
        & (summary.empty_complete_months == 0))
    fresh = [candidate_key(name) for name in PAIRINGS]
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
        selected = str(passing.index[0]) if len(passing) else AP21_STRICT
    selection = {
        'selected': selected, 'selected_simple': SIMPLE,
        'fresh_candidates': len(fresh),
        'fresh_joint_early_pass_count': int(summary.loc[fresh].joint_early_pass.sum()),
        'used_registered_rate_cap_fallback': fallback,
        'selection_year': 2023, 'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True, 'h1_known_validity_only': True,
        'reference': 'today-effective CBR', 'fresh_holdout': False,
    }
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    uncertainty.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


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
    folders = (BASE, AP12, AP18)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap12, ap18 = loaded
    eligible = old['eligible_next'].astype(bool)
    experts = {
        'rolling': old['expert__rolling'],
        'local': old['expert__local'],
        'ap18': ap18['prediction__full_recent50'],
        'cat': old['expert__cat'],
    }
    assert tuple(experts) == EXPERTS
    pairs, ranks, competence, counts, weights, utility = competence_pairings(
        experts, outcomes, cap['mature20'], panel.date.to_numpy(),
        panel.currency.to_numpy(), eligible)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, pairs, reserve, eligible, old)
    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])

    final = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal,
                                               old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    arrays = {key: old[key] for key in ('dates', 'currencies', 'early', 'later',
                                        'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays['unknown_utility'] = utility
    arrays.update({f'expert__{key}': value for key, value in experts.items()})
    arrays.update({f'inner_rank__{key}': value for key, value in ranks.items()})
    for (window, name), value in competence.items():
        arrays[f'competence{window}__{name}'] = value
        arrays[f'local_count{window}__{name}'] = counts[(window, name, 'local')]
        arrays[f'global_count{window}__{name}'] = counts[(window, name, 'global')]
    arrays.update({f'weight__{key}': value for key, value in weights.items()})
    for name, (primary, pace) in pairs.items():
        arrays[f'primary_score__{name}'] = primary
        arrays[f'pace_score__{name}'] = pace
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [DATA, *(folder / name for folder in folders
                       for name in ('metadata.json', 'outputs.npz')),
               Path('research/after_publication_ap23_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP23-E', 'n_rows': len(panel),
        'n_fresh_pairings': len(PAIRINGS),
        'n_frozen_controls': len(FROZEN_CONTROLS),
        'selection_horizons': list(UNKNOWN_H), 'selected': selection['selected'],
        'reference': 'today-effective CBR', 'h1_known_after_receipt': True,
        'router_target': 'mean(y3,y5,y10,y20)',
        'router_maturity': 'publication mature20 < decision date - 2 days',
        'historical_receipts_certified': False,
        'bank_execution_validated': False, 'fresh_holdout': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }, indent=2))
    print(final[final.h == 5].sort_values(
        'adjusted_lift', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
