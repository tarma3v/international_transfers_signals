"""AP27-E: sparse global backstop for cold-start specialist pace."""
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
from research.after_publication_ap27_effective_models import (
    POLICIES,
    triple_paced_month_policy,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap27_effective')
BASE = OUT.parent / 'ap26_effective'
AP24 = OUT.parent / 'ap24_effective'
UNKNOWN_H = (3, 5, 10, 20)
AP26_BEST = 'y20_shrink200_coldstart_dual_month24_cap2'
AP26_SELECTED = 'y20_hard100_coldstart_dual_month24_cap2'
AP23_BEST = 'soft730_pace_competence_dual_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP26_BEST, AP26_SELECTED, AP23_BEST, AP21_STRICT,
    AP18_CONTROL, AP17_CONTROL,
)


def candidate_key(name):
    return f'{name}_backstop_month24_cap2'


def policy_inputs(old, cat):
    return {
        'primary': old['expert__rolling'],
        's200': old['pace_score__y20_shrink200'],
        's100': old['pace_score__y20_shrink100'],
        'cat': cat,
        'reserve': old['reserve_score'],
    }


def build_policies(panel, inputs, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for name, (pace_name, rate, threshold, silence) in POLICIES.items():
        key = candidate_key(name)
        result = triple_paced_month_policy(
            inputs['primary'], inputs[pace_name], inputs['cat'], inputs['reserve'],
            dates, currencies, eligible, rate, threshold, silence)
        signals[key] = result[0]
        diagnostics[key] = {
            'primary_rank': result[1], 'pace_rank': result[2],
            'backstop_rank': result[3], 'reserve_rank': result[4],
            'trailing_rate': result[5], 'days_since': result[6],
            'reason': result[7],
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
    fresh = [candidate_key(name) for name in POLICIES]
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
    with np.load(BASE / 'outputs.npz') as source, \
            np.load(AP24 / 'outputs.npz') as ap24_source:
        old = {key: source[key] for key in source.files}
        ap24 = {key: ap24_source[key] for key in ap24_source.files}
    inputs = policy_inputs(old, ap24['expert__cat'])
    eligible = old['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(panel, inputs, eligible, old)
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
    arrays.update({f'input__{key}': value for key, value in inputs.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               AP24 / 'metadata.json', AP24 / 'outputs.npz',
               Path('research/after_publication_ap27_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP27-E', 'n_rows': len(panel),
        'n_fresh_policies': len(POLICIES),
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
