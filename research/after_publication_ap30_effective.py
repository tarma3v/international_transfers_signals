"""AP30-E: fixed AP26/AP23 pace interpolation in causal rank space."""
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
from research.after_publication_ap30_effective_models import rank_space_blend
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap30_effective')
BASE = OUT.parent / 'ap29_effective'
AP26 = OUT.parent / 'ap26_effective'
AP23 = OUT.parent / 'ap23_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'rank75_s200_comp25_dual_month24_cap2'
AP29 = 's200_comp95_r60_backstop_month24_cap2'
AP27_SELECTED = 's200_cat95_r70_backstop_month24_cap2'
AP27_STRICT = 's200_cat95_r60_backstop_month24_cap2'
AP26_BEST = 'y20_shrink200_coldstart_dual_month24_cap2'
AP23_BEST = 'soft730_pace_competence_dual_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP29, AP27_SELECTED, AP27_STRICT, AP26_BEST, AP23_BEST, AP21_STRICT,
    AP18_CONTROL, AP17_CONTROL,
)


def build_policies(panel, blend, rolling, reserve, eligible, old, ap27):
    result = dual_paced_month_policy(
        rolling, blend, reserve, panel.date.to_numpy(),
        panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: result[0]}
    diagnostics = {
        'primary_rank': result[1], 'pace_rank': result[2],
        'reserve_rank': result[3], 'trailing_rate': result[4],
        'reason': result[5],
    }
    signals[AP29] = old['signal__' + AP29]
    for control in (*FROZEN_CONTROLS[1:], SIMPLE):
        signals[control] = ap27['signal__' + control]
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
    summary['max_weekly_signals'] = [
        weekly_max(panel, signals[key], early) for key in summary.index]
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
    selected = CANDIDATE if bool(
        summary.loc[CANDIDATE, 'joint_early_pass']) else AP21_STRICT
    selection = {
        'selected': selected, 'selected_simple': SIMPLE,
        'fresh_candidates': 1,
        'fresh_joint_early_pass_count': int(
            summary.loc[CANDIDATE, 'joint_early_pass']),
        'used_registered_fallback': selected != CANDIDATE,
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
    loaded = []
    for folder in (BASE, BASE.parent / 'ap27_effective', AP26, AP23):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap27, ap26, ap23 = loaded
    specialist = ap26['pace_score__y20_shrink200']
    competence = ap23['pace_score__soft730_pace']
    blend, specialist_rank, competence_rank = rank_space_blend(
        specialist, competence, panel.currency.to_numpy())
    rolling = ap27['input__primary']
    reserve = ap27['input__reserve']
    eligible = ap27['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(
        panel, blend, rolling, reserve, eligible, old, ap27)
    selection = select_early(panel, outcomes, signals, ap27['early'], ap27['groups'])
    final = pd.DataFrame([{'candidate': key, **row}
                          for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal,
                                               ap27['later'], ap27['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, ap27['later'], np.isfinite(outcomes['y5']))).to_csv(
            OUT / 'retrospective_summary.csv')
    arrays = {key: ap27[key] for key in ('dates', 'currencies', 'early', 'later',
                                         'groups', 'eligible_next')}
    arrays.update({key: value for key, value in outcomes.items()
                   if not key.startswith('mature')})
    arrays.update({
        'input__specialist': specialist, 'input__competence': competence,
        'input__rolling': rolling, 'input__reserve': reserve,
        'inner_rank__specialist': specialist_rank,
        'inner_rank__competence': competence_rank, 'pace_score__blend': blend,
    })
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    arrays.update({f'{name}__{CANDIDATE}': value
                   for name, value in diagnostics.items()})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [
        DATA,
        *(folder / name
          for folder in (BASE, BASE.parent / 'ap27_effective', AP26, AP23)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap30_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP30-E', 'n_rows': len(panel), 'n_fresh_policies': 1,
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
