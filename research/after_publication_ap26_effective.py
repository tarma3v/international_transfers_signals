"""AP26-E: cold-start-safe mixtures for specialist pace experts."""
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
from research.after_publication_ap26_effective_models import SCORES, cold_start_scores
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap26_effective')
BASE = OUT.parent / 'ap25_effective'
AP24 = OUT.parent / 'ap24_effective'
UNKNOWN_H = (3, 5, 10, 20)
AP25_LIFT = 'pace_cat_y20_gate_full_specialist_dual_month24_cap2'
AP25_H5 = 'pace_cat_future5_gate_full_specialist_dual_month24_cap2'
AP24_BEST = 'rank_qtr_mean_full_yeti_pace_ranker_dual_month24_cap2'
AP23_BEST = 'soft730_pace_competence_dual_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP25_LIFT, AP25_H5, AP24_BEST, AP23_BEST, AP21_STRICT,
    AP18_CONTROL, AP17_CONTROL,
)


def candidate_key(name):
    return f'{name}_coldstart_dual_month24_cap2'


def count_by_row(panel, training_log, family):
    dates = panel.date.to_numpy()
    result = np.full(len(panel), -1, dtype=np.int32)
    rows = training_log[training_log.family == family]
    for row in rows.itertuples(index=False):
        origin = pd.Timestamp(row.origin)
        end = (origin.to_period('Q') + 1).start_time
        query = ((pd.to_datetime(dates) >= origin)
                 & (pd.to_datetime(dates) < end))
        result[query] = int(row.n_specialist_train)
    return result


def build_policies(panel, scores, rolling, reserve, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for name, pace in scores.items():
        key = candidate_key(name)
        result = dual_paced_month_policy(
            rolling, pace, reserve, dates, currencies, eligible)
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
    fresh = [candidate_key(name) for name in SCORES]
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
    logs = pd.read_csv(BASE / 'training_log.csv')
    y20_count = count_by_row(panel, logs, 'pace_cat_y20_gate_full')
    mean_count = count_by_row(panel, logs, 'pace_cat_mean_gate_full')
    future_count = count_by_row(panel, logs, 'pace_cat_future5_gate_full')
    cat = ap24['expert__cat']
    rolling = old['expert__rolling']
    specialists = {
        'y20': old['prediction__pace_cat_y20_gate_full'],
        'mean': old['prediction__pace_cat_mean_gate_full'],
        'future5': old['prediction__pace_cat_future5_gate_full'],
    }
    scores, weights, cat_rank, future_rank = cold_start_scores(
        cat, specialists['y20'], specialists['mean'], specialists['future5'],
        y20_count, mean_count, future_count, panel.currency.to_numpy())
    reserve = old['reserve_score']
    eligible = old['eligible_next'].astype(bool)
    signals, diagnostics = build_policies(
        panel, scores, rolling, reserve, eligible, old)
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
    arrays['expert__rolling'] = rolling
    arrays['expert__cat'] = cat
    arrays['reserve_score'] = reserve
    arrays['specialist__y20'] = specialists['y20']
    arrays['specialist__mean'] = specialists['mean']
    arrays['specialist__future5'] = specialists['future5']
    arrays['count__y20'] = y20_count
    arrays['count__mean'] = mean_count
    arrays['count__future5'] = future_count
    arrays['inner_rank__cat'] = cat_rank
    arrays['inner_rank__future5'] = future_rank
    arrays.update({f'pace_score__{key}': value for key, value in scores.items()})
    arrays.update({f'weight__{key}': value for key, value in weights.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               BASE / 'training_log.csv', AP24 / 'metadata.json', AP24 / 'outputs.npz',
               Path('research/after_publication_ap26_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP26-E', 'n_rows': len(panel),
        'n_fresh_scores': len(SCORES),
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
