"""AP25-E: specialist models for hard cadence candidates."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap12_effective_models import causal_rank_percentile
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap21_effective_models import dual_paced_month_policy
from research.after_publication_ap25_effective_models import SPECS, fit_specialist
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap25_effective')
BASE = OUT.parent / 'ap24_effective'
AP12 = OUT.parent / 'ap12_effective'
AP13 = OUT.parent / 'ap13_effective'
UNKNOWN_H = (3, 5, 10, 20)
AP24_BEST = 'rank_qtr_mean_full_yeti_pace_ranker_dual_month24_cap2'
AP23_BEST = 'soft730_pace_competence_dual_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
AP12_CONTROL = 'extra_h5_r30_nogap_cap2'
ROLLING_CONTROL = 'extra_roll2_primary_cap2'
LOCAL_CONTROL = 'local_extra_primary_cap2'
RESERVE_CONTROL = 'extra_roll2_reserve7_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP24_BEST, AP23_BEST, AP21_STRICT, AP18_CONTROL, AP17_CONTROL,
    AP12_CONTROL, ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL,
)


def candidate_key(name):
    return f'{name}_specialist_dual_month24_cap2'


def hard_pool(rolling, reserve, currencies, eligible):
    rolling_rank = causal_rank_percentile(rolling, currencies)
    reserve_rank = causal_rank_percentile(reserve, currencies)
    pool = (np.asarray(eligible, dtype=bool) & np.isfinite(rolling_rank)
            & np.isfinite(reserve_rank) & (rolling_rank <= .70)
            & (reserve_rank > .70))
    return pool, rolling_rank, reserve_rank


def fit_all(panel, X, labels, future5, eligible, pool, cap):
    dates = panel.date.to_numpy()
    predictions = {name: np.full(len(panel), np.nan) for name in SPECS}
    logs = []
    origins = [dt.date(year, month, 1)
               for year in range(2022, max(day.year for day in dates) + 1)
               for month in (1, 4, 7, 10)
               if dt.date(year, month, 1) >= dt.date(2022, 7, 1)]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        shared = matured_mask(panel, cap, origin)
        train_pool = (shared & pool & eligible
                      & np.isfinite(labels).all(axis=1))
        for name, spec in SPECS.items():
            value, stats = fit_specialist(
                X, labels, future5, train_pool, query, dates, spec)
            predictions[name][query] = value
            logs.append({
                'origin': str(origin), 'family': name,
                'n_shared_train': int(shared.sum()),
                'n_pool_train': int(train_pool.sum()),
                'n_query': int(query.sum()),
                'shared_mask_sha256': hashlib.sha256(
                    np.packbits(shared).tobytes()).hexdigest(),
                'pool_mask_sha256': hashlib.sha256(
                    np.packbits(train_pool).tobytes()).hexdigest(),
                'last_shared_mature20': str(max(cap['mature20'][shared])),
                'score_min': float(np.min(value)),
                'score_mean': float(np.mean(value)),
                'score_max': float(np.max(value)),
                **stats,
            })
        print(f'AP25 {origin}: shared {shared.sum()}, hard-pool '
              f'{train_pool.sum()}, {len(SPECS)} specialists', flush=True)
    return predictions, logs


def build_policies(panel, predictions, rolling, reserve, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for name, pace in predictions.items():
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
    fresh = [candidate_key(name) for name in SPECS]
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
    folders = (BASE, AP12, AP13)
    loaded = []
    for folder in folders:
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap12, ap13 = loaded
    X = ap13['features']
    labels = ap13['conditional_labels']
    eligible = old['eligible_next'].astype(bool)
    rolling = old['expert__rolling']
    reserve = ap12['score__known70_hazard30']
    pool, rolling_rank, reserve_rank = hard_pool(
        rolling, reserve, panel.currency.to_numpy(), eligible)
    predictions, logs = fit_all(
        panel, X, labels, outcomes['forward5'], eligible, pool, cap)
    signals, diagnostics = build_policies(
        panel, predictions, rolling, reserve, eligible, old)
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
    arrays['reserve_score'] = reserve
    arrays['hard_pool'] = pool
    arrays['hard_pool_rolling_rank'] = rolling_rank
    arrays['hard_pool_reserve_rank'] = reserve_rank
    arrays.update({f'prediction__{key}': value
                   for key, value in predictions.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, *(folder / name for folder in folders
                       for name in ('metadata.json', 'outputs.npz')),
               Path('research/after_publication_ap25_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP25-E', 'n_rows': len(panel),
        'n_fresh_specialists': len(SPECS),
        'n_hard_pool_rows': int(pool.sum()),
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
