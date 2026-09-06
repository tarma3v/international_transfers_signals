"""AP34-E: announced-anchor residual survival inside the frozen AP33 router."""
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
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap21_effective_models import dual_paced_month_policy
from research.after_publication_ap32_effective import (
    AP21_STRICT,
    AP23_BEST,
    AP26_BEST,
    CANDIDATE as AP32_CANDIDATE,
    SIMPLE,
)
from research.after_publication_ap33_effective import CANDIDATE as AP33_CANDIDATE
from research.after_publication_ap33_effective_models import calendar_fallback_router
from research.after_publication_ap34_effective_models import fit_residual_survival
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap34_effective')
BASE = OUT.parent / 'ap33_effective'
AP26 = OUT.parent / 'ap26_effective'
AP13 = OUT.parent / 'ap13_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'residual_survival_core_calendar_fallback_cap2'
FROZEN_CONTROLS = (
    AP33_CANDIDATE, AP32_CANDIDATE, AP23_BEST, AP26_BEST, AP21_STRICT,
)


def fit_all(panel, X, target, cap, eligible, known_change):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    score = np.full(len(panel), np.nan)
    diagnostics = {
        name: np.full(len(panel), np.nan)
        for name in ('ridge_prediction', 'local_bias', 'local_weight',
                     'global_probability', 'local_probability')
    }
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
        train = shared & eligible & np.isfinite(target)
        value, detail, stats = fit_residual_survival(
            X, target, train, query, dates, currencies, known_change, origin)
        score[query] = value
        for name, array in detail.items():
            diagnostics[name][query] = array
        local_state = stats.pop('local_state', {})
        logs.append({
            'origin': str(origin),
            'n_shared_train': int(shared.sum()),
            'n_train': int(train.sum()),
            'n_query': int(query.sum()),
            'shared_mask_sha256': hashlib.sha256(
                np.packbits(shared).tobytes()).hexdigest(),
            'train_mask_sha256': hashlib.sha256(
                np.packbits(train).tobytes()).hexdigest(),
            'last_shared_mature20': str(max(cap['mature20'][shared])),
            'local_state_json': json.dumps(local_state, sort_keys=True),
            'score_min': float(np.min(value)),
            'score_mean': float(np.mean(value)),
            'score_max': float(np.max(value)),
            **stats,
        })
        print(f'AP34 {origin}: shared {shared.sum()}, train {train.sum()}', flush=True)
    return score, diagnostics, pd.DataFrame(logs)


def build_policies(panel, score, rolling, reserve, fallback, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    source = dual_paced_month_policy(
        rolling, score, reserve, dates, currencies, eligible)
    final = calendar_fallback_router(
        source[0], fallback, dates, currencies, eligible)
    signals = {CANDIDATE: final[0]}
    for control in (*FROZEN_CONTROLS, SIMPLE):
        signals[control] = old['signal__' + control]
    diagnostics = {
        'source_signal': source[0],
        'source_primary_rank': source[1],
        'source_pace_rank': source[2],
        'source_reserve_rank': source[3],
        'source_trailing_rate': source[4],
        'source_reason': source[5],
        'final_trailing_rate': final[1],
        'final_days_since_signal': final[2],
        'final_reason': final[3],
    }
    assert all(not value[~eligible].any() for value in signals.values())
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
        summary.loc[CANDIDATE, 'joint_early_pass']) else AP33_CANDIDATE
    selection = {
        'selected': selected,
        'selected_simple': SIMPLE,
        'fresh_candidates': 1,
        'fresh_joint_early_pass_count': int(
            summary.loc[CANDIDATE, 'joint_early_pass']),
        'used_registered_fallback': selected != CANDIDATE,
        'registered_fallback': AP33_CANDIDATE,
        'selection_year': 2023,
        'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True,
        'h1_known_validity_only': True,
        'reference': 'today-effective CBR',
        'fresh_holdout': False,
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
    known_change = 10000 * np.log(
        panel.announced_price.to_numpy() / panel.current_price.to_numpy())
    target = outcomes['floor20'] - known_change
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    with np.load(BASE / 'outputs.npz') as source, \
            np.load(AP26 / 'outputs.npz') as ap26_source, \
            np.load(AP13 / 'outputs.npz') as ap13_source:
        old = {key: source[key] for key in source.files}
        ap26 = {key: ap26_source[key] for key in ap26_source.files}
        ap13 = {key: ap13_source[key] for key in ap13_source.files}
    compact = ap13['compact_feature_indices'].astype(int)
    X = ap13['features'][:, compact]
    eligible = old['eligible_next'].astype(bool)
    score, model_detail, training_log = fit_all(
        panel, X, target, cap, eligible, known_change)
    rolling = ap26['expert__rolling']
    reserve = ap26['reserve_score']
    fallback = old['signal__' + AP23_BEST].astype(bool)
    signals, policy_detail = build_policies(
        panel, score, rolling, reserve, fallback, eligible, old)
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
    arrays.update({
        'features': X,
        'compact_feature_indices': compact,
        'known_change': known_change,
        'residual_floor20_target': target,
        'model_score': score,
        'input__rolling': rolling,
        'input__reserve': reserve,
        'input__fallback_signal': fallback,
    })
    arrays.update({f'model__{name}': value
                   for name, value in model_detail.items()})
    arrays.update({f'policy__{name}': value
                   for name, value in policy_detail.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    training_log.to_csv(OUT / 'training_log.csv', index=False)
    sources = [
        DATA,
        *(folder / name for folder in (BASE, AP26, AP13)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap34_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP34-E',
        'n_rows': len(panel),
        'n_quarterly_fits': len(training_log),
        'n_features': X.shape[1],
        'n_fresh_scores': 1,
        'n_fresh_policies': 1,
        'n_frozen_controls': len(FROZEN_CONTROLS),
        'selection_horizons': list(UNKNOWN_H),
        'selected': selection['selected'],
        'reference': 'today-effective CBR',
        'h1_known_after_receipt': True,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'fresh_holdout': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }, indent=2))
    print(final[final.h == 5].sort_values(
        'adjusted_lift', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
