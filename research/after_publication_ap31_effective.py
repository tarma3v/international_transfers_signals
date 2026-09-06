"""AP31-E: factorized y3/y20 CatBoost pace expert."""
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
from research.after_publication_ap31_effective_models import (
    SCORE,
    fit_factorized_survival,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap31_effective')
BASE = OUT.parent / 'ap30_effective'
AP27 = OUT.parent / 'ap27_effective'
AP13 = OUT.parent / 'ap13_effective'
UNKNOWN_H = (3, 5, 10, 20)
CANDIDATE = 'cat_factor_y3_y20_dual_month24_cap2'
AP30 = 'rank75_s200_comp25_dual_month24_cap2'
AP29 = 's200_comp95_r60_backstop_month24_cap2'
AP27_STRICT = 's200_cat95_r60_backstop_month24_cap2'
AP26_BEST = 'y20_shrink200_coldstart_dual_month24_cap2'
AP23_BEST = 'soft730_pace_competence_dual_month24_cap2'
AP21_STRICT = 'roll_cat_dual_pace_month24_cap2'
AP18_CONTROL = 'full_recent50_pace365_p55_r70_month24_cap2'
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP30, AP29, AP27_STRICT, AP26_BEST, AP23_BEST, AP21_STRICT,
    AP18_CONTROL, AP17_CONTROL,
)


def fit_all(panel, X, y3, y20, eligible, cap):
    dates = panel.date.to_numpy()
    score = np.full(len(panel), np.nan)
    p3 = np.full(len(panel), np.nan)
    p20_given3 = np.full(len(panel), np.nan)
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
        train = (shared & np.asarray(eligible, dtype=bool)
                 & np.isfinite(y3) & np.isfinite(y20))
        values = fit_factorized_survival(X, y3, y20, train, query)
        score[query], p3[query], p20_given3[query], stats = values
        logs.append({
            'origin': str(origin), 'n_shared_train': int(shared.sum()),
            'n_query': int(query.sum()),
            'shared_mask_sha256': hashlib.sha256(
                np.packbits(shared).tobytes()).hexdigest(),
            'train_mask_sha256': hashlib.sha256(
                np.packbits(train).tobytes()).hexdigest(),
            'last_shared_mature20': str(max(cap['mature20'][shared])),
            'score_min': float(np.min(score[query])),
            'score_mean': float(np.mean(score[query])),
            'score_max': float(np.max(score[query])),
            **stats,
        })
        print(f'AP31 {origin}: shared {shared.sum()}, train {train.sum()}, '
              f'continuation {stats["long_n_train"]}', flush=True)
    return score, p3, p20_given3, logs


def build_policies(panel, score, rolling, reserve, eligible, old, ap27):
    result = dual_paced_month_policy(
        rolling, score, reserve, panel.date.to_numpy(),
        panel.currency.to_numpy(), eligible)
    signals = {CANDIDATE: result[0]}
    diagnostics = {
        'primary_rank': result[1], 'pace_rank': result[2],
        'reserve_rank': result[3], 'trailing_rate': result[4],
        'reason': result[5],
    }
    for control in (AP30, AP29):
        signals[control] = old['signal__' + control]
    for control in (*FROZEN_CONTROLS[2:], SIMPLE):
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
    for folder in (BASE, AP27, AP13):
        with np.load(folder / 'outputs.npz') as source:
            loaded.append({key: source[key] for key in source.files})
    old, ap27, ap13 = loaded
    X = ap13['features']
    y3 = ap13['conditional_labels'][:, 0]
    y20 = ap13['conditional_labels'][:, 3]
    eligible = ap27['eligible_next'].astype(bool)
    rolling = ap27['input__primary']
    reserve = ap27['input__reserve']
    score, p3, p20_given3, logs = fit_all(
        panel, X, y3, y20, eligible, cap)
    signals, diagnostics = build_policies(
        panel, score, rolling, reserve, eligible, old, ap27)
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
        'features': X, 'target_y3': y3, 'target_y20': y20,
        'expert__rolling': rolling, 'reserve_score': reserve,
        'prediction__score': score, 'prediction__p3': p3,
        'prediction__p20_given3': p20_given3,
    })
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    arrays.update({f'{name}__{CANDIDATE}': value
                   for name, value in diagnostics.items()})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [
        DATA,
        *(folder / name for folder in (BASE, AP27, AP13)
          for name in ('metadata.json', 'outputs.npz')),
        Path('research/after_publication_ap31_effective_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP31-E', 'n_rows': len(panel), 'n_fresh_models': 1,
        'n_quarterly_stage_fits': 2 * len(logs),
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
