"""AP18-E: new residual/stacking predictors under the frozen AP17 policy."""
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
from research.after_publication_ap12_effective_models import compact_feature_indices
from research.after_publication_ap13_effective import clustering_stats, unknown_summary
from research.after_publication_ap17_effective_models import paced_month_policy
from research.after_publication_ap18_effective_models import (
    SCORE_NAMES,
    fixed_blend,
    local_residual_ridge,
    residual_hist,
    residual_ridge,
    stack_logit,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap18_effective')
BASE = OUT.parent / 'ap17_effective'
AP12 = OUT.parent / 'ap12_effective'
AP13 = OUT.parent / 'ap13_effective'
FEATURE_META = OUT.parent / 'ap8' / 'metadata.json'
UNKNOWN_H = (3, 5, 10, 20)
AP17_CONTROL = 'extra_ap12_pace365_p55_r70_month24_cap2'
AP12_CONTROL = 'extra_h5_r30_nogap_cap2'
NEAR_CONTROL = 'extra_ap12_silence14_r80_cap2'
STRICT_CONTROL = 'extra_ap12_top35_cap2'
ROLLING_CONTROL = 'extra_roll2_primary_cap2'
LOCAL_CONTROL = 'local_extra_primary_cap2'
RESERVE_CONTROL = 'extra_roll2_reserve7_cap2'
ANCHOR = 'known_change_z_urgent_cap2'
HAZARD = 'hazard_hist_h5_urgent_cap2'
AP1_CAP = 'ap1_change_z_r25_exact_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    AP17_CONTROL, AP12_CONTROL, NEAR_CONTROL, STRICT_CONTROL,
    ROLLING_CONTROL, LOCAL_CONTROL, RESERVE_CONTROL, ANCHOR, HAZARD, AP1_CAP,
)


def candidate_key(score_name):
    return f'{score_name}_pace365_p55_r70_month24_cap2'


def fit_all(panel, X, y, eligible, cap, base, recent):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    names = json.loads(FEATURE_META.read_text())['feature_names']
    compact = compact_feature_indices(names)
    design = X[:, compact]
    assert np.isfinite(design).all()
    predictions = {name: np.full(len(panel), np.nan) for name in SCORE_NAMES}
    corrections = {name: np.full(len(panel), np.nan)
                   for name in ('resid_hist', 'resid_ridge', 'local_resid_ridge')}
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
        train = shared & eligible & np.isfinite(base) & np.isfinite(y)
        full_recent = fixed_blend(base[query], recent[query])
        hist100, hist_correction = residual_hist(design, y, base, train, query)
        hist50 = np.clip(base[query] + .5 * hist_correction, 0, 1)
        ridge_score, ridge_correction = residual_ridge(design, y, base, train, query)
        local_score, local_correction, local_counts = local_residual_ridge(
            design, y, base, train, query, currencies)
        stacked = stack_logit(design, y, base, train, query)
        values = {
            'full_recent50': full_recent,
            'resid_hist100': hist100,
            'resid_hist50': hist50,
            'resid_ridge': ridge_score,
            'local_resid_ridge': local_score,
            'stack_logit': stacked,
        }
        for name, value in values.items():
            predictions[name][query] = value
            correction = None
            if name.startswith('resid_hist'):
                correction = hist_correction * (1. if name.endswith('100') else .5)
            elif name == 'resid_ridge':
                correction = ridge_correction
            elif name == 'local_resid_ridge':
                correction = local_correction
            row = {
                'origin': str(origin), 'family': name,
                'n_shared_train': int(shared.sum()),
                'n_eligible_train': int(train.sum()), 'n_query': int(query.sum()),
                'shared_mask_sha256': hashlib.sha256(np.packbits(shared).tobytes()).hexdigest(),
                'train_mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
                'last_shared_mature20': str(max(cap['mature20'][shared])),
            }
            if correction is not None:
                row['correction_mean'] = float(np.mean(correction))
                row['correction_abs_mean'] = float(np.mean(np.abs(correction)))
                row['correction_max_abs'] = float(np.max(np.abs(correction)))
            if name == 'local_resid_ridge':
                row['local_counts'] = json.dumps(local_counts, sort_keys=True)
            logs.append(row)
        corrections['resid_hist'][query] = hist_correction
        corrections['resid_ridge'][query] = ridge_correction
        corrections['local_resid_ridge'][query] = local_correction
        print(f'AP18 {origin}: shared {shared.sum()}, train {train.sum()}, six scores', flush=True)
    return predictions, corrections, compact, logs


def build_policies(panel, predictions, reserve, eligible, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    signals, diagnostics = {}, {}
    for score_name in SCORE_NAMES:
        key = candidate_key(score_name)
        result = paced_month_policy(
            predictions[score_name], reserve, dates, currencies, eligible)
        signals[key] = result[0]
        diagnostics[key] = {
            'primary_rank': result[1], 'reserve_rank': result[2],
            'trailing_rate': result[3], 'reason': result[4],
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
    fresh = [candidate_key(name) for name in SCORE_NAMES]
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
        selected = str(passing.index[0]) if len(passing) else AP17_CONTROL
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
    with np.load(BASE / 'outputs.npz') as source, np.load(AP12 / 'outputs.npz') as b, \
            np.load(AP13 / 'outputs.npz') as p:
        old = {key: source[key] for key in source.files}
        ap12 = {key: b[key] for key in b.files}
        ap13 = {key: p[key] for key in p.files}
    X = ap13['features']
    y = ap13['conditional_labels'][:, 1]
    eligible = old['eligible_next'].astype(bool)
    base = ap12['prediction__extra_h5']
    recent = ap13['prediction__extra_roll2']
    predictions, corrections, compact, logs = fit_all(
        panel, X, y, eligible, cap, base, recent)
    reserve = ap12['score__known70_hazard30']
    signals, diagnostics = build_policies(panel, predictions, reserve, eligible, old)
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
    arrays.update({f'prediction__{key}': value for key, value in predictions.items()})
    arrays.update({f'correction__{key}': value for key, value in corrections.items()})
    arrays.update({f'signal__{key}': value for key, value in signals.items()})
    for candidate, detail in diagnostics.items():
        for name, value in detail.items():
            arrays[f'{name}__{candidate}'] = value
    arrays['compact_feature_indices'] = compact
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               AP12 / 'metadata.json', AP12 / 'outputs.npz',
               AP13 / 'metadata.json', AP13 / 'outputs.npz', FEATURE_META,
               Path('research/after_publication_ap18_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP18-E', 'n_rows': len(panel),
        'n_fresh_scores': len(SCORE_NAMES), 'n_frozen_controls': len(FROZEN_CONTROLS),
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
