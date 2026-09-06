"""AP12-E: fresh conditional rankers and a cadence-compatible controller."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap11_effective_models import eligible_next
from research.after_publication_ap12_effective_models import (
    MODEL_SCORES,
    causal_rank_cap2,
    causal_rank_percentile,
    compact_feature_indices,
    fit_extra,
    fit_first_failure_extra,
    fit_hist,
    fit_local_hist,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap12_effective')
BASE = OUT.parent / 'ap11_effective'
AP10 = OUT.parent / 'ap10_effective_extended'
UNKNOWN_H = (3, 5, 10, 20)
ANCHOR = 'known_change_z_urgent_cap2'
HAZARD = 'hazard_hist_h5_urgent_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FRESH_SCORE_NAMES = (*MODEL_SCORES, 'known70_hazard30', 'known_change_z')
FROZEN_CONTROLS = (
    ANCHOR,
    SIMPLE,
    'market_hist_gated_urgent_cap2',
    HAZARD,
    'margin_ridge_local_mean_urgent_cap2',
    'ap1_change_z_r25_exact',
    'ap1_change_z_r25_exact_cap2',
    'known_change_z_r25w250_cap2',
)


def fit_all(panel, X, names, conditional, eligible, cap):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    compact = compact_feature_indices(names)
    prediction = {name: np.full(len(panel), np.nan) for name in MODEL_SCORES}
    logs = []
    origins = [dt.date(y, m, 1) for y in range(2022, max(d.year for d in dates) + 1)
               for m in (1, 4, 7, 10) if dt.date(y, m, 1) >= dt.date(2022, 7, 1)]
    y5 = conditional[:, 1]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        assert np.isfinite(conditional[train]).all()
        full_extra = fit_extra(X, y5, train, query)
        compact_hist = fit_hist(X[:, compact], y5, train, query)
        compact_extra = fit_extra(X[:, compact], y5, train, query)
        local_hist, local_counts = fit_local_hist(
            X[:, compact], y5, train, query, currencies, compact_hist)
        failure, class_counts = fit_first_failure_extra(X, conditional, train, query)
        values = {
            'extra_h5': full_extra,
            'compact_hist_h5': compact_hist,
            'compact_extra_h5': compact_extra,
            'local_hist_h5': local_hist,
            'first_failure_extra_h5': failure,
        }
        for family, value in values.items():
            prediction[family][query] = value
            row = {
                'origin': str(origin), 'family': family,
                'n_shared_train': int(shared.sum()),
                'n_eligible_train': int(train.sum()),
                'n_query': int(query.sum()),
                'shared_mask_sha256': hashlib.sha256(np.packbits(shared).tobytes()).hexdigest(),
                'eligible_mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
                'last_shared_mature20': str(max(cap['mature20'][shared])),
                'compact_features': int(len(compact)) if 'compact' in family or family == 'local_hist_h5' else len(names),
            }
            if family == 'local_hist_h5':
                row['local_counts'] = json.dumps(local_counts, sort_keys=True)
            if family == 'first_failure_extra_h5':
                row['class_counts'] = json.dumps(class_counts, sort_keys=True)
            logs.append(row)
        print(f'AP12 {origin}: shared {shared.sum()}, eligible {train.sum()}, five fresh scores', flush=True)
    return prediction, logs, compact


def build_policies(panel, prediction, old, ap10, names):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    eligible = old['eligible_next'].astype(bool)
    known_z = old['announced_features'][:, names.index('known_change_z')]
    hazard = old['score__hazard_hist_h5']
    known_rank = causal_rank_percentile(known_z, currencies)
    hazard_rank = causal_rank_percentile(hazard, currencies)
    blend = .70 * known_rank + .30 * hazard_rank
    scores = {**prediction, 'known70_hazard30': blend, 'known_change_z': known_z}
    signals = {}
    for score_name in MODEL_SCORES:
        for gap_name, gap in (('nogap', 0), ('gap2', 2)):
            key = f'{score_name}_r30_{gap_name}_cap2'
            signals[key] = causal_rank_cap2(scores[score_name], dates, currencies, eligible, .30, gap)
    for score_name in ('known70_hazard30', 'known_change_z'):
        signals[f'{score_name}_r275_nogap_cap2'] = causal_rank_cap2(
            scores[score_name], dates, currencies, eligible, .275, 0)
        for gap_name, gap in (('nogap', 0), ('gap2', 2)):
            key = f'{score_name}_r30_{gap_name}_cap2'
            signals[key] = causal_rank_cap2(scores[score_name], dates, currencies, eligible, .30, gap)
    for key in FROZEN_CONTROLS:
        if key in (ANCHOR, SIMPLE, 'market_hist_gated_urgent_cap2'):
            signals[key] = ap10['signal__' + key]
        else:
            signals[key] = old['signal__' + key]
    assert len(signals) == 24 and len(scores) == 7
    for key, value in signals.items():
        if key != 'ap1_change_z_r25_exact':
            assert not value[~eligible].any(), key
    return scores, signals


def unknown_summary(frame):
    use = frame[frame.h.isin(UNKNOWN_H)]
    result = use.groupby('candidate', sort=False).agg(
        min_lift=('adjusted_lift', 'min'),
        mean_lift=('adjusted_lift', 'mean'),
        min_rate=('currency_rate_min', 'min'),
        max_rate=('currency_rate_max', 'max'),
        min_forward=('forward_bps', 'min'),
        min_symmetric=('symmetric_bps', 'min'),
    )
    return result


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, early, groups)])
    bootstrap = benefit_bootstrap(panel, outcomes, signals, early)
    summary = unknown_summary(frame)
    unknown_boot = bootstrap[bootstrap.h.isin(UNKNOWN_H)]
    summary['min_benefit_lower_ci'] = unknown_boot.groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[k], early) for k in summary.index]
    forward = frame[frame.h.isin(UNKNOWN_H)].pivot(index='candidate', columns='h', values='forward_bps')
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    summary['joint_early_pass'] = (
        (summary.min_lift >= 1.3)
        & (summary.min_rate >= 1)
        & (summary.max_rate <= 2)
        & (summary.min_benefit_lower_ci > 0)
        & (summary.max_weekly_signals <= 2)
        & (summary.early_forward_ratio_min >= .8)
    )
    fresh = [k for k in summary.index if k not in FROZEN_CONTROLS]
    feasible = summary.loc[fresh]
    feasible = feasible[feasible.joint_early_pass].sort_values(
        ['min_lift', 'mean_lift', 'min_rate'], ascending=[False, False, True], kind='stable')
    selected = str(feasible.index[0]) if len(feasible) else ANCHOR
    selection = {
        'selected': selected,
        'selected_simple': SIMPLE,
        'fresh_candidates': len(fresh),
        'fresh_joint_early_pass_count': len(feasible),
        'selection_year': 2023,
        'selected_before_later_scorecard': True,
        'selection_horizons': list(UNKNOWN_H),
        'h1_is_known_validity_check_only': True,
        'reference': 'today-effective CBR',
        'fresh_holdout': False,
    }
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    bootstrap.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, AP10):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    names = json.loads((BASE.parent / 'ap10_effective' / 'metadata.json').read_text())['cbr_feature_names']
    with np.load(BASE / 'outputs.npz') as z, np.load(AP10 / 'outputs.npz') as a:
        old, ap10 = ({k: obj[k] for k in obj.files} for obj in (z, a))
    X = old['features']
    conditional = old['conditional_labels']
    eligible = eligible_next(panel)
    np.testing.assert_array_equal(eligible, old['eligible_next'])
    prediction, logs, compact = fit_all(panel, X, names, conditional, eligible, cap)
    scores, signals = build_policies(panel, prediction, old, ap10, names)
    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).to_csv(OUT / 'retrospective_summary.csv')
    arrays = {k: old[k] for k in ('dates', 'currencies', 'early', 'later', 'groups',
                                  'eligible_next', 'conditional_labels')}
    arrays.update(features=X, compact_feature_indices=compact)
    arrays.update({f'prediction__{k}': v for k, v in prediction.items()})
    arrays.update({f'score__{k}': v for k, v in scores.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz', AP10 / 'metadata.json',
               AP10 / 'outputs.npz', Path('research/after_publication_ap12_effective_registered.md')]
    metadata = {
        'packet': 'AP12-E', 'n_rows': len(panel), 'n_fresh_scores': len(MODEL_SCORES),
        'n_fresh_policies': 16, 'n_total_policies': len(signals),
        'compact_feature_count': int(len(compact)), 'reference': 'today-effective CBR',
        'selection_horizons': list(UNKNOWN_H), 'h1_known_after_receipt': True,
        'historical_receipts_certified': False, 'bank_execution_validated': False,
        'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
