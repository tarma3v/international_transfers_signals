"""AP13-E: recent-window ExtraTrees, causal routers and cadence fallback."""
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
from research.after_publication_ap12_effective_models import compact_feature_indices
from research.after_publication_ap13_effective_models import (
    MODEL_SCORES,
    POLICY_KINDS,
    brier365_router,
    fit_extra,
    fit_local_extra,
    fit_meta_router,
    reserve_policy,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap13_effective')
BASE = OUT.parent / 'ap12_effective'
FEATURE_META = OUT.parent / 'ap8' / 'metadata.json'
UNKNOWN_H = (3, 5, 10, 20)
EXTRA_CONTROL = 'extra_h5_r30_nogap_cap2'
LOCAL_CONTROL = 'local_hist_h5_r30_nogap_cap2'
RESERVE_CONTROL = 'known70_hazard30_r275_nogap_cap2'
HAZARD = 'hazard_hist_h5_urgent_cap2'
ANCHOR = 'known_change_z_urgent_cap2'
SIMPLE = 'known_next_not_lower_cd3'
FROZEN_CONTROLS = (
    EXTRA_CONTROL,
    LOCAL_CONTROL,
    RESERVE_CONTROL,
    HAZARD,
    ANCHOR,
    'ap1_change_z_r25_exact',
    'ap1_change_z_r25_exact_cap2',
    SIMPLE,
)


def fit_all(panel, X, compact, conditional_y5, eligible, cap, extra, local, outcomes):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    prediction = {name: np.full(len(panel), np.nan) for name in MODEL_SCORES[:-1]}
    router_weight = np.full(len(panel), np.nan)
    logs = []
    origins = [dt.date(y, m, 1) for y in range(2022, max(d.year for d in dates) + 1)
               for m in (1, 4, 7, 10) if dt.date(y, m, 1) >= dt.date(2022, 7, 1)]
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        age = np.array([(origin - day).days for day in dates], dtype=float)
        roll2 = train & (dates >= origin - dt.timedelta(days=730))
        roll3 = train & (dates >= origin - dt.timedelta(days=1095))
        decay = np.exp2(-age / 730.)
        values = {
            'extra_roll2': fit_extra(X, conditional_y5, roll2, query),
            'extra_roll3': fit_extra(X, conditional_y5, roll3, query),
            'extra_decay730': fit_extra(X, conditional_y5, train, query, decay),
        }
        values['local_extra'], local_counts = fit_local_extra(
            X, conditional_y5, train, query, currencies, extra[query])
        meta, weight, meta_n, meta_extra = fit_meta_router(
            X[:, compact], conditional_y5, train, query, extra, local)
        values['router_extra_local'] = meta
        router_weight[query] = weight
        for family, value in values.items():
            prediction[family][query] = value
            row = {
                'origin': str(origin), 'family': family,
                'n_shared_train': int(shared.sum()), 'n_eligible_train': int(train.sum()),
                'n_query': int(query.sum()),
                'shared_mask_sha256': hashlib.sha256(np.packbits(shared).tobytes()).hexdigest(),
                'eligible_mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
                'last_shared_mature20': str(max(cap['mature20'][shared])),
            }
            if family == 'extra_roll2':
                row['n_model_train'] = int(roll2.sum())
            elif family == 'extra_roll3':
                row['n_model_train'] = int(roll3.sum())
            else:
                row['n_model_train'] = int(train.sum())
            if family == 'extra_decay730':
                row['weight_sum'] = float(decay[train].sum())
            if family == 'local_extra':
                row['local_counts'] = json.dumps(local_counts, sort_keys=True)
            if family == 'router_extra_local':
                row['meta_n'] = meta_n
                row['meta_prefer_extra'] = meta_extra
            logs.append(row)
        print(f'AP13 {origin}: shared {shared.sum()}, eligible {train.sum()}, five fitted scores', flush=True)
    brier, brier_weights, brier_counts = brier365_router(
        dates, outcomes['mature5'], conditional_y5, eligible, extra, local)
    prediction['brier365_extra_local'] = brier
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        shared = matured_mask(panel, cap, origin)
        train = shared & eligible
        logs.append({
            'origin': str(origin), 'family': 'brier365_extra_local',
            'n_shared_train': int(shared.sum()), 'n_eligible_train': int(train.sum()),
            'n_model_train': int(brier_counts[query].max()), 'n_query': int(query.sum()),
            'shared_mask_sha256': hashlib.sha256(np.packbits(shared).tobytes()).hexdigest(),
            'eligible_mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
            'last_shared_mature20': str(max(cap['mature20'][shared])),
            'brier_count_min': int(brier_counts[query].min()),
            'brier_count_max': int(brier_counts[query].max()),
            'extra_weight_min': float(brier_weights[query, 0].min()),
            'extra_weight_max': float(brier_weights[query, 0].max()),
        })
    return prediction, router_weight, brier_weights, brier_counts, logs


def build_policies(panel, prediction, old):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    eligible = old['eligible_next'].astype(bool)
    reserve = old['score__known70_hazard30']
    signals = {}
    for score_name in MODEL_SCORES:
        for kind in POLICY_KINDS:
            key = f'{score_name}_{kind}_cap2'
            signals[key] = reserve_policy(
                prediction[score_name], reserve, dates, currencies, eligible, kind)
    for key in FROZEN_CONTROLS:
        signals[key] = old['signal__' + key]
    assert len(signals) == 26
    for key, signal in signals.items():
        if key != 'ap1_change_z_r25_exact':
            assert not signal[~eligible].any(), key
    return signals


def unknown_summary(frame):
    use = frame[frame.h.isin(UNKNOWN_H)]
    return use.groupby('candidate', sort=False).agg(
        min_lift=('adjusted_lift', 'min'), mean_lift=('adjusted_lift', 'mean'),
        min_rate=('currency_rate_min', 'min'), max_rate=('currency_rate_max', 'max'),
        min_forward=('forward_bps', 'min'), min_symmetric=('symmetric_bps', 'min'))


def clustering_stats(panel, signals, scope, valid):
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    rows = []
    for key, signal in signals.items():
        empty, max_gap = 0, 0
        for currency in CORRIDORS:
            available = np.flatnonzero(scope & valid & (currencies == currency))
            chosen = dates[scope & valid & signal & (currencies == currency)]
            if not len(available):
                continue
            first, last = dates[available[0]], dates[available[-1]]
            months = pd.period_range(first, last, freq='M')
            selected_months = pd.Series(chosen).map(lambda d: pd.Period(d, freq='M')).value_counts()
            for month in months:
                complete = month.start_time.date() >= first and month.end_time.date() <= last
                if complete and selected_months.get(month, 0) == 0:
                    empty += 1
            if len(chosen) > 1:
                max_gap = max(max_gap, max((b - a).days for a, b in zip(chosen, chosen[1:])))
        rows.append({'candidate': key, 'empty_complete_months': empty, 'calendar_gap_max': max_gap})
    return pd.DataFrame(rows).set_index('candidate')


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, early, groups)])
    bootstrap = benefit_bootstrap(panel, outcomes, signals, early)
    summary = unknown_summary(frame)
    summary['min_benefit_lower_ci'] = bootstrap[bootstrap.h.isin(UNKNOWN_H)].groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[k], early) for k in summary.index]
    forward = frame[frame.h.isin(UNKNOWN_H)].pivot(index='candidate', columns='h', values='forward_bps')
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    clusters = clustering_stats(panel, signals, early, np.isfinite(outcomes['y5']))
    summary = summary.join(clusters)
    summary['joint_early_pass'] = (
        (summary.min_lift >= 1.3) & (summary.min_rate >= 1) & (summary.max_rate <= 2)
        & (summary.min_benefit_lower_ci > 0) & (summary.max_weekly_signals <= 2)
        & (summary.early_forward_ratio_min >= .8) & (summary.empty_complete_months == 0))
    fresh = [k for k in summary.index if k not in FROZEN_CONTROLS]
    feasible = summary.loc[fresh]
    feasible = feasible[feasible.joint_early_pass].sort_values(
        ['min_lift', 'mean_lift', 'calendar_gap_max'], ascending=[False, False, True], kind='stable')
    selected = str(feasible.index[0]) if len(feasible) else EXTRA_CONTROL
    selection = {
        'selected': selected, 'selected_simple': SIMPLE,
        'fresh_candidates': len(fresh), 'fresh_joint_early_pass_count': len(feasible),
        'selection_year': 2023, 'selection_horizons': list(UNKNOWN_H),
        'selected_before_later_scorecard': True, 'h1_known_validity_only': True,
        'reference': 'today-effective CBR', 'fresh_holdout': False,
    }
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    bootstrap.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, digest in json.loads((BASE / 'metadata.json').read_text())['source_sha256'].items():
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
    with np.load(BASE / 'outputs.npz') as z:
        old = {k: z[k] for k in z.files}
    names = json.loads(FEATURE_META.read_text())['feature_names']
    assert len(names) == old['features'].shape[1]
    compact = compact_feature_indices(names)
    X = old['features']
    eligible = old['eligible_next'].astype(bool)
    y5 = old['conditional_labels'][:, 1]
    extra = old['prediction__extra_h5']
    local = old['prediction__local_hist_h5']
    prediction, router_weight, brier_weights, brier_counts, logs = fit_all(
        panel, X, compact, y5, eligible, cap, extra, local, outcomes)
    signals = build_policies(panel, prediction, old)
    selection = select_early(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    unknown_summary(final).join(clustering_stats(
        panel, signals, old['later'], np.isfinite(outcomes['y5']))).to_csv(OUT / 'retrospective_summary.csv')
    arrays = {k: old[k] for k in ('dates', 'currencies', 'early', 'later', 'groups',
                                  'eligible_next', 'conditional_labels', 'features')}
    arrays.update({f'prediction__{k}': v for k, v in prediction.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update(router_weight=router_weight, brier_weights=brier_weights,
                  brier_counts=brier_counts, compact_feature_indices=compact)
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz', FEATURE_META,
               Path('research/after_publication_ap13_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP13-E', 'n_rows': len(panel), 'n_fresh_scores': len(MODEL_SCORES),
        'n_fresh_policies': len(MODEL_SCORES) * len(POLICY_KINDS),
        'n_total_policies': len(signals), 'selection_horizons': list(UNKNOWN_H),
        'reference': 'today-effective CBR', 'h1_known_after_receipt': True,
        'historical_receipts_certified': False, 'bank_execution_validated': False,
        'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
