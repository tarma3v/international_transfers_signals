"""AP11-E: rules and models for the unknown remainder after known step1."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap3_policy import past_percentiles, sequential_policy
from research.after_publication_ap11_effective_models import (FAMILIES, conditional_targets,
    effective_scale, eligible_next, fit_classifier, fit_hazard, fit_local_logit,
    fit_margin, full_curve, margin_scores, margin_targets)
from research.after_publication_effective_information import SIMPLE
from research.after_publication_panel import build_outcomes

OUT = Path('results/research/after_publication/ap11_effective')
BASE = OUT.parent / 'ap10_effective_extended'
AP1 = OUT.parent / 'ap1'
ANCHOR = 'known_change_z_urgent_cap2'
AP1_EXACT = 'ap1_change_z_r25_exact'


def old_ap1_signal(panel):
    old_panel = pd.read_csv(AP1 / 'announcement_panel.csv')
    old_panel.date = pd.to_datetime(old_panel.date).dt.date
    with np.load(AP1 / 'publication' / 'outputs.npz') as z:
        old = z['fired_change_z_r25']
    mapping = {(c, d): bool(v) for c, d, v in zip(old_panel.currency, old_panel.date, old)}
    keys = list(zip(panel.currency, panel.date))
    if len(mapping) != len(old_panel) or not all(k in mapping for k in keys):
        raise ValueError('AP1 signal mapping is not one-to-one and complete')
    return np.array([mapping[k] for k in keys], dtype=bool)


def cap_opportunities(opportunity, dates, currencies, eligible, minimum_gap=2):
    result = np.zeros(len(opportunity), dtype=bool)
    for currency in CORRIDORS:
        week, used, last = None, 0, None
        for i in np.flatnonzero(currencies == currency):
            iso = dates[i].isocalendar()[:2]
            if iso != week:
                week, used = iso, 0
            if (not opportunity[i] or not eligible[i] or used >= 2 or
                    (last is not None and (dates[i] - last).days < minimum_gap)):
                continue
            result[i], used, last = True, used + 1, dates[i]
    return result


def fixed_rank_cap2(values, dates, currencies, eligible, window=250, warmup=40):
    opportunity = np.zeros(len(values), dtype=bool)
    for currency in CORRIDORS:
        history = []
        for i in np.flatnonzero(currencies == currency):
            if len(history) >= warmup and eligible[i]:
                opportunity[i] = values[i] > np.quantile(history[-window:], .75)
            history.append(values[i])
    return cap_opportunities(opportunity, dates, currencies, eligible)


def fit_all(panel, X, announced_X, names, series, outcomes, cap):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    eligible = eligible_next(panel)
    labels = conditional_targets(outcomes)
    scale = effective_scale(announced_X, names)
    margins = margin_targets(series, panel, scale)
    known_buffer = announced_X[:, names.index('known_change_z')]
    predictions = {family: np.full((len(panel), 5), np.nan) for family in FAMILIES}
    raw_margins = {family: np.full((len(panel), 4), np.nan)
                   for family in ('margin_q25', 'margin_ridge_local')}
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
        assert np.isfinite(labels[train]).all() and np.isfinite(margins[train]).all()
        hist = fit_classifier(X, labels, train, query, 'hist')
        age = np.array([(origin - d).days for d in dates], dtype=float)
        weights = np.exp2(-age / 730.)
        recent = fit_classifier(X, labels, train, query, 'hist', weights)
        logit = fit_classifier(X, labels, train, query, 'logit')
        local, local_counts = fit_local_logit(X, labels, train, query, currencies, logit)
        hazard_hist, risk_h, failure_h, risk_rows_h, risk_int_h = fit_hazard(X, labels, train, query, 'hist')
        hazard_logit, risk_l, failure_l, risk_rows_l, risk_int_l = fit_hazard(X, labels, train, query, 'logit')
        qmargin, _ = fit_margin(X, margins, train, query, 'q25', currencies)
        rmargin, ridge_counts = fit_margin(X, margins, train, query, 'ridge_local', currencies)
        conditional = {'hist': hist, 'recent_hist': recent, 'logit': logit,
            'local_logit': local, 'hazard_hist': hazard_hist, 'hazard_logit': hazard_logit}
        query_eligible = eligible[query]
        for family, pred in conditional.items():
            predictions[family][query] = full_curve(pred, query_eligible)
        for family, pred in (('margin_q25', qmargin), ('margin_ridge_local', rmargin)):
            raw_margins[family][query] = pred
            predictions[family][query] = margin_scores(pred, known_buffer[query], query_eligible)
        query_ids = np.flatnonzero(query)
        for family in FAMILIES:
            log = {'origin': str(origin), 'family': family, 'n_shared_train': int(shared.sum()),
                'n_eligible_train': int(train.sum()), 'n_query': int(query.sum()),
                'shared_mask_sha256': hashlib.sha256(np.packbits(shared).tobytes()).hexdigest(),
                'eligible_mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
                'last_shared_mature20': str(max(cap['mature20'][shared])),
                'last_effective_mature20': str(max(outcomes['mature20'][shared])),
                'query_first': int(query_ids[0]), 'query_last': int(query_ids[-1])}
            if family == 'hazard_hist':
                log.update(n_risk=risk_h, n_failures=failure_h,
                           risk_hash=hashlib.sha256(np.column_stack([risk_rows_h, risk_int_h]).astype(np.int32).tobytes()).hexdigest())
            elif family == 'hazard_logit':
                log.update(n_risk=risk_l, n_failures=failure_l,
                           risk_hash=hashlib.sha256(np.column_stack([risk_rows_l, risk_int_l]).astype(np.int32).tobytes()).hexdigest())
            if family == 'local_logit':
                log['local_counts'] = json.dumps(local_counts, sort_keys=True)
            if family == 'margin_ridge_local':
                log['local_counts'] = json.dumps(ridge_counts, sort_keys=True)
            logs.append(log)
        print(f'AP11 {origin}: shared {shared.sum()}, eligible {train.sum()}, 8 families', flush=True)
    return predictions, raw_margins, logs, labels, margins, scale


def policies(panel, predictions, old):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    eligible = eligible_next(panel)
    known_z = old['score__known_change_z']
    raw, signals = {}, {}
    for family, curve in predictions.items():
        for suffix, score in (('h5', curve[:, 2]), ('mean', curve.mean(axis=1))):
            key = family + '_' + suffix
            raw[key] = score
            signals[key + '_urgent_cap2'] = sequential_policy(score, dates, currencies, 'urgent_cap2', gate=eligible)
    known_rank = past_percentiles(known_z, currencies)
    for family in ('hist', 'hazard_hist'):
        model_rank = past_percentiles(predictions[family][:, 2], currencies)
        score = .75 * known_rank + .25 * model_rank
        key = 'known75_' + family + '25_h5'
        raw[key] = score
        signals[key + '_urgent_cap2'] = sequential_policy(score, dates, currencies, 'urgent_cap2', gate=eligible)
    for key in (ANCHOR, SIMPLE, 'market_hist_gated_urgent_cap2',
                'market_survival_h5_gated_urgent_cap2'):
        signals[key] = old['signal__' + key]
    exact = old_ap1_signal(panel)
    signals[AP1_EXACT] = exact
    signals[AP1_EXACT + '_cap2'] = cap_opportunities(exact, dates, currencies, eligible)
    signals['known_change_z_r25w250_cap2'] = fixed_rank_cap2(known_z, dates, currencies, eligible)
    assert len(signals) == 25
    for key, signal in signals.items():
        if key != AP1_EXACT:
            assert not signal[~eligible].any()
    return raw, signals


def select(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, early, groups)])
    boot = benefit_bootstrap(panel, outcomes, signals, early)
    summary = summaries(frame)
    summary['min_benefit_lower_ci'] = boot.groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[k], early) for k in summary.index]
    forward = frame.pivot(index='candidate', columns='h', values='forward_bps')
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    summary['joint_early_pass'] = ((summary.min_lift >= 1.3) & (summary.min_rate >= 1)
        & (summary.max_rate <= 2) & (summary.min_benefit_lower_ci > 0)
        & (summary.max_weekly_signals <= 2) & (summary.early_forward_ratio_min >= .8))
    feasible = summary[summary.joint_early_pass].sort_values(['min_lift', 'mean_lift'], ascending=False, kind='stable')
    selected = str(feasible.index[0]) if len(feasible) else ANCHOR
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    boot.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    selection = {'selected': selected, 'selected_simple': SIMPLE,
        'incumbent': ANCHOR, 'old_exact': AP1_EXACT,
        'selection_year': 2023, 'joint_early_pass_count': len(feasible),
        'selected_before_later_scorecard': True, 'reference': 'today-effective CBR',
        'h1_known_after_receipt': True, 'fresh_holdout': False}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    with np.load(BASE / 'outputs.npz') as z:
        old = {key: z[key] for key in z.files}
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    for h in HORIZONS:
        valid = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~valid] = np.nan
    names = json.loads((OUT.parent / 'ap10_effective' / 'metadata.json').read_text())['cbr_feature_names']
    X, announced_X = old['features__market'], old['features__announced']
    predictions, raw_margins, logs, labels, margins, scale = fit_all(
        panel, X, announced_X, names, series, outcomes, cap)
    raw, signals = policies(panel, predictions, old)
    selection = select(panel, outcomes, signals, old['early'], old['groups'])
    final = pd.DataFrame([{'candidate': key, **row} for key, signal in signals.items()
                          for row in scorecard(panel, outcomes, signal, old['later'], old['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    arrays = {k: old[k] for k in ('dates', 'currencies', 'early', 'later', 'groups')}
    arrays.update({f'curve__{k}': v for k, v in predictions.items()})
    arrays.update({f'margin_prediction__{k}': v for k, v in raw_margins.items()})
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update(conditional_labels=labels, margin_targets=margins, effective_scale=scale,
                  eligible_next=eligible_next(panel), features=X, announced_features=announced_X)
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               AP1 / 'publication' / 'outputs.npz', AP1 / 'announcement_panel.csv',
               Path('research/after_publication_ap11_effective_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({'packet': 'AP11-E', 'n_rows': len(panel),
        'n_families': len(FAMILIES), 'n_policies': len(signals), 'decision': '18:30 MSK',
        'market_delay_minutes': 20, 'reference': 'today-effective CBR',
        'models_learn_unknown_steps': [2, 3, 4, 5, 6, 10, 11, 20],
        'h1_known_after_receipt': True, 'shared_publication_mature20_cap': True,
        'historical_receipts_certified': False, 'bank_execution_validated': False,
        'fresh_holdout': False, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
