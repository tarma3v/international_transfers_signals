"""AP4 frozen first-passage and soft-utility experiments; no late reselection."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries, paired_bootstrap
from research.after_publication_ap2 import matured_mask, residual_mask, regressor
from research.after_publication_ap2_features import append_features
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap3_policy import past_percentiles, sequential_policy, utility_from_forecast
from research.after_publication_ap4_survival import fit_predict_hazard, restricted_wait_targets
from research.after_publication_panel import build_features, build_outcomes

BASE = Path('results/research/after_publication/ap3')
MARKET = Path('results/research/after_publication/ap2_delay20')
OUT = Path('results/research/after_publication/ap4')
INCUMBENT = 'cny_hist50_urgent_cap2'
MODEL_NAMES = ('hazard_logit_h5', 'hazard_logit_mean', 'hazard_hist_h5', 'hazard_hist_mean',
               'hazard_local_h5', 'hazard_local_mean', 'restricted_wait_log',
               'local_hazard_residual25', 'local_hazard_residual50')


def fit_models(panel, X, names, outcomes, wait):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    y = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
    scores = {key: np.full(len(panel), np.nan) for key in MODEL_NAMES}
    curves = {key: np.full_like(y, np.nan) for key in ('logit', 'hist', 'local')}
    origins = np.full(len(panel), dt.date.max, dtype=object)
    local_cols = [i for i, n in enumerate(names) if n.startswith(('announced_', 'effective_',
                  'known_', 'market_', 'dow_', 'annual_')) or n in ('pre_new_year14', 'month_end', 'after2022')]
    logs = []
    for year in range(2022, max(d.year for d in dates) + 1):
        for month in (1, 4, 7, 10):
            origin = dt.date(year, month, 1)
            if origin < dt.date(2022, 7, 1):
                continue
            end = dt.date(year + 1, 1, 1) if month == 10 else dt.date(year, month + 3, 1)
            te = (dates >= origin) & (dates < end)
            if not te.any():
                continue
            tr = matured_mask(panel, outcomes, origin)
            counts = {}
            for kind in ('logit', 'hist'):
                pred, n_risk, n_failure = fit_predict_hazard(X[tr], y[tr], X[te], kind, 400)
                curves[kind][te] = pred
                counts.update(n_at_risk=n_risk, n_failures=n_failure)
            for c in CORRIDORS:
                train, test = tr & (currencies == c), te & (currencies == c)
                if not test.any():
                    continue
                pred, _, _ = fit_predict_hazard(X[train][:, local_cols], y[train],
                                                X[test][:, local_cols], 'logit', 60)
                curves['local'][test] = pred
                origins[test] = origin
            for kind in curves:
                scores[f'hazard_{kind}_h5'][te] = curves[kind][te, 2]
                scores[f'hazard_{kind}_mean'][te] = curves[kind][te].mean(axis=1)
            assert np.isfinite(wait[tr]).all()
            if tr.sum() < 400:
                pred = np.repeat(np.log1p(wait[tr]).mean() if tr.any() else 0., te.sum())
            else:
                model = regressor()
                model.fit(X[tr], np.log1p(wait[tr]))
                pred = model.predict(X[te])
            scores['restricted_wait_log'][te] = pred
            rr = residual_mask(panel, outcomes, origin, scores['hazard_local_mean'], origins)
            correction = np.zeros(te.sum())
            if rr.sum() >= 200:
                model = regressor()
                model.fit(X[rr], y[rr].mean(axis=1) - scores['hazard_local_mean'][rr])
                correction = model.predict(X[te])
            for weight in (.25, .5):
                scores[f'local_hazard_residual{int(weight * 100)}'][te] = scores['hazard_local_mean'][te] + weight * correction
            logs.append({'origin': str(origin), 'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                'last_train_mature20': str(max(outcomes['mature20'][tr])) if tr.any() else None,
                'n_residual_train': int(rr.sum()),
                'last_residual_mature20': str(max(outcomes['mature20'][rr])) if rr.any() else None,
                'last_residual_origin': str(max(origins[rr])) if rr.any() else None, **counts})
            print(f'AP4 {origin}: {tr.sum()} mature events, {counts["n_at_risk"]} risk intervals, {rr.sum()} OOS residuals', flush=True)
    return scores, curves, origins, logs


def build_policies(panel, saved, model_scores):
    cur, dates = panel.currency.to_numpy(), panel.date.to_numpy()
    cdf = lambda v: past_percentiles(v, cur)
    raw = {name: saved['score__' + name] for name in ('cny_last', 'market_hist', 'market_extra')}
    ranks = {name: cdf(value) for name, value in raw.items()}
    for other in ('hist', 'extra'):
        for weight in (.25, .5, .75):
            raw[f'cny_{other}{int(weight * 100)}'] = (1 - weight) * ranks['cny_last'] + weight * ranks['market_' + other]
    raw['cny_hist_extra_equal'] = np.mean(list(ranks.values()), axis=0)
    for name, values in model_scores.items():
        raw[name] = values
        raw['cny50_' + name] = .5 * ranks['cny_last'] + .5 * cdf(values)
    scale = saved['scale']
    forecast = saved['future_mean_forecast']
    predicted = saved['predicted_symmetric_benefit'] / scale[:, None]
    known = utility_from_forecast(np.zeros_like(forecast), scale, saved['known_past_sums']) / scale[:, None]
    utility = {'past_h5': known[:, 2], 'past_all': known.min(axis=1),
               'pred_h5': predicted[:, 2], 'pred_all': predicted.min(axis=1),
               'future_h5': forecast[:, 2], 'future_mean': forecast.mean(axis=1)}
    for name, values in utility.items():
        rank = cdf(values)
        for weight in (.1, .25, .5):
            raw[f'soft_{name}_w{int(weight * 100)}'] = (1 - weight) * raw['cny_hist50'] + weight * rank
    signals = {name + '_urgent_cap2': sequential_policy(values, dates, cur, 'urgent_cap2')
               for name, values in raw.items()}
    assert len(signals) == 46
    np.testing.assert_array_equal(signals[INCUMBENT], saved['signal__' + INCUMBENT])
    return raw, signals, utility


def choose_early(panel, outcomes, signals, early, groups, output=OUT):
    frame = pd.DataFrame([{'candidate': key, **row} for key, fired in signals.items()
                          for row in scorecard(panel, outcomes, fired, early, groups)])
    boot = benefit_bootstrap(panel, outcomes, signals, early)
    summary = summaries(frame)
    summary['min_benefit_lower_ci'] = boot.groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[k], early) for k in summary.index]
    forward = frame.pivot(index='candidate', columns='h', values='forward_bps')
    assert (forward.loc[INCUMBENT] > 0).all()
    summary['early_forward_ratio_min'] = (forward / forward.loc[INCUMBENT]).min(axis=1)
    summary['selection_value'] = summary.min_lift - 2 * summary.cadence_penalty - (-summary.min_benefit_lower_ci / 50).clip(lower=0)
    summary['joint_early_pass'] = ((summary.min_lift >= 1.3) & (summary.min_rate >= 1) &
        (summary.max_rate <= 2) & (summary.min_benefit_lower_ci > 0) &
        (summary.max_weekly_signals <= 2) & (summary.early_forward_ratio_min >= .8))
    ranked = summary.sort_values(['selection_value', 'mean_lift'], ascending=False, kind='stable')
    feasible = ranked[ranked.joint_early_pass]
    selected = feasible.index[0] if len(feasible) else INCUMBENT
    frame.to_csv(output / 'early_all_horizons.csv', index=False)
    boot.to_csv(output / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(output / 'early_summary.csv')
    selection = {'selected': selected, 'selected_simple': INCUMBENT, 'incumbent': INCUMBENT,
        'selection_year': 2023, 'allh_mature_before': '2024-01-01',
        'joint_early_pass_count': int(len(feasible)), 'selected_before_later_scorecard': True,
        'historical_receipts_certified': False, 'fresh_holdout': False}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, MARKET):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    series = load(DATA)
    panel, X, names = build_features(series)
    keep = panel.date.to_numpy() >= dt.date(2022, 1, 1)
    panel, X = panel[keep].reset_index(drop=True), X[keep]
    previous = pd.read_csv(BASE / 'announcement_panel.csv')
    market = pd.read_csv(BASE / 'market_panel.csv')
    market.date = pd.to_datetime(market.date).dt.date
    assert list(zip(panel.currency, panel.date)) == list(zip(previous.currency, pd.to_datetime(previous.date).dt.date))
    np.testing.assert_allclose(panel.announced_price, previous.announced_price, rtol=1e-12)
    panel['decision_at'] = market.decision_at
    X, names = append_features(X, names, market)
    assert names == json.loads((MARKET / 'feature_names.json').read_text())
    outcomes = build_outcomes(series, panel, 'publication')
    wait = restricted_wait_targets(series, panel)
    with np.load(BASE / 'outputs.npz') as saved:
        for h in HORIZONS:
            np.testing.assert_array_equal(outcomes[f'y{h}'], saved[f'y{h}'])
            valid = np.isfinite(wait)
            np.testing.assert_array_equal((wait[valid] > h).astype(float), outcomes[f'y{h}'][valid])
        scores, curves, origins, logs = fit_models(panel, X, names, outcomes, wait)
        raw, signals, utility = build_policies(panel, saved, scores)
        common = np.logical_and.reduce([np.isfinite(v) for v in raw.values()])
        early, later, groups = saved['early'], saved['later'], saved['groups']
        assert common[early | later].all(), 'Do not change the comparison date support'
    selection = choose_early(panel, outcomes, signals, early, groups)
    final = pd.DataFrame([{'candidate': key, **row} for key, fired in signals.items()
                          for row in scorecard(panel, outcomes, fired, later, groups)])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    focus = list(dict.fromkeys([selection['selected'], INCUMBENT, 'soft_past_all_w25_urgent_cap2',
        'soft_pred_all_w25_urgent_cap2', 'soft_future_mean_w25_urgent_cap2',
        'hazard_hist_mean_urgent_cap2', 'local_hazard_residual50_urgent_cap2']))
    paired_bootstrap(panel, outcomes, signals, later, groups, focus, INCUMBENT).to_csv(OUT / 'paired_vs_ap3.csv', index=False)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    market.to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    arrays = {'dates': np.array([str(d) for d in panel.date]), 'currencies': panel.currency.to_numpy().astype(str),
        'groups': groups, 'early': early, 'later': later, 'restricted_wait': wait,
        'anchor_origins': np.array([str(d) for d in origins])}
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({f'survival__{k}': v for k, v in curves.items()})
    arrays.update({f'utility__{k}': v for k, v in utility.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    paths = [DATA, BASE / 'outputs.npz', BASE / 'metadata.json', Path('research/after_publication_ap4_registered.md')]
    metadata = {'packet': 'AP4', 'n_rows': len(panel), 'n_features': len(names), 'n_policies': len(signals),
        'calendar_assumed': True, 'market_delay_minutes': 20, 'decision': '18:30 MSK',
        'reference': 'latest announced CBR', 'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(final[(final.h == 5) & final.candidate.isin(focus)].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
