"""AP10-E: refit today-effective targets with/without received next fixing."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, factory, scorecard, sign_policy, summaries
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap3 import benefit_bootstrap, weekly_max
from research.after_publication_ap3_policy import sequential_policy
from research.after_publication_ap4 import INCUMBENT
from research.after_publication_ap4_survival import fit_predict_hazard
from research.after_publication_panel import REFERENCES, history_features, build_features, build_outcomes

OUT = Path('results/research/after_publication/ap10_effective')
BASE = OUT.parent / 'ap8'
SIMPLE = 'known_next_not_lower_cd3'
OUTPUTS = ('hist', 'logit', 'survival_h5', 'survival_mean')


def past_features(series, panel, names):
    """Independent effective prefixes for ALL currencies, not dropped columns."""
    records = []
    for row in panel.itertuples():
        day = row.date
        own = series[row.currency]
        current = int(np.searchsorted(own.dates, day, side='right')) - 1
        past = history_features(own.values[:current + 1])
        f = {prefix + name: value for prefix in ('announced_', 'effective_') for name, value in past.items()}
        f.update(known_change=0., known_change_z=0.,
            effective_age_days=float((day - own.dates[current]).days), next_effective_gap=0.,
            dow_sin=float(np.sin(2 * np.pi * day.weekday() / 7)),
            dow_cos=float(np.cos(2 * np.pi * day.weekday() / 7)),
            annual_sin=float(np.sin(2 * np.pi * (day.timetuple().tm_yday - 1) / 365.25)),
            annual_cos=float(np.cos(2 * np.pi * (day.timetuple().tm_yday - 1) / 365.25)),
            pre_new_year14=float(day.month == 12 and day.day >= 17), month_end=float(day.day >= 24),
            after2022=float(day >= dt.date(2022, 2, 24)))
        for c in CORRIDORS:
            f['currency_' + c] = float(c == row.currency)
        changes = []
        for c in CORRIDORS + REFERENCES:
            s = series[c]
            j = int(np.searchsorted(s.dates, day, side='right')) - 1
            f[c + '_missing'] = float(j < 1)
            if j < 1:
                for lag in (1, 5, 20):
                    f[c + f'_ret{lag}'] = 0.
                f[c + '_source_age_days'] = 365.
                continue
            for lag in (1, 5, 20):
                f[c + f'_ret{lag}'] = float(1e4 * np.log(s.values[j] / s.values[max(0, j - lag)]))
            # Same18:00 CBR snapshot as announced features; no unreceived price.
            f[c + '_source_age_days'] = float((day - s.dates[j]).days + 1)
            if c in CORRIDORS:
                changes.append(f[c + '_ret1'])
        f['peer_change_mean'] = float(np.mean(changes)) if changes else 0.
        f['peer_change_std'] = float(np.std(changes)) if changes else 0.
        f['local_minus_common'] = -f['peer_change_mean']
        assert set(f) == set(names)
        records.append([f[n] for n in names])
    result = np.array(records, dtype=float)
    assert np.isfinite(result).all()
    return result


def fit_models(panel, matrices, outcomes, training_cap, origins=None):
    dates = panel.date.to_numpy()
    if origins is None:
        origins = [dt.date(y, m, 1) for y in range(2022, max(d.year for d in dates) + 1)
                   for m in (1, 4, 7, 10) if dt.date(y, m, 1) >= dt.date(2022, 7, 1)]
    scores = {f'{info}_{model}': np.full(len(panel), np.nan) for info in matrices for model in OUTPUTS}
    curves = {info: np.full((len(panel), 5), np.nan) for info in matrices}
    logs = []
    y = outcomes['y5']
    survival = np.column_stack([outcomes['y' + str(h)] for h in HORIZONS])
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        query = (dates >= origin) & (dates < end)
        if not query.any():
            continue
        train = matured_mask(panel, training_cap, origin)
        assert (outcomes['mature20'][train] < origin - dt.timedelta(days=2)).all()
        for info, X in matrices.items():
            for model_name in ('hist', 'logit'):
                model = (factory('hist7y').set_params(early_stopping=False) if model_name == 'hist'
                         else make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1500)))
                if train.sum() < 400 or np.unique(y[train]).size < 2:
                    pred = np.repeat(y[train].mean() if train.any() else 0., query.sum())
                else:
                    model.fit(X[train], y[train])
                    pred = model.predict_proba(X[query])[:, 1]
                scores[info + '_' + model_name][query] = pred
            curves[info][query], n_risk, n_failure = fit_predict_hazard(X[train], survival[train], X[query], 'hist', 400)
            scores[info + '_survival_h5'][query] = curves[info][query, 2]
            scores[info + '_survival_mean'][query] = curves[info][query].mean(axis=1)
            logs.append({'origin': str(origin), 'information': info, 'n_features': X.shape[1],
                'n_train': int(train.sum()), 'n_query': int(query.sum()),
                'mask_sha256': hashlib.sha256(np.packbits(train).tobytes()).hexdigest(),
                'last_training_date': str(max(dates[train])),
                'last_effective_mature20': str(max(outcomes['mature20'][train])),
                'last_shared_cap': str(max(training_cap['mature20'][train])),
                'n_at_risk': n_risk, 'n_failures': n_failure})
        print(f'AP10-E {origin}: {train.sum()} identical train rows, all three information sets fit', flush=True)
    return scores, curves, logs


def policies(panel, scores, announced_X, names, previous):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    eligible = panel.announced_price.to_numpy() >= panel.current_price.to_numpy()
    raw, signals = dict(scores), {}
    for name, value in scores.items():
        signals[name + '_urgent_cap2'] = sequential_policy(value, dates, currencies, 'urgent_cap2')
        if name.startswith(('announced_', 'market_')):
            raw[name + '_gated'] = value
            signals[name + '_gated_urgent_cap2'] = sequential_policy(value, dates, currencies, 'urgent_cap2', gate=eligible)
    raw['known_change_z'] = announced_X[:, names.index('known_change_z')]
    signals['known_change_z_urgent_cap2'] = sequential_policy(raw['known_change_z'], dates, currencies, 'urgent_cap2', gate=eligible)
    signals[SIMPLE] = sign_policy(panel.announced_price.to_numpy() - panel.current_price.to_numpy(), dates, currencies, 3)
    signals[INCUMBENT] = previous['signal__' + INCUMBENT]
    assert len(signals) == 23
    for key, value in signals.items():
        if '_gated_' in key or key in (SIMPLE, 'known_change_z_urgent_cap2'):
            assert not value[~eligible].any()
    return raw, signals


def select_early(panel, outcomes, signals, early, groups):
    frame = pd.DataFrame([{'candidate': k, **r} for k, value in signals.items()
                          for r in scorecard(panel, outcomes, value, early, groups)])
    boot = benefit_bootstrap(panel, outcomes, signals, early)
    summary = summaries(frame)
    summary['min_benefit_lower_ci'] = boot.groupby('candidate').ci_lo.min()
    summary['max_weekly_signals'] = [weekly_max(panel, signals[k], early) for k in summary.index]
    forward = frame.pivot(index='candidate', columns='h', values='forward_bps')
    assert (forward.loc[SIMPLE] > 0).all()
    summary['early_forward_ratio_min'] = (forward / forward.loc[SIMPLE]).min(axis=1)
    summary['joint_early_pass'] = ((summary.min_lift >= 1.3) & (summary.min_rate >= 1)
        & (summary.max_rate <= 2) & (summary.min_benefit_lower_ci > 0)
        & (summary.max_weekly_signals <= 2) & (summary.early_forward_ratio_min >= .8))
    ranked = summary[summary.joint_early_pass & (summary.index != INCUMBENT)].sort_values(['min_lift', 'mean_lift'], ascending=False, kind='stable')
    selected = str(ranked.index[0]) if len(ranked) else SIMPLE
    selection = {'selected': selected, 'selected_simple': SIMPLE, 'reference': 'today-effective CBR',
        'selection_year': 2023, 'n_early_pass': len(ranked), 'selected_before_later_scorecard': True,
        'h1_is_already_known_after_receipt': True, 'fresh_holdout': False}
    frame.to_csv(OUT / 'early_all_horizons.csv', index=False)
    boot.to_csv(OUT / 'early_benefit_uncertainty.csv', index=False)
    summary.to_csv(OUT / 'early_summary.csv')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    return selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, digest in json.loads((BASE / 'metadata.json').read_text())['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    series = load(DATA)
    panel, announced_X, names = build_features(series)
    keep = panel.date.to_numpy() >= dt.date(2022, 1, 1)
    panel, announced_X = panel[keep].reset_index(drop=True), announced_X[keep]
    with np.load(BASE / 'outputs.npz') as z:
        previous = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(panel.date.astype(str), previous['dates'])
    np.testing.assert_array_equal(panel.currency, previous['currencies'])
    np.testing.assert_array_equal(announced_X, previous['features__1830'][:, :len(names)])
    matrices = {'past': past_features(series, panel, names), 'announced': announced_X,
                'market': previous['features__1830']}
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    scores, curves, logs = fit_models(panel, matrices, outcomes, cap)
    raw, signals = policies(panel, scores, announced_X, names, previous)
    early, later, groups = [previous[k] for k in ('early', 'later', 'groups')]
    # Both information sets and both reporting references use the same old support.
    for h in HORIZONS:
        available = np.isfinite(cap['y' + str(h)])
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~available] = np.nan
    assert np.logical_and.reduce([np.isfinite(v) for v in scores.values()])[early | later].all()
    selection = select_early(panel, outcomes, signals, early, groups)
    final = pd.DataFrame([{'candidate': k, **r} for k, value in signals.items()
                          for r in scorecard(panel, outcomes, value, later, groups)])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    pub = pd.DataFrame([{'candidate': k, **r} for k, value in signals.items()
                        for r in scorecard(panel, cap, value, later, groups)])
    pub.to_csv(OUT / 'publication_reference_diagnostic.csv', index=False)
    arrays = {k: previous[k] for k in ('dates', 'currencies', 'early', 'later', 'groups')}
    arrays.update({f'features__{k}': v for k, v in matrices.items()})
    arrays.update({f'prediction__{k}': v for k, v in scores.items()})
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({f'survival__{k}': v for k, v in curves.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel['decision_at'] = pd.read_csv(BASE / 'announcement_panel.csv').decision_at
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               Path('research/after_publication_effective_information_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({'packet': 'AP10-E', 'n_rows': len(panel),
        'n_model_variants': 9, 'n_policies': len(signals), 'cbr_feature_names': names,
        'reference': 'today-effective CBR', 'same_shared_publication_mature20_train_cap': True,
        'calendar_assumed': True, 'historical_receipts_certified': False, 'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
