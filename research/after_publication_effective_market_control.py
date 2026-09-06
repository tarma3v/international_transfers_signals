"""Strong same-clock control: available market but no announced fixing input."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA, scorecard, summaries
from research.after_publication_ap2_features import CUTOFF, MOSCOW, append_features, load_market_frames, session_state
from research.after_publication_ap3_policy import sequential_policy
from research.after_publication_panel import build_outcomes
import research.after_publication_effective_information as experiment

OUT = Path('results/research/after_publication/ap10_effective_extended')
BASE = experiment.OUT


def effective_market(panel, series, frames):
    daily = {ticker: {day: f for day, f in frame.groupby(frame.begin.dt.date)} for ticker, frame in frames.items()}
    empty = next(iter(frames.values())).iloc[:0]
    cache, records = {}, []
    def state(ticker, day):
        if (ticker, day) not in cache:
            cache[ticker, day] = session_state(daily.get(ticker, {}).get(day, empty), day, feed_delay_minutes=20)
        return cache[ticker, day]
    for row in panel.itertuples():
        day = row.date
        ci = int(np.searchsorted(series['CNY'].dates, day, side='right')) - 1
        li = int(np.searchsorted(series[row.currency].dates, day, side='right')) - 1
        cvalues = series['CNY'].values[max(0, ci - 20):ci + 1]
        lvalues = series[row.currency].values[max(0, li - 20):li + 1]
        cs = state('CNYRUB_TOM', day)
        ts = state(row.currency + 'RUB_TOM', day)
        term = 'TOM' if ts['n'] else 'TOD'
        ls = ts if ts['n'] else state(row.currency + 'RUB_TOD', day)
        rec = {'date': str(day), 'currency': row.currency, 'decision_at': dt.datetime.combine(day, CUTOFF, MOSCOW).isoformat(),
               'reference_kind': 'effective', 'local_source_term': term}
        for prefix, st, values in (('cny', cs, cvalues), ('local', ls, lvalues)):
            reference = values[-1]
            vol = max(float(np.std(np.diff(np.log(values)) * 1e4)), 1.)
            for part in ('last', 'mean', 'post'):
                value = st[part]
                valid = np.isfinite(value)
                rec[f'{prefix}_basis_{part}_z'] = float(1e4 * np.log(value / reference) / vol) if valid else 0.
                rec[f'{prefix}_{part}_missing'] = float(not valid)
            for field in ('n', 'age', 'quality', 'range'):
                rec[prefix + '_' + field] = st[field]
            rec[prefix + '_late_z'] = st['late'] / vol
            rec[prefix + '_max_source'] = st['max_source']
        records.append(rec)
    return pd.DataFrame(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / 'metadata.json').read_text())
    for path, digest in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    market = effective_market(panel, series, load_market_frames())
    with np.load(BASE / 'outputs.npz') as z:
        saved = {key: z[key] for key in z.files}
    X, names = append_features(saved['features__past'], metadata['cbr_feature_names'], market)
    assert names == json.loads((experiment.BASE / 'metadata.json').read_text())['feature_names']
    outcomes = build_outcomes(series, panel, 'effective')
    cap = build_outcomes(series, panel, 'publication')
    scores, curves, logs = experiment.fit_models(panel, {'past_market': X}, outcomes, cap)
    signals = {k.removeprefix('signal__'): v for k, v in saved.items() if k.startswith('signal__')}
    for name, value in scores.items():
        signals[name + '_urgent_cap2'] = sequential_policy(value, panel.date.to_numpy(), panel.currency.to_numpy(), 'urgent_cap2')
    for h in (1, 3, 5, 10, 20):
        for kind in ('y', 'sym', 'forward', 'floor'):
            outcomes[kind + str(h)][~np.isfinite(cap['y' + str(h)])] = np.nan
    assert len(signals) == 27
    old_out = experiment.OUT
    experiment.OUT = OUT
    try:
        experiment.select_early(panel, outcomes, signals, saved['early'], saved['groups'])
    finally:
        experiment.OUT = old_out
    final = pd.DataFrame([{'candidate': k, **r} for k, value in signals.items()
                          for r in scorecard(panel, outcomes, value, saved['later'], saved['groups'])])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    saved['features__past_market'] = X
    for key, value in scores.items():
        saved['prediction__' + key] = value
        saved['score__' + key] = value
        saved['signal__' + key + '_urgent_cap2'] = signals[key + '_urgent_cap2']
    saved.update({'survival__' + k: v for k, v in curves.items()})
    np.savez_compressed(OUT / 'outputs.npz', **saved)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    market.to_csv(OUT / 'effective_market_panel.csv', index=False)
    pd.concat([pd.read_csv(BASE / 'training_log.csv'), pd.DataFrame(logs)]).to_csv(OUT / 'training_log.csv', index=False)
    pd.DataFrame([{'candidate': k, **r} for k, value in signals.items() for r in
                  scorecard(panel, cap, value, saved['later'], saved['groups'])]).to_csv(OUT / 'publication_reference_diagnostic.csv', index=False)
    sources = [BASE / 'metadata.json', BASE / 'outputs.npz', Path('research/after_publication_effective_market_control_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({'packet': 'AP10-E extended', 'n_models': 12,
        'n_policies': 27, 'reference': 'today-effective CBR', 'n_rows': len(panel),
        'frozen_previous_predictions': True, 'same_clock_information_withholding_not_before_release': True,
        'fresh_holdout': False, 'historical_receipts_certified': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
