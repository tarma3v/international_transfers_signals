"""AP9: resolved short feedback and observation-step censoring at18:30 only."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries
from research.after_publication_ap3_policy import past_percentiles, sequential_policy
from research.after_publication_ap4 import INCUMBENT, choose_early
from research.after_publication_ap5 import AP4_CONTROL
from research.after_publication_ap9_censoring import KINDS, observed_followup, fit_one
from research.after_publication_panel import build_outcomes

OUT = Path('results/research/after_publication/ap9')
BASE = OUT.parent / 'ap8'


def receipt_at_steps(series, panel, rows, steps):
    return np.array([series[panel.currency.iloc[i]].dates[int(panel.announced_index.iloc[i]) + int(h)]
                     - dt.timedelta(days=1) for i, h in zip(rows, steps)], dtype=object)


def fit_all(panel, X, hx, series):
    dates = panel.date.to_numpy()
    outputs, logs, followup = {}, [], {}
    for cadence, freq in (('quarter', 'Q'), ('month', 'M')):
        curves = {k: np.full((len(panel), 1 if k.startswith('direct') else 5 if k.startswith('coarse') else 20), np.nan) for k in KINDS}
        periods = pd.period_range('2022-07', pd.Timestamp(max(dates)).to_period(freq), freq=freq)
        for period in periods:
            origin, end = period.start_time.date(), (period + 1).start_time.date()
            query = np.flatnonzero((dates >= origin) & (dates < end))
            if not len(query):
                continue
            if origin not in followup:
                followup[origin] = observed_followup(series, panel, origin)
            counts, failures = followup[origin]
            for kind in KINDS:
                matrix = X if kind.startswith('direct') else hx
                curve, rows, intervals, target, endpoints, summary = fit_one(matrix, counts, failures, kind, query)
                curves[kind][query] = curve
                receipts = receipt_at_steps(series, panel, rows, endpoints[intervals])
                assert (receipts < origin - dt.timedelta(days=2)).all()
                logs.append({'cadence': cadence, 'origin': str(origin), 'kind': kind, **summary,
                    'n_query': len(query), 'last_event_date': str(max(dates[rows])),
                    'last_label_receipt': str(max(receipts)),
                    'n_full20_available': int((counts == 20).sum()),
                    'n_observed_events': int((counts > 0).sum()),
                    'n_observed_failures': int((failures > 0).sum())})
            print(f'AP9 {cadence} {origin}: full20 {(counts == 20).sum()}, observed {(counts > 0).sum()}', flush=True)
        outputs.update({cadence + '_' + key: value for key, value in curves.items()})
    return outputs, logs, followup


def policies(panel, forecasts, previous):
    raw = {key.removesuffix('_urgent_cap2'): previous['score__' + key.removesuffix('_urgent_cap2')]
           for key in (INCUMBENT, AP4_CONTROL)}
    cur = panel.currency.to_numpy()
    cny = past_percentiles(previous['score__t1830_cny'], cur)
    for name, curve in forecasts.items():
        direct = '_direct_' in name
        hcurve = curve if curve.shape[1] != 20 else curve[:, np.array(HORIZONS) - 1]
        p = hcurve[:, 0 if direct else 2]
        raw[name + '_h5'] = p
        if not direct:
            raw[name + '_mean'] = hcurve.mean(axis=1)
        raw['cny50_' + name] = .5 * cny + .5 * past_percentiles(p, cur)
    signals = {key + '_urgent_cap2': sequential_policy(value, panel.date.to_numpy(), cur, 'urgent_cap2') for key, value in raw.items()}
    assert len(signals) == 34
    for key in (INCUMBENT, AP4_CONTROL):
        np.testing.assert_array_equal(signals[key], previous['signal__' + key])
    return raw, signals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, digest in json.loads((BASE / 'metadata.json').read_text())['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'publication')
    with np.load(BASE / 'outputs.npz') as packet:
        previous = {key: packet[key] for key in packet.files}
    X, hx = previous['features__1830'], previous['hazard_features__1830']
    forecasts, logs, followup = fit_all(panel, X, hx, series)
    np.testing.assert_array_equal(forecasts['quarter_direct_full20'][:, 0], previous['hist__1830'])
    np.testing.assert_array_equal(forecasts['quarter_coarse_full20'], previous['survival__1830'])
    raw, signals = policies(panel, forecasts, previous)
    early, later, groups = [previous[key] for key in ('early', 'later', 'groups')]
    assert np.logical_and.reduce([np.isfinite(v) for v in raw.values()])[early | later].all()
    selection = choose_early(panel, outcomes, signals, early, groups, output=OUT)
    final = pd.DataFrame([{'candidate': key, **row} for key, fired in signals.items()
                          for row in scorecard(panel, outcomes, fired, later, groups)])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    arrays = {key: previous[key] for key in ('dates', 'currencies', 'early', 'later', 'groups')}
    arrays.update(features=X, hazard_features=hx)
    arrays.update({f'curve__{k}': v for k, v in forecasts.items()})
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    for origin, (counts, failures) in followup.items():
        arrays['observed_count__' + str(origin)] = counts
        arrays['observed_failure__' + str(origin)] = failures
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
               Path('research/after_publication_ap9_registered.md')]
    metadata = {'packet': 'AP9', 'n_rows': len(panel), 'n_models': len(forecasts), 'n_policies': len(signals),
        'decision': '18:30 MSK', 'market_delay_minutes': 20, 'calendar_assumed': True,
        'reference': 'latest announced CBR', 'fresh_holdout': False,
        'bank_execution_validated': False, 'compatibility_full20_quarter_max_error': 0.,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(final[final.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
