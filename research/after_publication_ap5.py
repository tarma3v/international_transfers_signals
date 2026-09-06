"""Frozen AP5: issued OOS calibration, delayed weighting, unchanged signal clock."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries, paired_bootstrap
from research.after_publication_ap3_policy import past_percentiles, sequential_policy
from research.after_publication_ap4 import choose_early, INCUMBENT
from research.after_publication_ap5_learning import calibrate, delayed_weights
from research.after_publication_panel import build_outcomes

BASE = Path('results/research/after_publication/ap4')
AP3 = BASE.parent / 'ap3'
OUT = BASE.parent / 'ap5'
EXPERTS = ('cny', 'hist', 'extra', 'survival_hist', 'survival_logit', 'survival_local')
AP4_CONTROL = 'cny50_hazard_hist_h5_urgent_cap2'


def expert_features(previous, ap3):
    def logit(x):
        p = np.clip(x, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    features = [np.repeat(np.arcsinh(ap3['score__cny_last'])[:, None], 5, axis=1)]
    for kind in ('market_hist', 'market_extra'):
        features.append(np.repeat(logit(ap3['score__' + kind])[:, None], 5, axis=1))
    for kind in ('hist', 'logit', 'local'):
        features.append(logit(previous['survival__' + kind]))
    return np.stack(features, axis=1)


def build_policies(panel, previous, predictions, y, maturity):
    dates, cur = panel.date.to_numpy(), panel.currency.to_numpy()
    cdf = lambda values: past_percentiles(values, cur)
    raw = {INCUMBENT.removesuffix('_urgent_cap2'): previous['score__cny_hist50'],
           AP4_CONTROL.removesuffix('_urgent_cap2'): previous['score__cny50_hazard_hist_h5']}
    curves, weight_arrays, weight_logs = {}, {}, []
    for mode, probability in predictions.items():
        for e, name in enumerate(EXPERTS):
            raw[f'cal_{mode}_{name}'] = probability[:, e, 2]
        equal = probability.mean(axis=1)
        raw[f'equal_{mode}_h5'], raw[f'equal_{mode}_mean'] = equal[:, 2], equal.mean(axis=1)
        cny50 = .5 * probability[:, 0, :] + .5 * probability[:, 1:, :].mean(axis=1)
        raw[f'cny50_probability_{mode}'] = cny50[:, 2]
        curves['equal_' + mode] = equal
        curves['cny50_probability_' + mode] = cny50
    issued = predictions['rolling365']
    cny_rank = cdf(previous['score__cny_last'])
    for local in (False, True):
        for half in (63, 252):
            for eta in (2, 10, 30):
                key = f'hedge_{"local" if local else "global"}_h{half}_e{eta}'
                weights, logs = delayed_weights(panel, issued, y, maturity, half, eta, local)
                weight_arrays[key] = weights
                weight_logs.extend({'candidate': key, **r} for r in logs)
                mixture = np.sum(issued * weights[:, :, None], axis=1)
                curves[key] = mixture
                raw[key] = mixture[:, 2]
                if not local:
                    raw['cny50_' + key] = .5 * cny_rank + .5 * cdf(mixture[:, 2])
                print(f'AP5 {key}: {len(logs)} as-of weight snapshots', flush=True)
    weights, logs = delayed_weights(panel, issued, y, maturity, 252, 10, freeze=dt.date(2023, 1, 1))
    weight_arrays['frozen_weights2023'] = weights
    weight_logs.extend({'candidate': 'frozen_weights2023', **r} for r in logs)
    curves['frozen_weights2023'] = np.sum(issued * weights[:, :, None], axis=1)
    raw['frozen_weights2023'] = curves['frozen_weights2023'][:, 2]
    signals = {key + '_urgent_cap2': sequential_policy(score, dates, cur, 'urgent_cap2') for key, score in raw.items()}
    assert len(signals) == 66
    for control in (INCUMBENT, AP4_CONTROL):
        np.testing.assert_array_equal(signals[control], previous['signal__' + control])
    return raw, signals, curves, weight_arrays, weight_logs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, AP3, BASE.parent / 'ap2_delay20'):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    outcomes = build_outcomes(load(DATA), panel, 'publication')
    y = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
    with np.load(BASE / 'outputs.npz') as previous, np.load(AP3 / 'outputs.npz') as ap3:
        np.testing.assert_array_equal(previous['dates'], ap3['dates'])
        for h in HORIZONS:
            np.testing.assert_array_equal(previous[f'y{h}'], outcomes[f'y{h}'])
        features = expert_features(previous, ap3)
        predictions, logs = calibrate(panel, features, y, outcomes['mature20'])
        raw, signals, curves, weights, weight_logs = build_policies(panel, previous, predictions, y, outcomes['mature20'])
        early, later, groups = previous['early'], previous['later'], previous['groups']
        common = np.logical_and.reduce([np.isfinite(v) for v in raw.values()])
        assert common[early | later].all(), 'Original comparison support must remain unchanged'
    selection = choose_early(panel, outcomes, signals, early, groups, output=OUT)
    final = pd.DataFrame([{'candidate': key, **row} for key, fired in signals.items()
                          for row in scorecard(panel, outcomes, fired, later, groups)])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    focus = list(dict.fromkeys([selection['selected'], INCUMBENT, AP4_CONTROL,
        'equal_rolling365_h5_urgent_cap2', 'hedge_global_h252_e10_urgent_cap2',
        'hedge_local_h63_e30_urgent_cap2', 'frozen_weights2023_urgent_cap2']))
    paired_bootstrap(panel, outcomes, signals, later, groups, focus, INCUMBENT).to_csv(OUT / 'paired_vs_ap3.csv', index=False)
    arrays = {'dates': np.array([str(d) for d in panel.date]), 'currencies': panel.currency.to_numpy().astype(str),
              'early': early, 'later': later, 'groups': groups}
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({f'probability__{k}': v for k, v in predictions.items()})
    arrays.update({f'ensemble__{k}': v for k, v in curves.items()})
    arrays.update({f'weights__{k}': v for k, v in weights.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    pd.DataFrame(logs).to_csv(OUT / 'calibration_log.csv', index=False)
    pd.DataFrame(weight_logs).to_csv(OUT / 'weight_log.csv', index=False)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    paths = [DATA, BASE / 'outputs.npz', BASE / 'metadata.json', AP3 / 'outputs.npz',
             Path('research/after_publication_ap5_registered.md')]
    meta = {'packet': 'AP5', 'n_rows': len(panel), 'n_policies': len(signals), 'experts': EXPERTS,
        'calendar_assumed': True, 'decision': '18:30 MSK', 'market_delay_minutes': 20,
        'reference': 'latest announced CBR', 'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(final[(final.h == 5) & final.candidate.isin(focus)].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
