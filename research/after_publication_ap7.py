"""AP7 normalized trajectory analogs; no new expert-stack weight grid."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries, paired_bootstrap
from research.after_publication_ap3_policy import past_percentiles, sequential_policy, known_past_sums
from research.after_publication_ap4 import INCUMBENT, choose_early
from research.after_publication_ap5 import AP4_CONTROL
from research.after_publication_ap6 import EQUAL_CONTROL
from research.after_publication_ap7_paths import normalized_paths, forecast_paths
from research.after_publication_panel import build_outcomes

BASE = Path('results/research/after_publication/ap6')
AP3, OUT = BASE.parent / 'ap3', BASE.parent / 'ap7'
AP6_CONTROL = 'ap4_w25_stack_local_residual_urgent_cap2'
CONTROLS = (INCUMBENT, AP4_CONTROL, EQUAL_CONTROL, AP6_CONTROL)


def core_features(wide, names):
    get = lambda n: wide[:, names.index(n)]
    cny = np.where(get('cny_last_missing') == 0, get('cny_basis_last_z'), get('known_change_z'))
    core_names = ['cny_z', 'known_change_z', 'announced_ret5_z', 'announced_ret20_z',
                  'log_vol20', 'cny_late_z', 'cny_quality']
    core = np.column_stack([np.arcsinh(cny), np.arcsinh(get('known_change_z')),
        np.arcsinh(get('announced_ret5_z')), np.arcsinh(get('announced_ret20_z')),
        get('log_vol20'), np.arcsinh(get('cny_late_z')), get('cny_quality')])
    return core, core_names, np.maximum(np.expm1(get('log_vol20')), 1.)


def build_policies(panel, previous, forecasts, scale, cny_score):
    cur, dates = panel.currency.to_numpy(), panel.date.to_numpy()
    cdf = lambda v: past_percentiles(v, cur)
    raw = {k.removesuffix('_urgent_cap2'): previous['score__' + k.removesuffix('_urgent_cap2')] for k in CONTROLS}
    ap4 = raw[AP4_CONTROL.removesuffix('_urgent_cap2')]
    cny = cdf(cny_score)
    for key, value in forecasts.items():
        probability = value['probability']
        rank = cdf(probability[:, 2])
        raw['path_' + key + '_h5'] = probability[:, 2]
        raw['path_' + key + '_mean'] = probability.mean(axis=1)
        raw['path_' + key + '_forward25'] = .75 * rank + .25 * cdf(value['forward'][:, 2] / scale)
        raw['path_' + key + '_symmetric25'] = .75 * rank + .25 * cdf((value['symmetric'] / scale[:, None]).min(axis=1))
        raw['cny50_path_' + key] = .5 * cny + .5 * rank
        raw['ap4_w25_path_' + key] = .75 * ap4 + .25 * rank
    signals = {k + '_urgent_cap2': sequential_policy(v, dates, cur, 'urgent_cap2') for k, v in raw.items()}
    assert len(signals) == 70
    for key in CONTROLS:
        np.testing.assert_array_equal(signals[key], previous['signal__' + key])
    return raw, signals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, BASE.parent / 'ap5', BASE.parent / 'ap4', AP3, BASE.parent / 'ap2_delay20'):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    series = load(DATA)
    outcomes = build_outcomes(series, panel, 'publication')
    names = json.loads((BASE / 'metadata.json').read_text())['source_context_names']
    past = known_past_sums(series, panel)
    with np.load(BASE / 'outputs.npz') as previous, np.load(AP3 / 'outputs.npz') as ap3:
        wide = previous['source_context']
        core, core_names, scale = core_features(wide, names)
        paths = normalized_paths(series, panel, scale)
        complete = np.isfinite(paths).all(axis=1)
        for h in HORIZONS:
            np.testing.assert_array_equal((paths[complete, :h].min(axis=1) >= 0).astype(float), outcomes[f'y{h}'][complete])
            np.testing.assert_array_equal(previous[f'y{h}'], outcomes[f'y{h}'])
        forecasts, logs, scenario_logs, neighbors, weights = forecast_paths(
            panel, core, wide, paths, scale, past, outcomes['mature20'])
        raw, signals = build_policies(panel, previous, forecasts, scale, ap3['score__cny_last'])
        early, later, groups = previous['early'], previous['later'], previous['groups']
        assert np.logical_and.reduce([np.isfinite(v) for v in raw.values()])[early | later].all()
    selection = choose_early(panel, outcomes, signals, early, groups, output=OUT)
    frame = pd.DataFrame([{'candidate': k, **r} for k, fired in signals.items()
                          for r in scorecard(panel, outcomes, fired, later, groups)])
    frame.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(frame).to_csv(OUT / 'retrospective_summary.csv')
    focus = list(dict.fromkeys([selection['selected'], *CONTROLS,
        *['path_' + k + '_h5_urgent_cap2' for k in forecasts]]))
    paired_bootstrap(panel, outcomes, signals, later, groups, focus, INCUMBENT).to_csv(OUT / 'paired_vs_ap3.csv', index=False)
    arrays = {'dates': panel.date.astype(str).to_numpy().astype(str), 'currencies': panel.currency.to_numpy().astype(str),
        'early': early, 'later': later, 'groups': groups, 'core': core, 'wide': wide, 'paths': paths,
        'scale': scale, 'known_past_sums': past}
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({f'{kind}__{k}': v for k, nested in forecasts.items() for kind, v in nested.items()})
    arrays.update({f'neighbor_ids__{k}': v for k, v in neighbors.items()})
    arrays.update({f'neighbor_weights__{k}': v for k, v in weights.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.read_csv(BASE / 'market_panel.csv').to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    pd.DataFrame(scenario_logs).to_csv(OUT / 'scenario_log.csv', index=False)
    sources = [DATA, BASE / 'outputs.npz', BASE / 'metadata.json', AP3 / 'outputs.npz',
        Path('research/after_publication_ap7_registered.md')]
    metadata = {'packet': 'AP7', 'n_rows': len(panel), 'n_policies': len(signals), 'families': list(forecasts),
        'core_names': core_names, 'wide_names': names, 'calendar_assumed': True,
        'market_delay_minutes': 20, 'decision': '18:30 MSK', 'reference': 'latest announced CBR',
        'fresh_holdout': False, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(frame[(frame.h == 5) & frame.candidate.isin(focus)].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
