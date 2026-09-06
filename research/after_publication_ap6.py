"""Registered AP6 OOS stacks; identical support and immutable early selection."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ml.data import CORRIDORS, load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard, summaries, paired_bootstrap
from research.after_publication_ap3_policy import past_percentiles, sequential_policy
from research.after_publication_ap4 import INCUMBENT, choose_early
from research.after_publication_ap5 import EXPERTS, AP4_CONTROL, expert_features
from research.after_publication_ap6_learning import fit_stacks
from research.after_publication_panel import build_features, build_outcomes

BASE = Path('results/research/after_publication/ap5')
AP3, AP4, OUT = BASE.parent / 'ap3', BASE.parent / 'ap4', BASE.parent / 'ap6'
EQUAL_CONTROL = 'equal_rolling365_h5_urgent_cap2'


def context_features(X, names, market, experts):
    lookup = {name: X[:, j] for j, name in enumerate(names)}
    scale = np.maximum(lookup['announced_vol20'], 1.)
    context = {name: lookup[name] for name in (
        'known_change_z', 'announced_range20', 'announced_range90', 'dow_sin', 'dow_cos',
        'month_end', 'pre_new_year14', *['currency_' + c for c in CORRIDORS])}
    for name in ('announced_ret5', 'announced_ret20', 'local_minus_common', 'peer_change_std'):
        context[name + '_z'] = lookup[name] / scale
    context['log_vol20'] = np.log1p(lookup['announced_vol20'])
    for name in ('cny_basis_last_z', 'cny_late_z', 'cny_quality', 'cny_last_missing',
                 'local_basis_last_z', 'local_quality', 'local_last_missing'):
        context[name] = market[name].to_numpy()
    source_names = list(context)
    source = np.column_stack(list(context.values()))
    transformed = expit(experts)
    context['expert_disagreement'] = transformed.std(axis=1)
    context['cny_minus_hist_probability_scale'] = transformed[:, 0] - transformed[:, 1]
    return np.column_stack(list(context.values())), list(context), source, source_names


def build_policies(panel, previous, models):
    dates, cur = panel.date.to_numpy(), panel.currency.to_numpy()
    cdf = lambda v: past_percentiles(v, cur)
    raw = {k.removesuffix('_urgent_cap2'): previous['score__' + k.removesuffix('_urgent_cap2')]
           for k in (INCUMBENT, AP4_CONTROL, EQUAL_CONTROL)}
    # Use original CNY z, not a time-varying calibrated score, for the fixed anchor.
    with np.load(AP3 / 'outputs.npz') as ap3:
        cny = cdf(ap3['score__cny_last'])
    ap4 = raw[AP4_CONTROL.removesuffix('_urgent_cap2')]
    for key, score in models.items():
        rank = cdf(score)
        raw['stack_' + key] = score
        raw['cny50_stack_' + key] = .5 * cny + .5 * rank
        raw['ap4_w25_stack_' + key] = .75 * ap4 + .25 * rank
    signals = {k + '_urgent_cap2': sequential_policy(v, dates, cur, 'urgent_cap2') for k, v in raw.items()}
    assert len(signals) == 51
    for k in (INCUMBENT, AP4_CONTROL, EQUAL_CONTROL):
        np.testing.assert_array_equal(signals[k], previous['signal__' + k])
    return raw, signals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, AP4, AP3, BASE.parent / 'ap2_delay20'):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    series = load(DATA)
    panel, X, names = build_features(series)
    keep = panel.date.to_numpy() >= dt.date(2022, 1, 1)
    panel, X = panel[keep].reset_index(drop=True), X[keep]
    previous_panel = pd.read_csv(BASE / 'announcement_panel.csv')
    np.testing.assert_array_equal(panel.date.astype(str), previous_panel.date)
    np.testing.assert_array_equal(panel.currency, previous_panel.currency)
    market = pd.read_csv(BASE / 'market_panel.csv')
    panel['decision_at'] = market.decision_at
    outcomes = build_outcomes(series, panel, 'publication')
    y = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
    with np.load(BASE / 'outputs.npz') as previous, np.load(AP3 / 'outputs.npz') as ap3, np.load(AP4 / 'outputs.npz') as ap4:
        experts = expert_features(ap4, ap3)[:, :, 2]
        context, context_names, source, source_names = context_features(X, names, market, experts)
        equal = previous['ensemble__equal_rolling365']
        for h in HORIZONS:
            np.testing.assert_array_equal(outcomes[f'y{h}'], previous[f'y{h}'])
        models, origins, logs, coefficients, corrections = fit_stacks(
            panel, experts, context, source, equal, y, outcomes['mature20'], list(EXPERTS), context_names, source_names)
        raw, signals = build_policies(panel, previous, models)
        early, later, groups = previous['early'], previous['later'], previous['groups']
        assert np.logical_and.reduce([np.isfinite(v) for v in raw.values()])[early | later].all()
    selection = choose_early(panel, outcomes, signals, early, groups, output=OUT)
    final = pd.DataFrame([{'candidate': k, **r} for k, v in signals.items()
                          for r in scorecard(panel, outcomes, v, later, groups)])
    final.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(final).to_csv(OUT / 'retrospective_summary.csv')
    focus = list(dict.fromkeys([selection['selected'], INCUMBENT, AP4_CONTROL, EQUAL_CONTROL,
        *['stack_' + k + '_urgent_cap2' for k in models]]))
    paired_bootstrap(panel, outcomes, signals, later, groups, focus, INCUMBENT).to_csv(OUT / 'paired_vs_ap3.csv', index=False)
    arrays = {'dates': panel.date.astype(str).to_numpy(), 'currencies': panel.currency.to_numpy().astype(str),
        'early': early, 'later': later, 'groups': groups, 'experts': experts, 'context': context,
        'source_context': source, 'equal': equal, 'local_origins': np.array([str(d) for d in origins])}
    arrays['dates'] = arrays['dates'].astype(str)
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({f'correction__{k}': v for k, v in corrections.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    market.to_csv(OUT / 'market_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'coefficients.csv', index=False)
    paths = [DATA, BASE / 'outputs.npz', BASE / 'metadata.json', AP4 / 'outputs.npz', AP3 / 'outputs.npz',
        Path('research/after_publication_ap6_registered.md')]
    meta = {'packet': 'AP6', 'n_rows': len(panel), 'n_policies': len(signals), 'experts': EXPERTS,
        'context_names': context_names, 'source_context_names': source_names,
        'calendar_assumed': True, 'decision': '18:30 MSK', 'market_delay_minutes': 20,
        'reference': 'latest announced CBR', 'fresh_holdout': False,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(final[(final.h == 5) & final.candidate.isin(focus)].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
