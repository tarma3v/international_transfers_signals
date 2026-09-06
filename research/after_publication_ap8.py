"""Fixed models at four post-publication clocks, with frozen-information controls."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, factory, scorecard, summaries
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_features import load_market_frames, market_features, append_features
from research.after_publication_ap3_policy import past_percentiles, sequential_policy
from research.after_publication_ap4 import INCUMBENT, choose_early
from research.after_publication_ap4_survival import fit_predict_hazard
from research.after_publication_ap5 import AP4_CONTROL
from research.after_publication_panel import build_features, build_outcomes

OUT = Path('results/research/after_publication/ap8')
BASE = OUT.parent / 'ap4'
AP2 = OUT.parent / 'ap2_delay20'
CLOCKS = {'1810': dt.time(18, 10), '1830': dt.time(18, 30),
          '1850': dt.time(18, 50), '1930': dt.time(19, 30)}
FAMILIES = ('cny', 'cny_hist50', 'cny_hazard50')


def fit_clock(panel, X, hazard_X, outcomes, origins=None):
    dates = panel.date.to_numpy()
    if origins is None:
        origins = [dt.date(y, m, 1) for y in range(2022, max(d.year for d in dates) + 1)
                   for m in (1, 4, 7, 10) if dt.date(y, m, 1) >= dt.date(2022, 7, 1)]
    direct = np.full(len(panel), np.nan)
    curve = np.full((len(panel), len(HORIZONS)), np.nan)
    survival = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
    logs = []
    for origin in origins:
        end = (pd.Timestamp(origin).to_period('Q') + 1).start_time.date()
        te = (dates >= origin) & (dates < end)
        if not te.any():
            continue
        tr = matured_mask(panel, outcomes, origin)
        target = outcomes['y5']
        if tr.sum() < 400 or np.unique(target[tr]).size < 2:
            direct[te] = target[tr].mean() if tr.any() else 0.
        else:
            model = factory('hist7y').set_params(early_stopping=False)
            model.fit(X[tr], target[tr])
            direct[te] = model.predict_proba(X[te])[:, 1]
        curve[te], risk, failures = fit_predict_hazard(hazard_X[tr], survival[tr], hazard_X[te], 'hist', 400)
        logs.append({'origin': str(origin), 'n_train': int(tr.sum()), 'n_test': int(te.sum()),
            'last_train_date': str(max(dates[tr])) if tr.any() else None,
            'last_mature20': str(max(outcomes['mature20'][tr])) if tr.any() else None,
            'mask_sha256': hashlib.sha256(np.packbits(tr).tobytes()).hexdigest(),
            'n_at_risk': risk, 'n_failures': failures})
        print(f'AP8 {origin}: {tr.sum()} mature training rows', flush=True)
    return direct, curve, logs


def clock_policies(panel, cny, hist, hazard):
    cur = panel.currency.to_numpy()
    rank = lambda x: past_percentiles(x, cur)
    return {'cny': cny, 'cny_hist50': .5 * rank(cny) + .5 * rank(hist),
            'cny_hazard50': .5 * rank(cny) + .5 * rank(hazard[:, 2])}


def freeze_policies(raw):
    """Freeze the whole issued18:30 snapshot; not a fresh older quote's age."""
    result = {}
    for clock in ('1850', '1930'):
        for family in FAMILIES:
            result[f't{clock}_frozen1830_{family}'] = raw[f't1830_{family}'].copy()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in (BASE, AP2):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    series = load(DATA)
    panel, X, names = build_features(series)
    keep = panel.date.to_numpy() >= dt.date(2022, 1, 1)
    panel, X = panel[keep].reset_index(drop=True), X[keep]
    outcomes = build_outcomes(series, panel, 'publication')
    frames = load_market_frames()
    arrays, raw, logs, compatibility = {}, {}, [], {}
    with np.load(BASE / 'outputs.npz') as saved, np.load(AP2 / 'outputs.npz') as old_ap2:
        saved, old_ap2 = ({k: obj[k] for k in obj.files} for obj in (saved, old_ap2))
    for key in ('dates', 'currencies', 'early', 'later', 'groups'):
        arrays[key] = saved[key]
    np.testing.assert_array_equal(panel.date.astype(str), saved['dates'])
    np.testing.assert_array_equal(panel.currency, saved['currencies'])
    for h in HORIZONS:
        np.testing.assert_array_equal(outcomes[f'y{h}'], saved[f'y{h}'])
    for clock, cutoff in CLOCKS.items():
        print('AP8 clock ' + clock, flush=True)
        market = market_features(panel, series, frames, 20, cutoff)
        market.to_csv(OUT / f'market_{clock}.csv', index=False)
        xx, full_names = append_features(X, names, market)
        # AP4 read its saved market CSV; preserve that numeric representation.
        hx, _ = append_features(X, names, pd.read_csv(OUT / f'market_{clock}.csv'))
        np.testing.assert_array_equal(full_names, json.loads((AP2 / 'feature_names.json').read_text()))
        hist, hazard, fit_logs = fit_clock(panel, xx, hx, outcomes)
        cny = np.where(market.cny_last_missing.to_numpy() == 0,
                       market.cny_basis_last_z, X[:, names.index('known_change_z')])
        raw.update({f't{clock}_{k}': v for k, v in clock_policies(panel, cny, hist, hazard).items()})
        arrays.update({f'features__{clock}': xx, f'hazard_features__{clock}': hx,
                       f'hist__{clock}': hist, f'survival__{clock}': hazard})
        logs.extend({'clock': clock, **r} for r in fit_logs)
        if clock == '1830':
            panel['decision_at'] = market.decision_at
            market.to_csv(OUT / 'market_panel.csv', index=False)
            for key, new, old in (('cny', cny, old_ap2['score__cny_last']),
                ('hist', hist, old_ap2['score__market_hist']), ('hazard', hazard, saved['survival__hist'])):
                delta = float(np.nanmax(np.abs(new - old)))
                compatibility[key + '_max_abs_error'] = delta
                np.testing.assert_allclose(new, old, rtol=1e-11, atol=1e-12)
    raw.update(freeze_policies(raw))
    for key in (INCUMBENT, AP4_CONTROL):
        raw[key.removesuffix('_urgent_cap2')] = saved['score__' + key.removesuffix('_urgent_cap2')]
    signals = {key + '_urgent_cap2': sequential_policy(value, panel.date.to_numpy(), panel.currency.to_numpy(), 'urgent_cap2')
               for key, value in raw.items()}
    assert len(signals) == 20
    for name, key in (('t1830_cny_hist50', INCUMBENT), ('t1830_cny_hazard50', AP4_CONTROL)):
        np.testing.assert_array_equal(signals[name + '_urgent_cap2'], saved['signal__' + key])
    common = np.logical_and.reduce([np.isfinite(value) for value in raw.values()])
    assert common[arrays['early'] | arrays['later']].all()
    selection = choose_early(panel, outcomes, signals, arrays['early'], arrays['groups'], output=OUT)
    later = pd.DataFrame([{'candidate': k, **row} for k, fired in signals.items()
                          for row in scorecard(panel, outcomes, fired, arrays['later'], arrays['groups'])])
    later.to_csv(OUT / 'retrospective_all_horizons.csv', index=False)
    summaries(later).to_csv(OUT / 'retrospective_summary.csv')
    arrays.update({f'score__{k}': v for k, v in raw.items()})
    arrays.update({f'signal__{k}': v for k, v in signals.items()})
    arrays.update({k: v for k, v in outcomes.items() if not k.startswith('mature')})
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    sources = [DATA, AP2 / 'metadata.json', AP2 / 'outputs.npz', BASE / 'metadata.json', BASE / 'outputs.npz',
               Path('research/after_publication_ap8_registered.md'), Path('data/after_publication_ap2/manifest.json'),
               Path('data/moex_direct_pairs/manifest.json')]
    metadata = {'packet': 'AP8', 'n_rows': len(panel), 'n_features': len(full_names), 'feature_names': full_names,
        'n_policies': len(signals), 'clocks': list(CLOCKS), 'market_delay_minutes': 20,
        'cbr_features': 'publication snapshot, unchanged across evening clocks',
        'calendar_assumed': True, 'fresh_holdout': False, 'bank_execution_validated': False,
        'compatibility_1830': compatibility, 'reference': 'latest announced CBR',
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(later[later.h == 5].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
