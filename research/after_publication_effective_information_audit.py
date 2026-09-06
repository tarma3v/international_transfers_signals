"""Matched information ablations, effective-reference checks and uncertainty."""
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2 import matured_mask
from research.after_publication_ap2_features import load_market_frames, append_features
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_panel import build_outcomes
from research.after_publication_reference_comparison import validate_references
from research.after_publication_effective_information import past_features, OUTPUTS, SIMPLE, INCUMBENT
from research.after_publication_effective_market_control import OUT, BASE, effective_market


def main():
    for folder in (OUT, BASE):
        for path, digest in json.loads((folder / 'metadata.json').read_text())['source_sha256'].items():
            assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == digest
    series = load(DATA)
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    outcomes = {r: build_outcomes(series, panel, r) for r in ('effective', 'publication')}
    validate_references(series, panel, outcomes)
    with np.load(OUT / 'outputs.npz') as z, np.load(BASE / 'outputs.npz') as old:
        saved, previous = ({k: obj[k] for k in obj.files} for obj in (z, old))
    for key, value in previous.items():
        np.testing.assert_array_equal(saved[key], value)
    names = json.loads((BASE / 'metadata.json').read_text())['cbr_feature_names']
    past = past_features(series, panel, names)
    np.testing.assert_array_equal(past, saved['features__past'])
    market = effective_market(panel, series, load_market_frames())
    matrix, _ = append_features(past, names, market)
    np.testing.assert_array_equal(matrix, saved['features__past_market'])
    old_market = pd.read_csv(OUT / 'market_panel.csv')
    for prefix in ('cny', 'local'):
        for field in ('n', 'age', 'quality', 'range', 'last_missing', 'mean_missing', 'post_missing'):
            np.testing.assert_allclose(market[prefix + '_' + field], old_market[prefix + '_' + field], rtol=1e-12, atol=1e-12)
    # Independent algebraic check from the original published-price bases:
    # the observable quote is the same; only the available CBR reference changes.
    rebase_errors = []
    for i, row in enumerate(panel.itertuples()):
        for prefix, c in (('cny', 'CNY'), ('local', row.currency)):
            s = series[c]
            current = int(np.searchsorted(s.dates, row.date, side='right')) - 1
            announced = int(np.searchsorted(s.dates, row.date + dt.timedelta(days=1), side='right')) - 1
            vol = max(float(np.std(np.diff(np.log(s.values[max(0, current - 20):current + 1])) * 1e4)), 1.)
            avol = max(float(np.std(np.diff(np.log(s.values[max(0, announced - 20):announced + 1])) * 1e4)), 1.)
            for part in ('last', 'mean', 'post'):
                if market[prefix + '_' + part + '_missing'].iloc[i]:
                    continue
                expected = (old_market[prefix + '_basis_' + part + '_z'].iloc[i] * avol +
                            1e4 * np.log(s.values[announced] / s.values[current])) / vol
                actual = market[prefix + '_basis_' + part + '_z'].iloc[i]
                rebase_errors.append(abs(actual - expected))
    assert max(rebase_errors) < 1e-8
    for h in HORIZONS:
        for kind in ('y', 'sym', 'forward', 'floor'):
            expected = outcomes['effective'][kind + str(h)].copy()
            expected[~np.isfinite(outcomes['publication']['y' + str(h)])] = np.nan
            np.testing.assert_array_equal(saved[kind + str(h)], expected)
    logs = pd.read_csv(OUT / 'training_log.csv')
    assert (logs.groupby('origin').mask_sha256.nunique() == 1).all()
    assert set(logs.information) == {'past', 'announced', 'market', 'past_market'}
    for row in logs.itertuples():
        origin = dt.date.fromisoformat(row.origin)
        mask = matured_mask(panel, outcomes['publication'], origin)
        assert hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() == row.mask_sha256
        assert mask.sum() == row.n_train
        assert (outcomes['effective']['mature20'][mask] < origin - dt.timedelta(days=2)).all()
        assert str(max(outcomes['effective']['mature20'][mask])) == row.last_effective_mature20
        assert str(max(outcomes['publication']['mature20'][mask])) == row.last_shared_cap
        y = np.column_stack([outcomes['effective']['y' + str(h)][mask] for h in HORIZONS])
        risk = np.column_stack([np.ones(len(y), dtype=bool), y[:, :-1] == 1])
        assert risk.sum() == row.n_at_risk
        assert (1 - y)[risk].sum() == row.n_failures
    keys = [k.removeprefix('signal__') for k in saved if k.startswith('signal__')]
    eligible = panel.announced_price.to_numpy() >= panel.current_price.to_numpy()
    for key in keys:
        if '_gated_' in key or key in (SIMPLE, 'known_change_z_urgent_cap2'):
            signal = saved['signal__' + key]
            assert not signal[~eligible].any()
            scored = signal & saved['later'] & np.isfinite(saved['y1'])
            assert (saved['y1'][scored] == 1).all()
    accuracy = []
    for info in ('past', 'announced', 'market', 'past_market'):
        curve = saved['survival__' + info]
        finite = np.isfinite(curve).all(axis=1)
        assert ((curve[finite] >= 0) & (curve[finite] <= 1)).all()
        assert (np.diff(curve[finite], axis=1) <= 1e-12).all()
        for scope in ('early', 'later'):
            for model in OUTPUTS:
                p = saved['prediction__' + info + '_' + model]
                mask = saved[scope] & np.isfinite(saved['y5'])
                if model != 'survival_mean':
                    accuracy.append({'information': info, 'model': model, 'period': scope,
                        'n': int(mask.sum()), 'brier_h5': float(np.mean((p[mask] - saved['y5'][mask]) ** 2))})
    pd.DataFrame(accuracy).to_csv(OUT / 'probability_accuracy.csv', index=False)
    pairs = [(SIMPLE, keys), ('known_change_z_urgent_cap2', keys)]
    for model in OUTPUTS:
        pairs += [(f'past_{model}_urgent_cap2', [f'announced_{model}_urgent_cap2']),
                  (f'announced_{model}_urgent_cap2', [f'market_{model}_urgent_cap2']),
                  (f'past_market_{model}_urgent_cap2', [f'market_{model}_urgent_cap2', f'market_{model}_gated_urgent_cap2'])]
        for info in ('announced', 'market'):
            pairs.append((f'{info}_{model}_urgent_cap2', [f'{info}_{model}_gated_urgent_cap2']))
    common_audit(OUT, keys, pairs)
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    pd.concat([selected_bootstrap(panel, saved, selected, b) for b in (20, 50)]).to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
    detail = []
    years = np.array([d.year for d in panel.date])
    for key in dict.fromkeys([selected, SIMPLE, 'market_hist_gated_urgent_cap2', 'past_market_hist_urgent_cap2']):
        for year in (2024, 2025, 2026):
            for c in sorted(panel.currency.unique()):
                scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == c)
                effective_scored = {k: v for k, v in saved.items() if k.startswith(('y', 'sym', 'forward', 'floor'))}
                detail.extend({'candidate': key, 'year': year, 'currency': c, **r} for r in
                    scorecard(panel, effective_scored, saved['signal__' + key], scope, saved['groups']))
    pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
    (OUT / 'audit_checks.json').write_text(json.dumps({'source_hashes_verified': True,
        'events_reindexed_and_labels_checked': len(panel), 'training_records_checked': len(logs),
        'all_four_information_masks_identical': True, 'all_old_predictions_unchanged': True,
        'effective_market_rebase_checks': len(rebase_errors), 'max_rebase_abs_error': max(rebase_errors),
        'gated_h1_known_and_exact': True, 'all_survival_curves_valid': True,
        'historical_receipts_certified': False, 'fresh_holdout': False}, indent=2))


if __name__ == '__main__':
    main()
