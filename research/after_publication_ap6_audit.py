"""AP6 paired ablations, replay invariants and descriptive early regime errors."""
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from ml.data import load
from ml.targets import HORIZONS
from research.after_publication_ap1 import DATA, scorecard
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_ap5_learning import available_mask
from research.after_publication_ap6 import OUT, BASE, INCUMBENT, AP4_CONTROL, EQUAL_CONTROL
from research.after_publication_ap6_learning import MODEL_NAMES, ranking_pairs
from research.after_publication_panel import build_outcomes


def main():
    meta = json.loads((OUT / 'metadata.json').read_text())
    for path, digest in meta['source_sha256'].items():
        assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == digest, path
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    logs = pd.read_csv(OUT / 'training_log.csv')
    coefficients = pd.read_csv(OUT / 'coefficients.csv')
    with np.load(OUT / 'outputs.npz') as saved, np.load(BASE / 'outputs.npz') as previous:
        signals = {k.removeprefix('signal__'): saved[k] for k in saved.files if k.startswith('signal__')}
        outcomes = build_outcomes(load(DATA), panel, 'publication')
        y = np.column_stack([outcomes[f'y{h}'] for h in HORIZONS])
        for name, value in outcomes.items():
            if not name.startswith('mature'):
                np.testing.assert_array_equal(value, saved[name])
        for key in (INCUMBENT, AP4_CONTROL, EQUAL_CONTROL):
            np.testing.assert_array_equal(signals[key], previous['signal__' + key])
        for key in ('early', 'later', 'groups', 'dates', 'currencies'):
            np.testing.assert_array_equal(saved[key], previous[key])
        source = (np.isfinite(saved['experts']).all(axis=1) & np.isfinite(saved['context']).all(axis=1)
                  & np.isfinite(saved['equal']).all(axis=1))
        assert len(meta['source_context_names']) + 2 == len(meta['context_names'])
        np.testing.assert_array_equal(saved['context'][:, :-2], saved['source_context'])
        local = saved['score__stack_local_logit']
        origins = np.array([dt.date.fromisoformat(d) for d in saved['local_origins']])
        for kind, anchor in (('local', local), ('equal', saved['equal'][:, 2])):
            np.testing.assert_array_equal(saved['score__stack_' + kind + '_residual'],
                np.clip(anchor + saved['correction__' + kind], 0., 1.))
        inputs = dict(zip(meta['experts'], saved['experts'].T))
        inputs.update(dict(zip(meta['context_names'], saved['context'].T)))
        for (origin_text, model, currency), group in coefficients.groupby(['origin', 'model', 'currency']):
            origin = dt.date.fromisoformat(origin_text)
            mask = available_mask(dates, outcomes['mature20'], origin, np.isfinite(y).all(axis=1)) & source
            if '730' in model:
                mask &= dates >= origin - dt.timedelta(days=730)
            if currency != 'ALL':
                mask &= currencies == currency
            matrix = np.column_stack([inputs[name][mask] for name in group.feature])
            np.testing.assert_allclose(group['mean'], matrix.mean(axis=0), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(group.scale, np.maximum(matrix.std(axis=0), 1e-6), rtol=1e-10, atol=1e-10)
        pair_counts = {}
        for row in logs.itertuples():
            origin = dt.date.fromisoformat(row.origin)
            mask = available_mask(dates, outcomes['mature20'], origin, np.isfinite(y).all(axis=1)) & source
            if '730' in row.model:
                mask &= dates >= origin - dt.timedelta(days=730)
            if row.currency != 'ALL':
                mask &= currencies == row.currency
            if row.model == 'local_residual':
                mask &= np.isfinite(local)
                assert (origins[mask] <= dates[mask]).all() and (origins[mask] < origin).all()
            assert mask.sum() == row.n_train, row
            if mask.any():
                assert str(max(outcomes['mature20'][mask])) == row.last_train_maturity
                assert str(max(dates[mask])) == row.last_train_date
            if row.model == 'pairwise_context':
                pairs = ranking_pairs(dates[mask], currencies[mask], y[mask, 2])
                assert len(pairs) == row.n_pairs
                pair_counts[row.origin] = len(pairs)
        selected = json.loads((OUT / 'selection.json').read_text())['selected']
        pd.concat([selected_bootstrap(panel, saved, selected, block) for block in (20, 50)]).to_csv(
            OUT / 'selected_block_sensitivity.csv', index=False)
        years = np.array([d.year for d in dates])
        rows = []
        for key in dict.fromkeys([selected, INCUMBENT, AP4_CONTROL]):
            for year in (2024, 2025, 2026):
                for c in sorted(set(currencies)):
                    scope = saved['later'] & (years == year) & (currencies == c)
                    rows.extend({'candidate': key, 'year': year, 'currency': c, **r}
                                for r in scorecard(panel, outcomes, signals[key], scope, saved['groups']))
        pd.DataFrame(rows).to_csv(OUT / 'selected_year_currency.csv', index=False)
        probability_rows, regime_rows = [], []
        known_change = saved['source_context'][:, meta['source_context_names'].index('known_change_z')]
        regimes = np.array([f'cny_{"up" if a >= 0 else "down"}_own_{"up" if b >= 0 else "down"}'
                           for a, b in zip(saved['experts'][:, 0], known_change)])
        for key in MODEL_NAMES:
            prediction = saved['score__stack_' + key]
            if key == 'pairwise_context':
                continue  # Linear ranking score is not a probability.
            target = y.mean(axis=1) if key == 'hist_multi' else y[:, 2]
            for period in ('early', 'later'):
                scope = saved[period] & np.isfinite(target)
                probability_rows.append({'model': key, 'period': period, 'target': 'mean5' if key == 'hist_multi' else 'h5',
                    'n': int(scope.sum()), 'brier_or_mse': float(np.mean((prediction[scope] - target[scope]) ** 2)),
                    'mean_prediction': float(prediction[scope].mean()), 'mean_target': float(target[scope].mean())})
            for regime in sorted(set(regimes)):
                scope = saved['early'] & (regimes == regime) & np.isfinite(target)
                regime_rows.append({'model': key, 'regime': regime, 'period': '2023_validation_descriptive',
                    'n': int(scope.sum()), 'mean_prediction': float(prediction[scope].mean()),
                    'mean_target': float(target[scope].mean()),
                    'brier_or_mse': float(np.mean((prediction[scope] - target[scope]) ** 2))})
        pd.DataFrame(probability_rows).to_csv(OUT / 'probability_accuracy.csv', index=False)
        pd.DataFrame(regime_rows).to_csv(OUT / 'early_regime_errors.csv', index=False)
    suffix = '_urgent_cap2'
    pairs = [(AP4_CONTROL, [k for k in signals if k != AP4_CONTROL]),
        ('stack_context_logit' + suffix, ['stack_logit_context_exp' + suffix]),
        ('stack_context_hist' + suffix, ['stack_hist_context' + suffix]),
        ('stack_logit_exp' + suffix, ['stack_logit_context_exp' + suffix, 'stack_local_logit' + suffix,
                                    'stack_shrunken_logit' + suffix, 'stack_positive_exp' + suffix]),
        ('stack_hist_experts' + suffix, ['stack_hist_context' + suffix, 'stack_hist_multi' + suffix]),
        (EQUAL_CONTROL, ['stack_equal_residual' + suffix]),
        ('stack_local_logit' + suffix, ['stack_local_residual' + suffix])]
    common_audit(OUT, list(signals), pairs)
    assert (coefficients.loc[coefficients.model.str.startswith('positive_'), 'coefficient'] >= 0).all()
    coefficients['year'] = pd.to_datetime(coefficients.origin).dt.year
    coefficients.groupby(['model', 'currency', 'year', 'feature']).coefficient.agg(['min', 'mean', 'max']).to_csv(
        OUT / 'coefficient_summary.csv')
    (OUT / 'audit_checks.json').write_text(json.dumps({
        'source_hashes_verified': True, 'train_log_rows_rebuilt': len(logs),
        'month_origins': int(logs.origin.nunique()), 'pair_counts_rebuilt': pair_counts,
        'unchanged_controls': [INCUMBENT, AP4_CONTROL, EQUAL_CONTROL],
        'support_unchanged': True, 'all_outcomes_rebuilt': True,
        'local_oos_origins_checked': True, 'source_only_excludes_experts': True,
        'positive_coefficients_checked': True, 'all_linear_training_scalers_rebuilt': True,
        'issued_residual_addition_reproduced': True}, indent=2))


if __name__ == '__main__':
    main()
