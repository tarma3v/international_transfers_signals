"""AP5 calibration/weight evidence, uncertainty and descriptive slices."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from research.after_publication_ap1 import scorecard, paired_bootstrap
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap4_audit import selected_bootstrap
from research.after_publication_ap5 import OUT, BASE, AP3, EXPERTS, INCUMBENT


def main():
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    with np.load(OUT / 'outputs.npz') as saved:
        keys = [k.removeprefix('signal__') for k in saved.files if k.startswith('signal__')]
    pairs = [('equal_rolling365_h5_urgent_cap2', [f'hedge_{scope}_h{half}_e{eta}_urgent_cap2'
              for scope in ('global', 'local') for half in (63, 252) for eta in (2, 10, 30)]),
             ('frozen_weights2023_urgent_cap2', ['hedge_global_h252_e10_urgent_cap2'])]
    common_audit(OUT, keys, pairs)
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    years = np.array([d.year for d in panel.date])
    weight_log = pd.read_csv(OUT / 'weight_log.csv')
    calibration_log = pd.read_csv(OUT / 'calibration_log.csv')
    for row in weight_log.dropna(subset=['last_revealed_maturity']).itertuples():
        now = dt.date.fromisoformat(row.feedback_asof)
        assert dt.date.fromisoformat(row.last_revealed_maturity) < now - dt.timedelta(days=2)
        assert dt.date.fromisoformat(row.last_revealed_prediction_date) < now
    for row in calibration_log.dropna(subset=['last_calibration_maturity']).itertuples():
        assert dt.date.fromisoformat(row.last_calibration_maturity) < dt.date.fromisoformat(row.fit_origin) - dt.timedelta(days=2)
    for key, group in weight_log.groupby('candidate'):
        w = group[[f'weight_{e}' for e in range(6)]].to_numpy()
        np.testing.assert_allclose(w.sum(axis=1), 1., atol=1e-12)
        assert (w >= .1 / 6 - 1e-12).all()
        if 'global' in key:
            assert group.groupby('date')[[f'weight_{e}' for e in range(6)]].nunique().max().max() == 1
    weight_log['year'] = pd.to_datetime(weight_log.date).dt.year
    weight_log.groupby(['candidate', 'year'])[[f'weight_{e}' for e in range(6)]].agg(['min', 'mean', 'max']).to_csv(OUT / 'weight_summary.csv')
    with np.load(OUT / 'outputs.npz') as saved, np.load(BASE / 'outputs.npz') as base, np.load(AP3 / 'outputs.npz') as ap3:
        pd.concat([selected_bootstrap(panel, saved, selected, block) for block in (20, 50)]).to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
        outcomes = {k: saved[k] for k in saved.files if k.startswith(('y', 'sym', 'forward', 'floor'))}
        detail = []
        for key in dict.fromkeys([selected, INCUMBENT]):
            for year in (2024, 2025, 2026):
                for c in sorted(panel.currency.unique()):
                    scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == c)
                    detail.extend({'candidate': key, 'year': year, 'currency': c, **r} for r in
                        scorecard(panel, outcomes, saved['signal__' + key], scope, saved['groups']))
        pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
        probabilities = {}
        for key in saved.files:
            if key.startswith('probability__'):
                for e, expert in enumerate(EXPERTS):
                    probabilities[key.removeprefix('probability__') + '_' + expert] = saved[key][:, e, :]
            elif key.startswith('ensemble__'):
                probabilities[key.removeprefix('ensemble__')] = saved[key]
        for kind in ('hist', 'logit', 'local'):
            probabilities['raw_survival_' + kind] = base['survival__' + kind]
        rows = []
        for key, pred in probabilities.items():
            finite = np.isfinite(pred).all(axis=1)
            assert ((pred[finite] >= 0) & (pred[finite] <= 1)).all()
            assert (np.diff(pred[finite], axis=1) <= 1e-12).all()
            for period in ('early', 'later'):
                for j, h in enumerate(HORIZONS):
                    y = saved[f'y{h}']
                    m = saved[period] & finite & np.isfinite(y)
                    rows.append({'candidate': key, 'period': period, 'h': h, 'n': int(m.sum()),
                        'brier': float(np.mean((pred[m, j] - y[m]) ** 2)),
                        'mean_probability': float(pred[m, j].mean()), 'event_rate': float(y[m].mean())})
        for kind in ('hist', 'extra'):
            pred = ap3['score__market_' + kind]
            for period in ('early', 'later'):
                y = saved['y5']
                m = saved[period] & np.isfinite(pred) & np.isfinite(y)
                rows.append({'candidate': 'raw_' + kind, 'period': period, 'h': 5, 'n': int(m.sum()),
                    'brier': float(np.mean((pred[m] - y[m]) ** 2)),
                    'mean_probability': float(pred[m].mean()), 'event_rate': float(y[m].mean())})
        pd.DataFrame(rows).to_csv(OUT / 'probability_accuracy.csv', index=False)
        # Raw issued controls from the already registered AP4 are diagnostics only.
        signals = {k.removeprefix('signal__'): saved[k] for k in saved.files if k.startswith('signal__')}
        raw_names = ('cny_last', 'market_hist', 'market_extra', 'hazard_hist_h5', 'hazard_logit_h5', 'hazard_local_h5')
        ablations = []
        for expert, raw in zip(EXPERTS, raw_names):
            control = 'raw_' + expert
            signals[control] = base['signal__' + raw + '_urgent_cap2']
            candidates = [f'cal_{mode}_{expert}_urgent_cap2' for mode in ('expanding', 'rolling365')]
            ablations.append(paired_bootstrap(panel, outcomes, signals, saved['later'], saved['groups'], candidates, control))
        pd.concat(ablations).to_csv(OUT / 'paired_calibration_vs_raw.csv', index=False)
    print(pd.read_csv(OUT / 'selected_block_sensitivity.csv').query('h == 5').to_string(index=False))


if __name__ == '__main__':
    main()
