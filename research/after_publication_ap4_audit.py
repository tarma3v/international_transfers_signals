"""AP4 uncertainty, target diagnostics and narrow slices without re-selection."""
import json

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from research.after_publication_ap1 import SEED, adjusted, scorecard
from research.after_publication_ap2_audit import main as common_audit
from research.after_publication_ap4 import OUT, BASE, INCUMBENT


def selected_bootstrap(panel, saved, selected, block):
    idx = np.where(saved['later'])[0]
    _, date_id = np.unique(panel.date.to_numpy()[idx], return_inverse=True)
    nd = int(date_id.max() + 1)
    rng = np.random.default_rng(SEED)
    weights = []
    for _ in range(1000):
        starts = rng.integers(0, nd, size=int(np.ceil(nd / block)))
        chosen = ((starts[:, None] + np.arange(block)) % nd).ravel()[:nd]
        weights.append(np.bincount(chosen, minlength=nd)[date_id])
    weights = np.array(weights)
    rows = []
    for h in HORIZONS:
        valid = np.isfinite(saved[f'y{h}'][idx])
        for metric in ('lift', 'sym', 'forward'):
            samples, points = [], []
            for key in (selected, INCUMBENT):
                signal = saved['signal__' + key][idx]
                if metric == 'lift':
                    y = saved[f'y{h}'][idx]
                    estimates = np.array([adjusted(y, signal, valid, saved['groups'][idx], w) for w in weights])
                    point = adjusted(y, signal, valid, saved['groups'][idx])
                else:
                    values = saved[f'{metric}{h}'][idx]
                    usable = signal & valid & np.isfinite(values)
                    n = weights[:, usable].sum(axis=1)
                    estimates = np.divide(weights[:, usable] @ values[usable], n,
                        out=np.full(1000, np.nan), where=n > 0)
                    point = float(values[usable].mean())
                samples.append(estimates)
                points.append(point)
            diff = samples[0] - samples[1]
            rows.append({'candidate': selected, 'control': INCUMBENT, 'h': h,
                'metric': metric, 'block_dates': block, 'draws': 1000,
                'estimate': points[0], 'ci_lo': float(np.nanquantile(samples[0], .025)),
                'ci_hi': float(np.nanquantile(samples[0], .975)),
                'difference': points[0] - points[1],
                'delta_lo': float(np.nanquantile(diff, .025)), 'delta_hi': float(np.nanquantile(diff, .975))})
    return pd.DataFrame(rows)


def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    selected = selection['selected']
    with np.load(OUT / 'outputs.npz') as saved:
        keys = [key.removeprefix('signal__') for key in saved.files if key.startswith('signal__')]
    pairs = [
        (INCUMBENT, ['soft_past_all_w25_urgent_cap2', 'soft_pred_all_w25_urgent_cap2', 'soft_future_mean_w25_urgent_cap2']),
        ('soft_past_all_w25_urgent_cap2', ['soft_pred_all_w25_urgent_cap2']),
        ('hazard_local_mean_urgent_cap2', ['local_hazard_residual25_urgent_cap2', 'local_hazard_residual50_urgent_cap2']),
        ('market_hist_urgent_cap2', ['hazard_hist_h5_urgent_cap2', 'hazard_hist_mean_urgent_cap2', 'restricted_wait_log_urgent_cap2']),
    ]
    common_audit(OUT, keys, pairs)
    panel = pd.read_csv(OUT / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    years = np.array([d.year for d in panel.date])
    with np.load(OUT / 'outputs.npz') as saved, np.load(BASE / 'outputs.npz') as previous:
        bootstrap = pd.concat([selected_bootstrap(panel, saved, selected, block) for block in (20, 50)])
        bootstrap.to_csv(OUT / 'selected_block_sensitivity.csv', index=False)
        outcomes = {k: saved[k] for k in saved.files if k.startswith(('y', 'sym', 'forward', 'floor'))}
        detail = []
        for key in dict.fromkeys([selected, INCUMBENT]):
            for year in (2024, 2025, 2026):
                for currency in sorted(panel.currency.unique()):
                    scope = saved['later'] & (years == year) & (panel.currency.to_numpy() == currency)
                    detail.extend({'candidate': key, 'year': year, 'currency': currency, **row}
                        for row in scorecard(panel, outcomes, saved['signal__' + key], scope, saved['groups']))
        pd.DataFrame(detail).to_csv(OUT / 'selected_year_currency.csv', index=False)
        calibration = []
        for kind in ('logit', 'hist', 'local'):
            curve = saved['survival__' + kind]
            finite = np.isfinite(curve).all(axis=1)
            assert ((curve[finite] >= 0) & (curve[finite] <= 1)).all()
            assert (np.diff(curve[finite], axis=1) <= 1e-12).all()
            for period in ('early', 'later'):
                for j, h in enumerate(HORIZONS):
                    y = saved[f'y{h}']
                    m = saved[period] & finite & np.isfinite(y)
                    calibration.append({'candidate': 'hazard_' + kind, 'period': period, 'h': h,
                        'n': int(m.sum()), 'brier': float(np.mean((curve[m, j] - y[m]) ** 2)),
                        'mean_probability': float(curve[m, j].mean()), 'event_rate': float(y[m].mean())})
        for period in ('early', 'later'):
            y = saved['y5']
            for kind in ('market_hist', 'market_extra'):
                pred = previous['score__' + kind]
                m = saved[period] & np.isfinite(pred) & np.isfinite(y)
                calibration.append({'candidate': kind, 'period': period, 'h': 5, 'n': int(m.sum()),
                    'brier': float(np.mean((pred[m] - y[m]) ** 2)),
                    'mean_probability': float(pred[m].mean()), 'event_rate': float(y[m].mean())})
        pd.DataFrame(calibration).to_csv(OUT / 'survival_calibration.csv', index=False)
        print(bootstrap.query('h == 5').to_string(index=False))


if __name__ == '__main__':
    main()
