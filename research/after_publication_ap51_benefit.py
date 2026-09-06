"""AP51-T: expected future-only CBR benefit for the temperature widget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap50_temperature import HORIZONS
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap51_benefit')
BASE = OUT.parent / 'ap50_temperature'


def magnitude_metrics(target, prediction, prior):
    valid = (np.isfinite(target) & np.isfinite(prediction) & np.isfinite(prior))
    y, p, b = target[valid], prediction[valid], prior[valid]
    correlation = spearmanr(y, p).statistic if len(y) > 1 else np.nan
    return {
        'n': int(len(y)),
        'mae_model': float(mean_absolute_error(y, p)),
        'mae_prior': float(mean_absolute_error(y, b)),
        'rmse_model': float(mean_squared_error(y, p) ** .5),
        'rmse_prior': float(mean_squared_error(y, b) ** .5),
        'spearman': float(correlation),
        'positive_sign_accuracy': float(((p >= 0) == (y >= 0)).mean()),
        'mean_predicted_bps': float(p.mean()),
        'mean_actual_bps': float(y.mean()),
    }


def calibration_bins(target, prediction, horizon):
    valid = np.isfinite(target) & np.isfinite(prediction)
    frame = pd.DataFrame({'target': target[valid], 'prediction': prediction[valid]})
    frame['bin'] = pd.qcut(frame.prediction, 5, duplicates='drop')
    result = frame.groupby('bin', observed=True).agg(
        n=('target', 'size'), predicted_bps=('prediction', 'mean'),
        actual_bps=('target', 'mean'), mae=('target', lambda value: 0.)).reset_index()
    # Compute bin MAE explicitly because groupby aggregation needs both columns.
    maes = frame.groupby('bin', observed=True).apply(
        lambda group: np.mean(np.abs(group.target - group.prediction)),
        include_groups=False).to_numpy()
    result['mae'] = maes
    result.insert(0, 'h', horizon)
    result['bin'] = result['bin'].astype(str)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(BASE / 'outputs.npz') as source:
        old = {key: source[key] for key in source.files}
    cap = build_outcomes(load(DATA), panel, 'publication')
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    predictions, priors, counts, logs = {}, {}, {}, []
    for h in HORIZONS:
        features = np.column_stack([
            old['multihorizon_features'],
            old['calibrated_probability_' + str(h)],
        ])
        result = fit_quarterly_benefit(
            features, old['forward' + str(h)], cap['mature' + str(h)], dates)
        predictions[h], priors[h], counts[h], head_logs = result
        logs.extend({'h': h, **row} for row in head_logs)

    scope = old['later'].astype(bool)
    rows, bins = [], []
    for h in HORIZONS:
        target = old['forward' + str(h)]
        rows.append({'h': h, 'slice': 'ALL', 'group': 'ALL',
                     **magnitude_metrics(target[scope], predictions[h][scope],
                                         priors[h][scope])})
        bins.append(calibration_bins(
            target[scope], predictions[h][scope], h))
        for currency in sorted(set(currencies)):
            part = scope & (currencies == currency)
            rows.append({'h': h, 'slice': 'currency', 'group': currency,
                         **magnitude_metrics(target[part], predictions[h][part],
                                             priors[h][part])})
        for year in sorted({day.year for day in dates[scope]}):
            part = scope & np.array([day.year == year for day in dates])
            rows.append({'h': h, 'slice': 'year', 'group': str(year),
                         **magnitude_metrics(target[part], predictions[h][part],
                                             priors[h][part])})
    pd.DataFrame(rows).to_csv(OUT / 'benefit_metrics.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(
        OUT / 'benefit_calibration_bins.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)

    arrays = {key: old[key] for key in old}
    for h in HORIZONS:
        arrays['expected_future_bps_' + str(h)] = predictions[h]
        arrays['benefit_prior_' + str(h)] = priors[h]
        arrays['benefit_n_train_' + str(h)] = counts[h]
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    sources = [
        DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
        Path('research/after_publication_ap51_benefit_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP51-T',
        'model': 'quarterly mature-only winsorized Ridge future benefit',
        'units': 'CBR future-only basis points',
        'push_policy_changed': False,
        'bank_execution_validated': False,
        'intraday_widget_validated': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(pd.DataFrame(rows).query("slice == 'ALL'").to_string(index=False))


if __name__ == '__main__':
    main()
