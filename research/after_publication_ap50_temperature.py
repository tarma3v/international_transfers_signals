"""AP50-T: calibrated AP49 scores and latest-valid widget artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap46_effective_models import currency_features
from research.after_publication_ap49_effective import CANDIDATE as AP49_CANDIDATE
from research.after_publication_ap49_effective_models import geometric_consensus
from research.after_publication_ap50_temperature_models import (
    expected_calibration_error,
    fit_quarterly_calibrator,
)
from research.after_publication_panel import build_outcomes


OUT = Path('results/research/after_publication/ap50_temperature')
BASE = OUT.parent / 'ap49_effective'
HORIZONS = (3, 5, 10, 20)


def probability_metrics(target, raw, calibrated, prior):
    valid = (np.isfinite(target) & np.isfinite(raw)
             & np.isfinite(calibrated) & np.isfinite(prior))
    y = target[valid].astype(int)
    row = {'n': int(valid.sum()), 'positive_rate': float(y.mean())}
    for name, value in (('raw', raw[valid]), ('calibrated', calibrated[valid]),
                        ('prior', prior[valid])):
        value = np.clip(value, 1e-6, 1. - 1e-6)
        row['brier_' + name] = float(brier_score_loss(y, value))
        row['logloss_' + name] = float(log_loss(y, value, labels=[0, 1]))
        row['ece_' + name] = expected_calibration_error(value, y)
        row['auc_' + name] = float(roc_auc_score(y, value))
        row['average_precision_' + name] = float(
            average_precision_score(y, value))
    return row


def reliability_rows(target, probability, horizon, currency='ALL', year='ALL'):
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(target) & np.isfinite(probability)
    edges = np.linspace(0., 1., 11)
    ids = np.minimum(np.searchsorted(
        edges, probability[valid], side='right') - 1, 9)
    y, p = target[valid], probability[valid]
    return [{
        'h': horizon, 'currency': currency, 'year': year,
        'bin_left': edges[i], 'bin_right': edges[i + 1],
        'n': int((ids == i).sum()),
        'predicted': float(p[ids == i].mean()),
        'actual': float(y[ids == i].mean()),
    } for i in range(10) if (ids == i).any()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    with np.load(BASE / 'outputs.npz') as source:
        old = {key: source[key] for key in source.files}
    cap = build_outcomes(load(DATA), panel, 'publication')
    dates = panel.date.to_numpy()
    currencies = panel.currency.to_numpy()
    calibrated, priors, train_counts, logs = {}, {}, {}, []
    for h in HORIZONS:
        result = fit_quarterly_calibrator(
            old['head_probability_' + str(h)], old['y' + str(h)],
            cap['mature' + str(h)], dates, currencies)
        calibrated[h], priors[h], train_counts[h], head_logs = result
        logs.extend({'h': h, **row} for row in head_logs)
    composite = geometric_consensus(calibrated)

    scope = old['later'].astype(bool)
    metric_rows, reliability = [], []
    for h in HORIZONS:
        raw = old['head_probability_' + str(h)]
        target = old['y' + str(h)]
        metric_rows.append({
            'h': h, 'slice': 'ALL', 'group': 'ALL',
            **probability_metrics(target[scope], raw[scope],
                                  calibrated[h][scope], priors[h][scope])})
        reliability.extend(reliability_rows(
            target[scope], calibrated[h][scope], h))
        for currency in sorted(set(currencies)):
            part = scope & (currencies == currency)
            metric_rows.append({
                'h': h, 'slice': 'currency', 'group': currency,
                **probability_metrics(target[part], raw[part],
                                      calibrated[h][part], priors[h][part])})
        for year in sorted({day.year for day in dates[scope]}):
            part = scope & np.array([day.year == year for day in dates])
            metric_rows.append({
                'h': h, 'slice': 'year', 'group': str(year),
                **probability_metrics(target[part], raw[part],
                                      calibrated[h][part], priors[h][part])})
    pd.DataFrame(metric_rows).to_csv(OUT / 'calibration_metrics.csv', index=False)
    pd.DataFrame(reliability).to_csv(OUT / 'reliability_bins.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'calibration_log.csv', index=False)

    arrays = {key: old[key] for key in old}
    arrays['composite_calibrated_probability'] = composite
    for h in HORIZONS:
        arrays['calibrated_probability_' + str(h)] = calibrated[h]
        arrays['calibration_prior_' + str(h)] = priors[h]
        arrays['calibration_n_train_' + str(h)] = train_counts[h]
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    panel.to_csv(OUT / 'announcement_panel.csv', index=False)
    sources = [
        DATA, BASE / 'metadata.json', BASE / 'outputs.npz',
        Path('research/after_publication_ap50_temperature_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'AP50-T',
        'model': 'quarterly mature-only Platt calibration of AP49 heads',
        'horizons': list(HORIZONS),
        'push_candidate_unchanged': AP49_CANDIDATE,
        'historical_receipts_certified': False,
        'availability_evidence': 'calendar_assumed',
        'bank_execution_validated': False,
        'intraday_widget_validated': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(pd.DataFrame(metric_rows).query(
        "slice == 'ALL'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
