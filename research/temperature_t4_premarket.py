"""T4: causal history-only premarket probability expert."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import probability_metrics
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features
from research.extended_features import LONG_DATA
from research.round6_cny_reliability_surface import causal_percentiles
from research.temperature_t4_premarket_models import (
    FEATURES,
    MIN_TRAIN_DATE,
    compact_features,
    fit_quarterly_hist,
)


OUT = Path('results/research/temperature/t4_premarket')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, names, index, series, *_ = load_round5_features()
    features = compact_features(matrix, names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    anchor = (.5 * matrix[:, names.index('pct_range_90')]
              + .3 * matrix[:, names.index('pct_range_30')]
              + .2 * matrix[:, names.index('pct_range_180')])
    anchor_probability = 1. - causal_percentiles(
        anchor, dates, currencies, 250, 20)

    arrays = {'dates': dates, 'currencies': currencies,
              'anchor_raw_probability': anchor_probability}
    metrics, model_logs, calibration_logs = [], [], []
    for h in HORIZONS:
        target = targets['fav_h' + str(h)]
        maturity = target_reach_dates(index, series, h)
        raw, count, logs = fit_quarterly_hist(
            features, target, maturity, dates)
        model_logs.extend({'h': h, **row} for row in logs)
        arrays['model_raw_probability_' + str(h)] = raw
        arrays['model_n_train_' + str(h)] = count
        for candidate, probability in (
                ('history_hist', raw), ('range_anchor', anchor_probability)):
            calibrated, prior, cal_count, logs = fit_quarterly_calibrator(
                probability, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            prefix = candidate + '__h' + str(h)
            arrays['prob__' + prefix] = calibrated
            arrays['prior__' + prefix] = prior
            arrays['calibration_n_train__' + prefix] = cal_count
            calibration_logs.extend({'candidate': candidate, 'h': h, **row}
                                    for row in logs)
            for period, years in (('screen_2024', (2024,)),
                                  ('opened_2025_2026', (2025, 2026))):
                scope = np.array([day.year in years for day in dates])
                metrics.append({
                    'candidate': candidate, 'h': h, 'period': period,
                    **probability_metrics(target[scope], probability[scope],
                                          calibrated[scope], prior[scope]),
                })

    pd.DataFrame(metrics).to_csv(OUT / 'calibration_metrics.csv', index=False)
    pd.DataFrame(model_logs).to_csv(OUT / 'model_training_log.csv', index=False)
    pd.DataFrame(calibration_logs).to_csv(
        OUT / 'calibration_log.csv', index=False)
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    sources = [
        Path('research/temperature_t4_premarket_registered.md'),
        Path('research/cache/round5_path_features.npz'),
        LONG_DATA,
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T4',
        'phase': 'overnight_to_first_market_slice',
        'features': list(FEATURES),
        'min_train_date': str(MIN_TRAIN_DATE),
        'tomorrow_cbr_used': False,
        'market_data_used': False,
        'retrospective_research': True,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    frame = pd.DataFrame(metrics)
    print(frame[frame.period == 'opened_2025_2026'].to_string(index=False),
          flush=True)


if __name__ == '__main__':
    main()
