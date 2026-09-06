"""T3: regime-aware causal calibration for the 15:30-to-receipt bridge."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import (
    probability_metrics,
    reliability_rows,
)
from research.after_publication_ap50_temperature_models import (
    fit_quarterly_calibrator,
)
from research.round5_features import load_round5_features


OUT = Path('results/research/temperature/t3_phase_calibration')
BASE = Path('results/research/temperature/t2_pre_receipt_bridge')
MIN_TRAIN_DATE = dt.date(2022, 2, 24)


def loadz(path):
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_ = load_round5_features()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    scores = loadz(BASE / 'scores.npz')
    candidates = sorted(key.removeprefix('rank__') for key in scores
                        if key.startswith('rank__'))

    arrays, logs, metrics, reliability = {}, [], [], []
    for candidate in candidates:
        raw = scores['rank__' + candidate]
        arrays['raw__' + candidate] = raw
        for h in HORIZONS:
            target = targets['fav_h' + str(h)]
            maturity = target_reach_dates(index, series, h)
            calibrated, prior, count, head_logs = fit_quarterly_calibrator(
                raw, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            prefix = candidate + '__h' + str(h)
            arrays['prob__' + prefix] = calibrated
            arrays['prior__' + prefix] = prior
            arrays['n_train__' + prefix] = count
            logs.extend({'candidate': candidate, 'h': h, **row}
                        for row in head_logs)
            for period, years in (('screen_2024', (2024,)),
                                  ('opened_2025_2026', (2025, 2026))):
                scope = np.array([day.year in years for day in dates])
                metrics.append({
                    'candidate': candidate,
                    'h': h,
                    'period': period,
                    **probability_metrics(
                        target[scope], raw[scope], calibrated[scope],
                        prior[scope]),
                })
                reliability.extend({
                    'candidate': candidate, 'period': period, **row}
                    for row in reliability_rows(
                        target[scope], calibrated[scope], h))

    pd.DataFrame(metrics).to_csv(OUT / 'calibration_metrics.csv', index=False)
    pd.DataFrame(reliability).to_csv(
        OUT / 'reliability_bins.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'calibration_log.csv', index=False)
    np.savez_compressed(
        OUT / 'outputs.npz', dates=dates, currencies=currencies, **arrays)
    sources = [
        BASE / 'metadata.json', BASE / 'scores.npz',
        Path('research/temperature_t3_phase_calibration_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T3',
        'min_train_date': str(MIN_TRAIN_DATE),
        'calibration': 'quarterly Platt, mature-only, two-day embargo',
        'half_life_days': 730,
        'retrospective_methodological_repair': True,
        'tomorrow_cbr_used': False,
        'historical_receipt_certified': False,
        'physical_future_candle_corruption_inherited_from_t2': True,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    frame = pd.DataFrame(metrics)
    print(frame[frame.period == 'opened_2025_2026'].to_string(index=False),
          flush=True)


if __name__ == '__main__':
    main()
