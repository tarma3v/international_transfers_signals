"""T5: post-2022 calibration of every causal pre-publication market cutoff."""
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
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_cutoff_frontier import (
    CUTOFFS,
    cutoff_causality_check,
    cutoff_name,
    cutoff_scores,
)
from research.round6_moex_spot_1530_features import load_spot_1530_history
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE


OUT = Path('results/research/temperature/t5_market_grid')
DATA = Path('data/moex_spot_fx_10min_2022_2026.json')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_ = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    arrays = {'dates': dates, 'currencies': currencies}
    metrics, logs = [], []
    for cutoff in CUTOFFS:
        cutoff_causality_check(index, history, references, cutoff)
        candidate = cutoff_name(cutoff)
        raw = cutoff_scores(index, history, references, cutoff).astype(float)
        rank = causal_percentiles(raw, dates, currencies, 250, 20)
        arrays['raw__' + candidate] = raw
        arrays['rank__' + candidate] = rank
        for h in HORIZONS:
            target = targets['fav_h' + str(h)]
            maturity = target_reach_dates(index, series, h)
            probability, prior, count, head_logs = fit_quarterly_calibrator(
                rank, target, maturity, dates, currencies,
                min_train_date=MIN_TRAIN_DATE)
            prefix = candidate + '__h' + str(h)
            arrays['prob__' + prefix] = probability
            arrays['prior__' + prefix] = prior
            arrays['n_train__' + prefix] = count
            logs.extend({'candidate': candidate, 'h': h, **row}
                        for row in head_logs)
            for period, years in (('screen_2024', (2024,)),
                                  ('opened_2025_2026', (2025, 2026))):
                scope = np.array([day.year in years for day in dates])
                metrics.append({
                    'candidate': candidate, 'h': h, 'period': period,
                    **probability_metrics(target[scope], rank[scope],
                                          probability[scope], prior[scope]),
                })
    pd.DataFrame(metrics).to_csv(OUT / 'calibration_metrics.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'calibration_log.csv', index=False)
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    sources = [
        DATA, Path('research/temperature_t5_market_grid_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T5',
        'cutoffs': [value.isoformat() for value in CUTOFFS],
        'min_train_date': str(MIN_TRAIN_DATE),
        'payload_sha256': digest,
        'tomorrow_cbr_used': False,
        'physical_future_candle_corruption_all_cutoffs': True,
        'historical_receipt_certified': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    frame = pd.DataFrame(metrics)
    summary = frame[frame.period == 'opened_2025_2026'].groupby(
        'candidate').agg(
            mean_brier=('brier_calibrated', 'mean'),
            max_brier=('brier_calibrated', 'max'),
            mean_auc=('auc_calibrated', 'mean'),
            min_auc=('auc_calibrated', 'min'),
            mean_ece=('ece_calibrated', 'mean')).reset_index()
    summary.to_csv(OUT / 'opened_summary_by_clock.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
