"""T6: causal future-only benefit estimates for all pre-receipt states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS
from ml.validation import target_reach_dates
from research.after_publication_ap51_benefit import magnitude_metrics
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.round5_features import load_round5_features
from research.round6_uzbek_central_bank_models import _forward
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import (
    MIN_TRAIN_DATE,
    compact_features,
)


OUT = Path('results/research/temperature/t6_pre_receipt_benefit')
T3 = Path('results/research/temperature/t3_phase_calibration')
T4 = Path('results/research/temperature/t4_premarket')
T5 = Path('results/research/temperature/t5_market_grid')


def states_for_horizon(h, t3, t4, t5):
    states = {
        'premarket': (
            t4['model_raw_probability_' + str(h)],
            t4['prob__history_hist__h' + str(h)]),
    }
    for cutoff in ('1030', '1130', '1230', '1330', '1430', '1500',
                   '1520', '1530'):
        name = 'cutoff_' + cutoff
        states[name] = (
            t5['rank__' + name], t5['prob__' + name + '__h' + str(h)])
    for clock in ('1630', '1730'):
        name = 'update_' + clock
        states[name] = (
            t3['raw__' + name], t3['prob__' + name + '__h' + str(h)])
    return states


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matrix, names, index, series, *_ = load_round5_features()
    base_features = compact_features(matrix, names)
    dates = np.asarray([row[2] for row in index], dtype=object)
    t3, t4, t5 = loadz(T3 / 'outputs.npz'), loadz(
        T4 / 'outputs.npz'), loadz(T5 / 'outputs.npz')
    arrays = {'dates': dates,
              'currencies': np.asarray([row[0] for row in index])}
    metrics, logs = [], []
    for h in HORIZONS:
        target = _forward(series, index, h)
        maturity = target_reach_dates(index, series, h)
        for state, (raw, probability) in states_for_horizon(
                h, t3, t4, t5).items():
            features = np.column_stack([base_features, raw, probability])
            prediction, prior, count, head_logs = fit_quarterly_benefit(
                features, target, maturity, dates,
                min_train_date=MIN_TRAIN_DATE)
            prefix = state + '__h' + str(h)
            arrays['expected_bps__' + prefix] = prediction
            arrays['prior_bps__' + prefix] = prior
            arrays['n_train__' + prefix] = count
            logs.extend({'state': state, 'h': h, **row}
                        for row in head_logs)
            for period, years in (('screen_2024', (2024,)),
                                  ('opened_2025_2026', (2025, 2026))):
                scope = np.array([day.year in years for day in dates])
                metrics.append({
                    'state': state, 'h': h, 'period': period,
                    **magnitude_metrics(target[scope], prediction[scope],
                                        prior[scope]),
                })
    pd.DataFrame(metrics).to_csv(OUT / 'benefit_metrics.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_log.csv', index=False)
    np.savez_compressed(OUT / 'outputs.npz', **arrays)
    sources = [
        T3 / 'metadata.json', T3 / 'outputs.npz',
        T4 / 'metadata.json', T4 / 'outputs.npz',
        T5 / 'metadata.json', T5 / 'outputs.npz',
        Path('research/temperature_t6_pre_receipt_benefit_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T6',
        'states': ['premarket', 'cutoff_1030', 'cutoff_1130',
                   'cutoff_1230', 'cutoff_1330', 'cutoff_1430',
                   'cutoff_1500', 'cutoff_1520', 'cutoff_1530',
                   'update_1630', 'update_1730'],
        'horizons': list(HORIZONS),
        'min_train_date': str(MIN_TRAIN_DATE),
        'units': 'CBR future-only basis points',
        'tomorrow_cbr_used': False,
        'bank_execution_validated': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    frame = pd.DataFrame(metrics)
    opened = frame[frame.period == 'opened_2025_2026']
    print(opened.groupby('state').agg(
        mean_mae=('mae_model', 'mean'), mean_prior_mae=('mae_prior', 'mean'),
        mean_spearman=('spearman', 'mean'),
        min_sign_accuracy=('positive_sign_accuracy', 'min')).to_string(),
          flush=True)


if __name__ == '__main__':
    main()
