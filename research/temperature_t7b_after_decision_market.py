"""T7B: genuinely new delayed candles after the frozen 18:30 decision."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load
from research.after_publication_ap1 import DATA
from research.after_publication_ap50_temperature import probability_metrics
from research.after_publication_ap51_benefit import magnitude_metrics
from research.after_publication_ap51_benefit_models import fit_quarterly_benefit
from research.after_publication_panel import build_outcomes
from research.round6_moex_spot_1530_features import _arrays, load_spot_1530_history
from research.temperature_t3_phase_calibration import loadz
from research.temperature_t4_premarket_models import MIN_TRAIN_DATE
from research.temperature_t7_after_receipt_market_models import (
    fit_quarterly_delta_calibrator,
)


OUT = Path('results/research/temperature/t7b_after_decision_market')
BASE = Path('results/research/after_publication/ap51_benefit')
BASE_CLOCK = dt.time(18, 30)
CLOCKS = (dt.time(19, 0), dt.time(20, 0))
DELAY = dt.timedelta(minutes=20)
HORIZONS = (3, 5, 10, 20)


def clock_name(clock):
    return f'{clock.hour:02d}{clock.minute:02d}'


def delayed_market_delta(panel, history, clock):
    item = _arrays(history)['CNYRUB_TOM']
    result = np.zeros(len(panel), dtype=float)
    available = np.zeros(len(panel), dtype=bool)
    for row, day in enumerate(panel.date):
        base = dt.datetime.combine(day, BASE_CLOCK)
        query = dt.datetime.combine(day, clock)
        begins, ends = item['begin'], item['end']
        ids = np.arange(
            int(np.searchsorted(begins, dt.datetime.combine(day, dt.time(7)),
                                side='left')),
            int(np.searchsorted(begins, query, side='left')), dtype=int)
        if not len(ids):
            continue
        nominal_end = np.array([
            begins[i] + dt.timedelta(minutes=10) for i in ids], dtype=object)
        same_day = np.array([value.date() == day for value in begins[ids]])
        base_ok = (same_day & np.array([value + DELAY < base for value in ends[ids]])
                   & np.array([value + DELAY <= base for value in nominal_end]))
        query_ok = (same_day & np.array([value + DELAY < query for value in ends[ids]])
                    & np.array([value + DELAY <= query for value in nominal_end]))
        anchor, latest = ids[base_ok], ids[query_ok]
        if not len(anchor) or not len(latest) or ends[latest[-1]] <= ends[anchor[-1]]:
            continue
        result[row] = 10000. * np.log(
            item['close'][latest[-1]] / item['close'][anchor[-1]])
        available[row] = True
    return result, available


def physical_causality_check(panel, history, clock,
                             boundary=dt.date(2025, 1, 6)):
    original = delayed_market_delta(panel, history, clock)[0]
    changed = {}
    boundary_time = dt.datetime.combine(boundary, dt.time.min)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row['end'] >= boundary_time:
                for key in ('open', 'close', 'high', 'low'):
                    clone[key] *= 9.
            changed[ticker].append(clone)
    altered = delayed_market_delta(panel, changed, clock)[0]
    past = np.array([day < boundary for day in panel.date])
    np.testing.assert_array_equal(original[past], altered[past])
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    arrays = loadz(BASE / 'outputs.npz')
    series = load(DATA)
    cap = build_outcomes(series, panel, 'publication')
    outcome = build_outcomes(series, panel, 'effective')
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    history, digest = load_spot_1530_history()
    known_h1 = (panel.current_price.to_numpy()
                <= panel.announced_price.to_numpy()).astype(float)
    known_bps_h1 = 10000. * (
        panel.announced_price.to_numpy() - panel.current_price.to_numpy()
    ) / panel.announced_price.to_numpy()
    np.testing.assert_array_equal(known_h1, outcome['y1'])
    np.testing.assert_allclose(known_bps_h1, outcome['forward1'])

    saved = {'dates': dates, 'currencies': currencies,
             'known_probability_h1': known_h1,
             'known_future_bps_h1': known_bps_h1}
    probability_rows, benefit_rows, probability_logs, benefit_logs = [], [], [], []
    for clock in CLOCKS:
        name = clock_name(clock)
        assert physical_causality_check(panel, history, clock)
        delta, available = delayed_market_delta(panel, history, clock)
        saved['delta_bps_' + name] = delta
        saved['available_' + name] = available
        for h in HORIZONS:
            target = arrays['y' + str(h)]
            probability, count, logs = fit_quarterly_delta_calibrator(
                arrays['head_probability_' + str(h)], delta, available,
                target, cap['mature' + str(h)], dates, currencies)
            prefix = name + '__h' + str(h)
            saved['probability__' + prefix] = probability
            saved['probability_n_train__' + prefix] = count
            probability_logs.extend({'clock': name, 'h': h, **row}
                                    for row in logs)
            benefit_features = np.column_stack([
                arrays['multihorizon_features'],
                arrays['calibrated_probability_' + str(h)],
                delta / 100., np.abs(delta) / 100., available.astype(float),
            ])
            prediction, prior, benefit_count, logs = fit_quarterly_benefit(
                benefit_features, arrays['forward' + str(h)],
                cap['mature' + str(h)], dates,
                min_train_date=MIN_TRAIN_DATE)
            saved['expected_bps__' + prefix] = prediction
            saved['benefit_prior__' + prefix] = prior
            saved['benefit_n_train__' + prefix] = benefit_count
            benefit_logs.extend({'clock': name, 'h': h, **row}
                                for row in logs)
            for period, years in (('screen_2024', (2024,)),
                                  ('opened_2025_2026', (2025, 2026))):
                scope = np.array([day.year in years for day in dates])
                probability_rows.append({
                    'clock': name, 'h': h, 'period': period,
                    **probability_metrics(
                        target[scope],
                        arrays['calibrated_probability_' + str(h)][scope],
                        probability[scope],
                        arrays['calibrated_probability_' + str(h)][scope]),
                })
                benefit_rows.append({
                    'clock': name, 'h': h, 'period': period,
                    **magnitude_metrics(
                        arrays['forward' + str(h)][scope], prediction[scope],
                        arrays['expected_future_bps_' + str(h)][scope]),
                })
    pd.DataFrame(probability_rows).to_csv(
        OUT / 'probability_metrics.csv', index=False)
    pd.DataFrame(benefit_rows).to_csv(OUT / 'benefit_metrics.csv', index=False)
    pd.DataFrame(probability_logs).to_csv(
        OUT / 'probability_training_log.csv', index=False)
    pd.DataFrame(benefit_logs).to_csv(
        OUT / 'benefit_training_log.csv', index=False)
    np.savez_compressed(OUT / 'outputs.npz', **saved)
    sources = [
        BASE / 'metadata.json', BASE / 'outputs.npz',
        Path('research/temperature_t7b_after_decision_market_registered.md'),
        Path('data/moex_spot_fx_10min_2022_2026.json'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T7B', 'base_clock': BASE_CLOCK.isoformat(),
        'query_clocks': [value.isoformat() for value in CLOCKS],
        'market_delay_minutes': int(DELAY.total_seconds() / 60),
        'h1_is_known_not_forecast': True,
        'historical_receipts_certified': False,
        'payload_sha256': digest, 'bank_execution_validated': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    p = pd.DataFrame(probability_rows).query("period == 'opened_2025_2026'")
    b = pd.DataFrame(benefit_rows).query("period == 'opened_2025_2026'")
    print(p.groupby('clock').agg(
        brier_base=('brier_raw', 'mean'), brier_market=('brier_calibrated', 'mean'),
        auc_base=('auc_raw', 'mean'), auc_market=('auc_calibrated', 'mean'),
        ece_base=('ece_raw', 'mean'), ece_market=('ece_calibrated', 'mean')),
          flush=True)
    print(b.groupby('clock').agg(
        mae_base=('mae_prior', 'mean'), mae_market=('mae_model', 'mean'),
        mean_spearman=('spearman', 'mean')), flush=True)


if __name__ == '__main__':
    main()
