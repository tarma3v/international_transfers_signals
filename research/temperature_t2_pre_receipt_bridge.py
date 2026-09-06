"""T2: causal post-15:30 market correction before tomorrow-CBR receipt."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.targets import HORIZONS, build_targets
from ml.validation import target_reach_dates
from research.after_publication_ap50_temperature import probability_metrics
from research.after_publication_ap50_temperature_models import fit_quarterly_calibrator
from research.round5_adaptation import _outputs
from research.round5_features import load_round5_features
from research.round6_broad_cbr_features import load_broad_features
from research.round6_cny_reliability_surface import causal_percentiles
from research.round6_fixing_cutoff_frontier import cutoff_scores
from research.round6_moex_spot_1530_features import (
    SESSION_START,
    _arrays,
    load_spot_1530_history,
)
from research.round6_uzbek_central_bank_models import _forward, horizon_rows, summarize


OUT = Path('results/research/temperature/t2_pre_receipt_bridge')
DATA = Path('data/moex_spot_fx_10min_2022_2026.json')
ANCHOR = dt.time(15, 30)
CLOCKS = (dt.time(16, 30), dt.time(17, 30))


def clock_name(clock):
    return f'{clock.hour:02d}{clock.minute:02d}'


def post_window_delta(index, history, clock):
    item = _arrays(history)['CNYRUB_TOM']
    result = np.zeros(len(index), dtype=float)
    has_update = np.zeros(len(index), dtype=bool)
    for row, (_currency, _position, day) in enumerate(index):
        start = dt.datetime.combine(day, SESSION_START)
        anchor = dt.datetime.combine(day, ANCHOR)
        cutoff = dt.datetime.combine(day, clock)
        begins, ends = item['begin'], item['end']
        ids = np.arange(
            int(np.searchsorted(begins, start, side='left')),
            int(np.searchsorted(ends, cutoff, side='left')), dtype=int)
        if not len(ids):
            continue
        nominal_end = np.array([
            begins[i] + dt.timedelta(minutes=10) for i in ids], dtype=object)
        valid = ((begins[ids] >= start) & (ends[ids] < cutoff)
                 & (nominal_end <= cutoff))
        ids = ids[valid]
        anchor_ids = ids[(ends[ids] < anchor)
                         & (nominal_end[valid] <= anchor)]
        if not len(anchor_ids) or not len(ids):
            continue
        latest_anchor = anchor_ids[-1]
        latest = ids[-1]
        if ends[latest] <= ends[latest_anchor]:
            continue
        result[row] = 10000. * np.log(
            item['close'][latest] / item['close'][latest_anchor])
        has_update[row] = True
    return result, has_update


def score_family(index, history, references):
    anchor = cutoff_scores(index, history, references, ANCHOR).astype(float)
    scores, availability = {}, {}
    for clock in CLOCKS:
        name = clock_name(clock)
        delta, available = post_window_delta(index, history, clock)
        scores['hold_' + name] = anchor.copy()
        scores['update_' + name] = anchor + delta
        availability[name] = available
    return scores, availability


def physical_causality_check(index, history, references,
                             boundary=dt.date(2025, 1, 6)):
    original = score_family(index, history, references)[0]
    changed = {}
    boundary_time = dt.datetime.combine(boundary, ANCHOR)
    for ticker, rows in history.items():
        changed[ticker] = []
        for row in rows:
            clone = dict(row)
            if row['end'] >= boundary_time:
                for key in ('open', 'close', 'high', 'low'):
                    clone[key] *= 7.
            changed[ticker].append(clone)
    altered = score_family(index, changed, references)[0]
    past = np.array([row[2] < boundary for row in index])
    future = ~past
    for key in original:
        np.testing.assert_array_equal(original[key][past], altered[key][past])
    assert any(np.any(original[key][future] != altered[key][future])
               for key in original)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _X, _names, index, series, *_ = load_round5_features()
    _broad, _broad_names, references = load_broad_features(index, series)
    history, digest = load_spot_1530_history()
    assert physical_causality_check(index, history, references)
    dates = np.asarray([row[2] for row in index], dtype=object)
    currencies = np.asarray([row[0] for row in index])
    targets = build_targets(series, index)
    forwards = {h: _forward(series, index, h) for h in HORIZONS}
    raw, availability = score_family(index, history, references)
    ranks = {key: causal_percentiles(value, dates, currencies, 250, 20)
             for key, value in raw.items()}
    outputs = {key: _outputs(value, targets['fav_h5'], dates)
               for key, value in ranks.items()}
    rows = []
    for label, years in (('screen_2024', (2024,)),
                         ('opened_2025_2026', (2025, 2026))):
        part = horizon_rows(outputs, years, targets, forwards, dates, currencies)
        part['period'] = label
        rows.append(part)
    scorecards = pd.concat(rows, ignore_index=True)
    scorecards.to_csv(OUT / 'push_all_horizons.csv', index=False)
    summaries = []
    for period in scorecards.period.unique():
        frame = summarize(scorecards[scorecards.period == period])
        frame['period'] = period
        summaries.append(frame)
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT / 'push_summary.csv', index=False)

    later = np.array([day.year in (2025, 2026) for day in dates])
    calibration_rows, calibration_logs = [], []
    calibrated = {}
    for candidate, rank in ranks.items():
        calibrated[candidate] = {}
        for h in HORIZONS:
            reach = target_reach_dates(index, series, h)
            result = fit_quarterly_calibrator(
                rank, targets['fav_h' + str(h)], reach, dates, currencies)
            probability, prior, count, logs = result
            calibrated[candidate][h] = probability
            calibration_logs.extend({'candidate': candidate, 'h': h, **row}
                                    for row in logs)
            calibration_rows.append({
                'candidate': candidate, 'h': h,
                **probability_metrics(
                    targets['fav_h' + str(h)][later], rank[later],
                    probability[later], prior[later]),
            })
    pd.DataFrame(calibration_rows).to_csv(
        OUT / 'widget_calibration_metrics.csv', index=False)
    pd.DataFrame(calibration_logs).to_csv(
        OUT / 'widget_calibration_log.csv', index=False)
    pd.DataFrame({
        'clock': [clock_name(clock) for clock in CLOCKS],
        'new_candle_share_all': [availability[clock_name(clock)].mean()
                                 for clock in CLOCKS],
        'new_candle_share_2025_2026': [availability[clock_name(clock)][later].mean()
                                       for clock in CLOCKS],
    }).to_csv(OUT / 'availability.csv', index=False)
    np.savez_compressed(
        OUT / 'scores.npz', dates=dates, currencies=currencies,
        **{'raw__' + key: value for key, value in raw.items()},
        **{'rank__' + key: value for key, value in ranks.items()},
        **{f'prob__{key}__h{h}': value
           for key, heads in calibrated.items() for h, value in heads.items()},
        **{'available__' + key: value for key, value in availability.items()},
    )
    with (OUT / 'push_outputs.pkl').open('wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sources = [DATA, Path('research/temperature_t2_pre_receipt_bridge_registered.md')]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T2',
        'anchor_clock': ANCHOR.isoformat(),
        'query_clocks': [clock.isoformat() for clock in CLOCKS],
        'update': 'anchor plus one-for-one post-window CNY log return',
        'tomorrow_cbr_used': False,
        'historical_receipt_certified': False,
        'physical_future_candle_corruption_passed': True,
        'payload_sha256': digest,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }, indent=2))
    print(pd.read_csv(OUT / 'push_summary.csv').to_string(index=False))
    print(pd.read_csv(OUT / 'widget_calibration_metrics.csv').to_string(index=False))


if __name__ == '__main__':
    main()
