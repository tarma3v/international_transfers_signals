"""T8B: package all validated temperature phases into one snapshot stream."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ml.transfer_temperature import score_snapshot_as_of
from research.after_publication_ap37_effective import CANDIDATE as PUSH_CANDIDATE
from research.temperature_t3_phase_calibration import loadz


OUT = Path('results/research/temperature/t8b_unified_router')
T3 = Path('results/research/temperature/t3_phase_calibration')
T4 = Path('results/research/temperature/t4_premarket')
T5 = Path('results/research/temperature/t5_market_grid')
T6 = Path('results/research/temperature/t6_pre_receipt_benefit')
T7B = Path('results/research/temperature/t7b_after_decision_market')
AP = Path('results/research/after_publication/ap51_benefit')
MOSCOW = ZoneInfo('Europe/Moscow')
HORIZONS = (1, 3, 5, 10, 20)
CUTOFFS = ('1030', '1130', '1230', '1330', '1430', '1500', '1520', '1530')


def local_timestamp(day, hhmm):
    return dt.datetime(
        day.year, day.month, day.day, int(hhmm[:2]), int(hhmm[2:]),
        tzinfo=MOSCOW).isoformat()


def _probability_columns(row, getter):
    for h in HORIZONS:
        row['probability_h' + str(h)] = float(getter(h))


def _benefit_columns(row, getter):
    for h in HORIZONS:
        row['expected_future_bps_h' + str(h)] = float(getter(h))


def pre_receipt_rows(index, t3, t4, t5, t6):
    rows = []
    dates = np.asarray([item[2] for item in index], dtype=object)
    currencies = np.asarray([item[0] for item in index])
    for i, (day, currency) in enumerate(zip(dates, currencies)):
        if day.year < 2024:
            continue
        row = {
            'currency': currency, 'valid_from': local_timestamp(day, '0000'),
            'source_at': local_timestamp(day, '0000'),
            'source_kind': 'cbr_history', 'phase': 'premarket',
            'confidence': 'limited', 'availability_evidence': 'calendar_assumed',
            'push_now': False,
        }
        _probability_columns(row, lambda h: t4[
            'prob__history_hist__h' + str(h)][i])
        _benefit_columns(row, lambda h: t6[
            'expected_bps__premarket__h' + str(h)][i])
        rows.append(row)
        for cutoff in CUTOFFS:
            state = 'cutoff_' + cutoff
            row = {
                'currency': currency,
                'valid_from': local_timestamp(day, cutoff),
                'source_at': local_timestamp(day, cutoff),
                'source_kind': 'moex_prefix', 'phase': 'market_prefix',
                'confidence': 'mature_history',
                'availability_evidence': 'completed_candles_strict_before_cutoff',
                'push_now': False,
            }
            _probability_columns(row, lambda h, state=state: t5[
                'prob__' + state + '__h' + str(h)][i])
            _benefit_columns(row, lambda h, state=state: t6[
                'expected_bps__' + state + '__h' + str(h)][i])
            rows.append(row)
        for clock in ('1630', '1730'):
            state = 'update_' + clock
            row = {
                'currency': currency,
                'valid_from': local_timestamp(day, clock),
                'source_at': local_timestamp(day, clock),
                'source_kind': 'post_window_market',
                'phase': 'pre_receipt_bridge',
                'confidence': 'mature_history',
                'availability_evidence': 'completed_candles_strict_before_cutoff',
                'push_now': False,
            }
            _probability_columns(row, lambda h, state=state: t3[
                'prob__' + state + '__h' + str(h)][i])
            _benefit_columns(row, lambda h, state=state: t6[
                'expected_bps__' + state + '__h' + str(h)][i])
            rows.append(row)
    return rows


def after_receipt_rows(panel, ap, t7b):
    rows = []
    for i, item in panel.iterrows():
        day, currency = item.date, item.currency
        if day.year < 2024:
            continue
        receipt_at = pd.Timestamp(item.decision_at).tz_convert(MOSCOW).isoformat()
        row = {
            'currency': currency, 'valid_from': receipt_at,
            'source_at': receipt_at, 'source_kind': 'cbr_receipt',
            'phase': 'after_new_cbr', 'confidence': 'mature_history',
            'availability_evidence': str(item.availability_evidence),
            'push_now': bool(ap['signal__' + PUSH_CANDIDATE][i]),
        }
        row['probability_h1'] = float(t7b['known_probability_h1'][i])
        row['expected_future_bps_h1'] = float(t7b['known_future_bps_h1'][i])
        for h in HORIZONS[1:]:
            row['probability_h' + str(h)] = float(
                ap['calibrated_probability_' + str(h)][i])
            row['expected_future_bps_h' + str(h)] = float(
                ap['expected_future_bps_' + str(h)][i])
        rows.append(row)
        for clock in ('1900', '2000'):
            stamp = local_timestamp(day, clock)
            row = {
                'currency': currency, 'valid_from': stamp, 'source_at': stamp,
                'source_kind': 'post_receipt_market',
                'phase': 'after_new_cbr_market_update',
                'confidence': 'mature_history',
                'availability_evidence': 'calendar_assumed_receipt_plus_completed_candles',
                'push_now': False,
                'probability_h1': float(t7b['known_probability_h1'][i]),
                'expected_future_bps_h1': float(t7b['known_future_bps_h1'][i]),
            }
            for h in HORIZONS[1:]:
                prefix = clock + '__h' + str(h)
                row['probability_h' + str(h)] = float(
                    t7b['probability__' + prefix][i])
                row['expected_future_bps_h' + str(h)] = float(
                    t7b['expected_bps__' + prefix][i])
            rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from research.round5_features import load_round5_features
    _X, _names, index, _series, *_ = load_round5_features()
    t3, t4, t5 = (loadz(path / 'outputs.npz') for path in (T3, T4, T5))
    t6, t7b, ap = (loadz(path / 'outputs.npz') for path in (T6, T7B, AP))
    panel = pd.read_csv(AP / 'announcement_panel.csv')
    panel.date = pd.to_datetime(panel.date).dt.date
    rows = pre_receipt_rows(index, t3, t4, t5, t6)
    rows.extend(after_receipt_rows(panel, ap, t7b))
    snapshots = pd.DataFrame(rows)
    snapshots.valid_from = pd.to_datetime(snapshots.valid_from, utc=True)
    snapshots.source_at = pd.to_datetime(snapshots.source_at, utc=True)
    snapshots = snapshots.sort_values(
        ['valid_from', 'currency', 'source_kind']).reset_index(drop=True)
    assert not snapshots.duplicated(['currency', 'valid_from']).any()
    assert (snapshots.source_at <= snapshots.valid_from).all()
    snapshots.to_csv(OUT / 'snapshots.csv.gz', index=False, compression='gzip')

    local_days = snapshots.valid_from.dt.tz_convert(MOSCOW).dt.date
    last_day = local_days.max()
    full_day = local_days[snapshots.source_kind.eq('cbr_receipt')].max()
    examples = {}
    for label, day, clock in (
            ('premarket', last_day, '0900'),
            ('intraday', last_day, '1445'),
            ('after_decision', full_day, '1845'),
            ('after_market_update', full_day, '1930'),
            ('overnight', full_day, '2300')):
        query = dt.datetime(
            day.year, day.month, day.day,
            int(clock[:2]), int(clock[2:]), tzinfo=MOSCOW)
        examples[label] = score_snapshot_as_of(
            snapshots, 'KZT', query, 5)
    weekend = last_day
    while weekend.weekday() < 5:
        weekend += dt.timedelta(days=1)
    examples['weekend'] = score_snapshot_as_of(
        snapshots, 'KZT', dt.datetime.combine(
            weekend, dt.time(12), tzinfo=MOSCOW), 5)
    (OUT / 'query_examples.json').write_text(json.dumps(
        examples, ensure_ascii=False, indent=2))
    sources = [
        *(path / 'metadata.json' for path in (T3, T4, T5, T6, T7B, AP)),
        Path('research/temperature_t8b_unified_router_registered.md'),
    ]
    (OUT / 'metadata.json').write_text(json.dumps({
        'packet': 'temperature-T8B',
        'snapshot_rows': int(len(snapshots)),
        'first_valid_from': snapshots.valid_from.min().isoformat(),
        'last_valid_from': snapshots.valid_from.max().isoformat(),
        'horizons': list(HORIZONS),
        'push_candidate': PUSH_CANDIDATE,
        'historical_receipts_certified': False,
        'bank_execution_validated': False,
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources},
    }, indent=2))
    print(json.dumps(examples, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
